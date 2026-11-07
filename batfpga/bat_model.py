"""
bat_model.py - bit-exact integer reference model of the FPGA pipeline.

    int16 samples @256 kHz
      -> 256-pt frames (no overlap, 1 ms), Hann window (Q15 LUT)
      -> FFT, re/im scaled >>8 and rounded, mag = |re|+|im| (sat 16 bit)     [FFT CORE]
      -> per frame over bins 15..125 (1 kHz/bin): P = max mag, pk = first argmax
                                                                               [bat_features.v]
      -> call detector: active if P > 4*noise_ema and P > MIN_E;
         noise_ema += (P - noise_ema) >>> 4 on inactive frames
      -> on call end (dur >= 2 frames): 8 integer features                    [bat_features.v]
      -> decision tree -> class id + confidence (0..255)                       [bat_tree.v]

Everything after the FFT is pure integer and must match bat_features.v / bat_tree.v exactly.
"""
import numpy as np

N_FFT = 256
BIN_LO, BIN_HI = 15, 125          # inclusive, 1 kHz bins at 256 kHz / 256
MIN_E = 100                        # min peak magnitude for a call frame
EMA_SHIFT = 4
THR_SHIFT = 2                      # threshold = ema << 2  (~12 dB)
MIN_DUR = 2
MAX_DUR = 63
FEATURES = ["dur", "f_start", "f_end", "f_peak", "f_min", "f_max", "bw", "sweep"]

HANN_Q15 = np.round(32767 * np.hanning(N_FFT)).astype(np.int64)


def fft_mags(x):
    """Frames -> int magnitude array [n_frames, 128]. Stand-in for the FFT core."""
    n = len(x) // N_FFT
    fr = x[: n * N_FFT].astype(np.int64).reshape(n, N_FFT)
    xw = (fr * HANN_Q15) >> 15
    X = np.fft.rfft(xw, axis=1)[:, :N_FFT // 2]
    re = np.round(X.real / 256).astype(np.int64)
    im = np.round(X.imag / 256).astype(np.int64)
    return np.minimum(np.abs(re) + np.abs(im), 65535)


def frame_stats(mags):
    band = mags[:, BIN_LO:BIN_HI + 1]
    E = band.max(axis=1)                      # peak magnitude (detects tonal calls in noise)
    pk = BIN_LO + band.argmax(axis=1)        # argmax returns first max == strict '>' update
    return E, pk


def detect(E, pk):
    """Integer call detector. Returns list of (start_frame, end_frame_excl, features)."""
    events = []
    ema = None
    active = False
    for i in range(len(E)):
        e, p = int(E[i]), int(pk[i])
        if ema is None:
            ema = e
        act = (e > (ema << THR_SHIFT)) and (e > MIN_E)
        if act:
            if not active:
                active = True; st = i
                dur = 1; fs = fe = fmin = fmax = fpk = p; emax = e
            else:
                dur = min(dur + 1, MAX_DUR); fe = p
                fmin = min(fmin, p); fmax = max(fmax, p)
                if e > emax:
                    emax = e; fpk = p
        else:
            if active:
                active = False
                if dur >= MIN_DUR:
                    feats = [dur, fs, fe, fpk, fmin, fmax, fmax - fmin, fs - fe if fs > fe else 0]
                    events.append((st, i, feats))
            ema = ema + ((e - ema) >> EMA_SHIFT)
    return events


def extract(rec):
    """Run pipeline on one recording, label each event by overlap with true calls."""
    mags = fft_mags(rec["x"])
    E, pk = frame_stats(mags)
    out = []
    for st, en, f in detect(E, pk):
        s0, s1 = st * N_FFT, en * N_FFT
        hit = any(a < s1 and b > s0 for a, b in rec["intervals"])
        label = rec["cls"] if (hit and rec["cls"] != "NOISE") else "NOISE"
        out.append(dict(rec_id=rec["rec_id"], label=label, start=st, end=en, feats=f))
    return out
