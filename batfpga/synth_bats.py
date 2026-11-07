"""
synth_bats.py - synthetic UK bat echolocation calls for building the pipeline before
real data arrives.  Parameters are APPROXIMATE textbook values: have your bat teammate
check/tune SPECIES below.  Swap for ChiroVox recordings later (same int16 @ FS format).

Each "recording" = one individual of one species: a short clip with several calls,
background noise, and per-individual frequency offset (so a grouped split by recording
is meaningful, like real data).
"""
import numpy as np

FS = 256_000          # sample rate (Hz). Nyquist 128 kHz covers all UK species.

# name: (start_kHz, end_kHz, dur_ms, shape, rel_amp)
#   shape: 'fmqcf' = steep FM into quasi-constant tail (pipistrelles, serotine, noctule)
#          'fm'    = steep broadband FM (Myotis, Plecotus, Barbastelle)
#          'cf'    = long constant frequency with short FM tails (horseshoes)
# ranges are (lo, hi) sampled per individual / per call
SPECIES = {
    "MYODAU": ((78, 90), (27, 33), (3, 6), "fm", 1.0),       # Daubenton's  (EBLV-2)
    "EPTSER": ((52, 65), (23, 27), (8, 14), "fmqcf", 1.0),   # Serotine     (EBLV-1)
    "PIPPIP": ((68, 80), (44, 47), (4, 7), "fmqcf", 1.0),    # Common pipistrelle
    "PIPPYG": ((78, 92), (52, 57), (4, 7), "fmqcf", 1.0),    # Soprano pipistrelle
    "PIPNAT": ((60, 70), (36, 40), (5, 8), "fmqcf", 1.0),    # Nathusius' pipistrelle
    "NYCNOC": ((26, 34), (18, 21), (15, 25), "fmqcf", 1.0),  # Noctule
    "NYCLEI": ((35, 45), (23, 27), (8, 15), "fmqcf", 1.0),   # Leisler's
    "MYONAT": ((105, 120), (20, 26), (3, 5), "fm", 0.8),     # Natterer's
    "MYOMYS": ((88, 100), (30, 35), (3, 5), "fm", 0.9),      # Whiskered
    "PLEAUR": ((50, 62), (20, 26), (2, 4), "fm", 0.3),       # Brown long-eared (quiet)
    "BARBAR": ((40, 44), (28, 33), (3, 5), "fm", 0.8),       # Barbastelle
    "RHIFER": ((80, 84), (80, 84), (40, 60), "cf", 1.0),     # Greater horseshoe (CF ~82)
    "RHIHIP": ((109, 113), (109, 113), (30, 50), "cf", 1.0), # Lesser horseshoe (CF ~111)
}
CLASSES = list(SPECIES) + ["NOISE"]
RISK = {"MYODAU": "EBLV2_reservoir", "EPTSER": "EBLV1_reservoir", "NOISE": "noise"}


def _call(rng, start, end, dur_ms, shape):
    n = int(FS * dur_ms / 1000)
    t = np.arange(n) / FS
    u = t / t[-1]
    if shape == "fm":            # slightly curved steep sweep
        f = end + (start - end) * (1 - u) ** 1.5
    elif shape == "fmqcf":       # fast drop then flat tail
        f = end + (start - end) * np.exp(-u * 6)
    else:                        # cf with short FM tails
        f = np.full(n, start, float)
        k = max(1, int(0.1 * n))
        f[:k] -= np.linspace(4, 0, k)
        f[-k:] -= np.linspace(0, 8, k)
    phase = 2 * np.pi * np.cumsum(f * 1000) / FS
    env = np.sin(np.pi * u) ** 0.7
    sig = env * np.sin(phase)
    f2 = 2 * f                    # weak 2nd harmonic where below Nyquist
    h2 = np.where(f2 < FS / 2 * 0.98, 0.15 * env * np.sin(2 * phase), 0)
    return sig + h2


def _noise(rng, n, level):
    white = rng.standard_normal(n)
    pink = np.cumsum(rng.standard_normal(n)); pink -= np.convolve(pink, np.ones(512) / 512, "same")
    return level * (white + 0.02 * pink)


def make_recording(rng, cls, dur_s=0.5):
    """Return (int16 samples, list of (start_sample, end_sample) true call intervals)."""
    n = int(FS * dur_s)
    noise_lvl = rng.uniform(40, 250)
    x = _noise(rng, n, noise_lvl)
    intervals = []
    if cls == "NOISE":
        # insect stridulation (tonal 15-40 kHz, long) + broadband clicks (rain/twigs)
        if rng.random() < 0.6:
            f0 = rng.uniform(15, 40); m = rng.integers(0, n // 2); L = rng.integers(n // 8, n // 2)
            t = np.arange(L) / FS
            am = (np.sin(2 * np.pi * rng.uniform(20, 80) * t) > 0).astype(float)
            x[m:m + L] += rng.uniform(300, 3000) * am * np.sin(2 * np.pi * f0 * 1000 * t)
        for _ in range(rng.integers(2, 12)):
            p = rng.integers(0, n - 200)
            x[p:p + 64] += rng.uniform(500, 8000) * rng.standard_normal(64) * np.hanning(64)
    else:
        (s_lo, s_hi), (e_lo, e_hi), (d_lo, d_hi), shape, rel = SPECIES[cls]
        indiv = rng.uniform(-0.04, 0.04)               # per-individual frequency offset
        amp = rel * rng.uniform(1500, 12000)           # distance to mic
        t0 = int(rng.uniform(0.005, 0.05) * FS)
        ipi = rng.uniform(0.05, 0.12) if shape != "cf" else rng.uniform(0.08, 0.15)
        while True:
            s = rng.uniform(s_lo, s_hi) * (1 + indiv); e = rng.uniform(e_lo, e_hi) * (1 + indiv)
            if shape == "cf":
                e = s
            d = rng.uniform(d_lo, d_hi)
            c = amp * rng.uniform(0.7, 1.0) * _call(rng, s, e, d, shape)
            if t0 + len(c) >= n:
                break
            x[t0:t0 + len(c)] += c
            intervals.append((t0, t0 + len(c)))
            t0 += int(ipi * FS * rng.uniform(0.85, 1.15))
    return np.clip(np.round(x), -32768, 32767).astype(np.int16), intervals


def make_dataset(n_rec_per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for ci, cls in enumerate(CLASSES):
        for r in range(n_rec_per_class):
            x, iv = make_recording(rng, cls)
            recs.append(dict(cls=cls, rec_id=f"{cls}_{r:03d}", x=x, intervals=iv))
    return recs
