"""
fft_model.py - bit-exact integer model of the RTL FFT core (rtl/fft256.v).

bat_model.fft_mags() is the float reference (the spec). This module is what the
hardware actually computes; it approximates fft_mags() to within a few LSB.

Algorithm (iterative radix-2 DIT, in place, one butterfly at a time):
    xw  = (x * HANN_Q15) >> 15                       same window as bat_model
    z   = xw << FRAC  (imag 0), written at bitrev8(n)
    for stage s = 0..7 (half = 2^s), butterfly (a, b) with twiddle W_k = e^{-j2pi k/256}:
        t   = (b * W) rounded >> TW_BITS              W in Q16, 18-bit signed
        a'  = (a + t + 1) >> 1,  b' = (a - t + 1) >> 1     per-stage /2 -> total /256
    re, im = (z + 2^(FRAC-1)) >> FRAC                 bins 0..127
    mag    = min(|re| + |im|, 65535)
All shifts are arithmetic (floor), as in Verilog >>> on signed values.
"""
import numpy as np
from bat_model import N_FFT, HANN_Q15

LOG2N = 8
FRAC = 4            # guard fraction bits carried through the butterflies
DW = 22             # signed data width in RAM (per component)
TW_BITS = 16        # twiddle Q16, stored as 18-bit signed

_k = np.arange(N_FFT // 2)
TW_C = np.round(np.cos(2 * np.pi * _k / N_FFT) * (1 << TW_BITS)).astype(np.int64)
TW_S = np.round(-np.sin(2 * np.pi * _k / N_FFT) * (1 << TW_BITS)).astype(np.int64)
BITREV = np.array([int(f"{i:08b}"[::-1], 2) for i in range(N_FFT)])
_LIM = 1 << (DW - 1)


def _fft_frames(xw):
    """xw: int64 [n_frames, 256] windowed samples -> (re, im) int64 [n_frames, 256] scaled /256."""
    re = np.zeros_like(xw); im = np.zeros_like(xw)
    re[:, BITREV] = xw << FRAC                 # sample n goes to address bitrev(n)
    for s in range(LOG2N):
        half = 1 << s
        n = np.arange(N_FFT // 2)
        j = n & (half - 1)
        i1 = ((n >> s) << (s + 1)) | j
        i2 = i1 + half
        k = j << (LOG2N - 1 - s)
        C, S = TW_C[k], TW_S[k]
        ar, ai, br, bi = re[:, i1], im[:, i1], re[:, i2], im[:, i2]
        rnd = 1 << (TW_BITS - 1)
        tr = (br * C - bi * S + rnd) >> TW_BITS
        ti = (br * S + bi * C + rnd) >> TW_BITS
        re[:, i1] = (ar + tr + 1) >> 1; im[:, i1] = (ai + ti + 1) >> 1
        re[:, i2] = (ar - tr + 1) >> 1; im[:, i2] = (ai - ti + 1) >> 1
        assert np.abs(re).max() < _LIM and np.abs(im).max() < _LIM, "FFT data overflow"
    rnd = 1 << (FRAC - 1)
    return (re + rnd) >> FRAC, (im + rnd) >> FRAC


def fft_mags_fixed(x):
    """Same contract as bat_model.fft_mags(): int16 audio -> int mags [n_frames, 128]."""
    n = len(x) // N_FFT
    fr = x[: n * N_FFT].astype(np.int64).reshape(n, N_FFT)
    xw = (fr * HANN_Q15) >> 15
    re, im = _fft_frames(xw)
    re, im = re[:, :N_FFT // 2], im[:, :N_FFT // 2]
    return np.minimum(np.abs(re) + np.abs(im), 65535)


if __name__ == "__main__":
    # accuracy vs the float reference
    from bat_model import fft_mags
    rng = np.random.default_rng(0)
    x = rng.integers(-32768, 32767, 256 * 200).astype(np.int16)
    t = np.arange(256 * 200) / 256e3
    x2 = (20000 * np.sin(2 * np.pi * 45e3 * t)).astype(np.int16)
    for name, sig in [("white noise", x), ("45 kHz tone", x2), ("quiet noise", (x // 300).astype(np.int16))]:
        d = fft_mags_fixed(sig) - fft_mags(sig)
        print(f"{name:12s} |fixed - float| max {np.abs(d).max()}  mean {np.abs(d).mean():.3f}")
