"""Intrusive (full-reference) metrics: PESQ-WB, SI-SDR, STOI/ESTOI.

Only meaningful when a *clean* reference of the same take exists (synthetic
tests, e.g. testdata/smoke/clean.wav). Never use the raw noisy recording as
the reference for these: that rewards doing nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags

from .audio import resample

MAX_LAG_S = 1.0


@dataclass
class IntrusiveScores:
    lag_samples: int  # how far `deg` was shifted to align with `ref` (at 48 kHz)
    pesq_wb: float | None  # ITU-T P.862.2, 16 kHz, MOS-LQO 1.04..4.64
    si_sdr_db: float
    stoi: float | None  # 0..1
    estoi: float | None
    note: str = ""


def align(ref: np.ndarray, deg: np.ndarray, sr: int, max_lag_s: float = MAX_LAG_S) -> tuple[np.ndarray, np.ndarray, int]:
    """Shift `deg` by the cross-correlation peak (|lag| <= max_lag_s) and crop
    both to the common length. Processing chains add latency; SI-SDR and STOI
    are not delay-invariant, so without this a 1 ms delay looks like garbage."""
    c = correlate(deg, ref, mode="full", method="fft")
    lags = correlation_lags(len(deg), len(ref), mode="full")
    lim = int(max_lag_s * sr)
    m = np.abs(lags) <= lim
    lag = int(lags[m][np.argmax(np.abs(c[m]))])
    if lag > 0:
        d = deg[lag:]
        r = ref
    else:
        d = deg
        r = ref[-lag:]
    L = min(len(r), len(d))
    return r[:L], d[:L], lag


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR (Le Roux et al. 2019), zero-mean, in dB."""
    ref = ref - ref.mean()
    est = est - est.mean()
    rr = float(np.dot(ref, ref))
    if rr == 0.0:
        return math.nan
    alpha = float(np.dot(est, ref)) / rr
    target = alpha * ref
    noise = est - target
    nn = float(np.dot(noise, noise))
    tt = float(np.dot(target, target))
    if nn == 0.0:
        return math.inf
    if tt == 0.0:
        return -math.inf
    return 10.0 * math.log10(tt / nn)


def intrusive(ref: np.ndarray, deg: np.ndarray, sr: int) -> IntrusiveScores:
    from pesq import pesq as _pesq
    from pesq import PesqError
    from pystoi import stoi as _stoi

    r, d, lag = align(ref, deg, sr)
    notes: list[str] = []
    sdr = si_sdr(r, d)

    r16 = resample(r, sr, 16_000)
    d16 = resample(d, sr, 16_000)
    p: float | None
    try:
        p = float(_pesq(16_000, r16, d16, "wb"))
    except PesqError as e:  # e.g. no utterance detected
        p = None
        notes.append(f"PESQ failed: {type(e).__name__}")
    try:
        s = float(_stoi(r16, d16, 16_000, extended=False))
        es = float(_stoi(r16, d16, 16_000, extended=True))
    except Exception as e:  # pystoi raises on too-short input
        s = es = None
        notes.append(f"STOI failed: {e}")
    return IntrusiveScores(lag, p, sdr, s, es, "; ".join(notes))
