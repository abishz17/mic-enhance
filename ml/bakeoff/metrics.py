"""Bake-off metrics. Everything here runs locally; no audio leaves the machine.

Reference-free: DNSMOS P.835 (SIG/BAK/OVRL, Microsoft ONNX via `speechmos`, 16 kHz), effective
bandwidth, leading-silence energy ("babble"), clipping, timing lag vs input, F0 shift vs input,
speaker-embedding cosine vs input (ECAPA, speechbrain), Whisper WER vs a reference transcript.
Intrusive (only where a clean reference exists): PESQ-WB (16 kHz), SI-SDR.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlstage.dsp import effective_bandwidth  # noqa: E402

__all__ = ["effective_bandwidth", "dnsmos", "pesq_wb", "si_sdr", "lag_ms", "lead_energy_db", "clip_frac",
           "f0_stats", "f0_summary", "f0_shift_cents", "spectral_tilt", "band_ratio_db", "spk_emb", "spk_cos",
           "transcribe", "wer", "norm_text", "rs"]


def rs(x, a, b):
    return x if a == b else soxr.resample(x, a, b, quality="VHQ").astype(np.float32)


def dnsmos(x48: np.ndarray) -> dict:
    from speechmos import dnsmos as d

    x16 = np.clip(rs(x48, 48_000, 16_000), -1, 1)
    r = d.run(x16, sr=16_000)
    return {"SIG": r["sig_mos"], "BAK": r["bak_mos"], "OVRL": r["ovrl_mos"]}


def pesq_wb(ref48, deg48) -> float:
    from pesq import pesq

    n = min(len(ref48), len(deg48))
    try:
        return float(pesq(16_000, rs(ref48[:n], 48_000, 16_000), rs(deg48[:n], 48_000, 16_000), "wb"))
    except Exception:  # noqa: BLE001 - pesq raises on silent/no-utterance input
        return float("nan")


def si_sdr(ref, est) -> float:
    n = min(len(ref), len(est))
    r, e = ref[:n].astype(np.float64), est[:n].astype(np.float64)
    r, e = r - r.mean(), e - e.mean()
    a = np.dot(e, r) / (np.dot(r, r) + 1e-12)
    t = a * r
    return float(10 * np.log10(np.sum(t ** 2) / (np.sum((e - t) ** 2) + 1e-12) + 1e-12))


def lag_ms(ref48, out48, max_ms=50.0, band=(100, 7000)) -> float:
    """Timing offset of out vs ref (positive = out is late), from band-limited cross-correlation."""
    import scipy.signal as ss

    sos = ss.butter(4, band, btype="band", fs=48_000, output="sos")
    a = ss.sosfiltfilt(sos, ref48.astype(np.float64))
    b = ss.sosfiltfilt(sos, out48.astype(np.float64))
    n = min(len(a), len(b))
    c = ss.correlate(b[:n], a[:n], mode="full", method="fft")
    mid = n - 1
    m = int(max_ms * 48)
    seg = c[mid - m: mid + m + 1]
    k = int(np.argmax(np.abs(seg)))
    # parabolic interpolation for sub-sample precision
    if 0 < k < len(seg) - 1:
        y0, y1, y2 = np.abs(seg[k - 1: k + 2])
        k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-20)
    return float((k - m) / 48.0)


def lead_energy_db(x48, seconds: float) -> float:
    """RMS level (dBFS) of the leading `seconds` (room tone / silence)."""
    s = x48[: int(seconds * 48_000)]
    return float(10 * np.log10(np.mean(s.astype(np.float64) ** 2) + 1e-20))


def clip_frac(x48, thr=0.999) -> float:
    return float(np.mean(np.abs(x48) >= thr))


def f0_stats(x48) -> np.ndarray:
    """Frame-wise F0 (Hz, NaN where unvoiced), pYIN on 16 kHz, 10 ms hop."""
    import librosa

    x16 = rs(x48, 48_000, 16_000)
    f0, vflag, _ = librosa.pyin(x16, fmin=60, fmax=400, sr=16_000, frame_length=1024, hop_length=160)
    f0[~vflag] = np.nan
    return f0


def f0_summary(f0) -> tuple[float, float]:
    """(median F0 Hz, range in semitones between 5th and 95th percentile) over voiced frames."""
    v = f0[~np.isnan(f0)]
    if len(v) < 10:
        return float("nan"), float("nan")
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    return float(p50), float(12 * np.log2(p95 / p5))


def spectral_tilt(x48, fmax: float = 8000.0) -> float:
    """Long-term spectral slope (dB/octave) of speech-active frames, 100 Hz..fmax, by a
    least-squares line through the 1/3-octave band levels vs log2(f)."""
    import scipy.signal as ss

    f, t, Z = ss.stft(x48, 48_000, nperseg=2048, noverlap=1536)
    P = np.abs(Z) ** 2
    e = P.sum(axis=0)
    act = e > np.percentile(e, 70)
    S = P[:, act].mean(axis=1)
    centers = 100 * 2 ** (np.arange(0, np.log2(fmax / 100) + 1e-9, 1 / 3))
    lv, lf = [], []
    for c in centers:
        m = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        if m.any():
            lv.append(10 * np.log10(S[m].mean() + 1e-20))
            lf.append(np.log2(c))
    return float(np.polyfit(lf, lv, 1)[0])


def band_ratio_db(x48, num=(2000, 5000), den=(500, 2000)) -> float:
    """Level of `num` band minus `den` band (dB) over speech-active frames ("presence" vs body)."""
    import scipy.signal as ss

    f, t, Z = ss.stft(x48, 48_000, nperseg=2048, noverlap=1536)
    P = np.abs(Z) ** 2
    e = P.sum(axis=0)
    S = P[:, e > np.percentile(e, 70)].mean(axis=1)
    a = S[(f >= num[0]) & (f < num[1])].sum()
    b = S[(f >= den[0]) & (f < den[1])].sum()
    return float(10 * np.log10((a + 1e-20) / (b + 1e-20)))


def f0_shift_cents(f0_ref, f0_out) -> tuple[float, float]:
    """(median cents difference over frames voiced in both, fraction of ref-voiced frames lost)."""
    n = min(len(f0_ref), len(f0_out))
    a, b = f0_ref[:n], f0_out[:n]
    both = ~np.isnan(a) & ~np.isnan(b)
    refv = ~np.isnan(a)
    if both.sum() < 10:
        return float("nan"), float("nan")
    cents = 1200 * np.log2(b[both] / a[both])
    lost = 1 - both.sum() / max(1, refv.sum())
    return float(np.median(cents)), float(lost)


@lru_cache(maxsize=1)
def _ecapa():
    import torch  # noqa: F401
    from speechbrain.inference.speaker import EncoderClassifier

    # speechbrain/spkrec-ecapa-voxceleb @ 0f99f2d0, files sha256-verified by the download step
    local = Path(__file__).resolve().parents[1] / "models" / "ecapa"
    return EncoderClassifier.from_hparams(source=str(local), savedir=str(local), run_opts={"device": "cpu"})


def spk_emb(x48) -> np.ndarray:
    import torch

    x16 = rs(x48, 48_000, 16_000)
    with torch.inference_mode():
        e = _ecapa().encode_batch(torch.from_numpy(x16)[None]).reshape(-1).numpy()
    return e / (np.linalg.norm(e) + 1e-12)


def spk_cos(a48, b48) -> float:
    return float(np.dot(spk_emb(a48), spk_emb(b48)))


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel

    # Systran/faster-whisper-small.en @ d1d751a5 (CTranslate2), sha256 of model.bin verified:
    # 62b2a45b05ee59acb4a5341b33ee35e041395d378d418a18acfe4c9e768ee37a
    path = Path(__file__).resolve().parents[1] / "models" / "faster-whisper-small.en"
    return WhisperModel(str(path), device="cpu", compute_type="int8")


def transcribe(x48) -> str:
    """Deterministic decode, no VAD (VAD would hide babble/hallucinated words in silences)."""
    x16 = rs(x48, 48_000, 16_000)
    segs, _ = _whisper().transcribe(x16, language="en", beam_size=5, temperature=0.0,
                                    condition_on_previous_text=False, vad_filter=False)
    return " ".join(s.text.strip() for s in segs).strip()


_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
        "8": "eight", "9": "nine", "10": "ten"}


def norm_text(t: str) -> str:
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = " ".join(_NUM.get(w, w) for w in t.split())
    return t


def wer(ref: str, hyp: str) -> dict:
    import jiwer

    r, h = norm_text(ref), norm_text(hyp)
    if not r:
        return {"wer": float("nan"), "ins": 0, "del": 0, "sub": 0}
    o = jiwer.process_words(r, h)
    return {"wer": o.wer, "ins": o.insertions, "del": o.deletions, "sub": o.substitutions}


def lsd_band(ref48, est48, lo: float, hi: float) -> float:
    """Log-spectral distance (dB) between reference and estimate in [lo, hi) Hz over the reference's
    speech-active frames. The standard bandwidth-extension accuracy metric (lower = closer)."""
    import scipy.signal as ss

    n = min(len(ref48), len(est48))
    f, t, R = ss.stft(ref48[:n], 48_000, nperseg=2048, noverlap=1536)
    _, _, E = ss.stft(est48[:n], 48_000, nperseg=2048, noverlap=1536)
    pr, pe = np.abs(R) ** 2, np.abs(E) ** 2
    act = pr.sum(axis=0) > np.percentile(pr.sum(axis=0), 50)
    b = (f >= lo) & (f < hi)
    d = 10 * np.log10(pr[b][:, act] + 1e-10) - 10 * np.log10(pe[b][:, act] + 1e-10)
    return float(np.mean(np.sqrt(np.mean(d ** 2, axis=0))))


def f0_compare(f0_a, f0_b) -> dict:
    """Pitch change from a to b over frames voiced in both (pYIN, 10 ms hop).
    median_cents: overall pitch shift; p90_abs_cents: how big local deviations get;
    octave_frac: frames off by > 6 semitones (tracker octave jumps / pitch artifacts);
    range_delta_st: change of the 10th-90th percentile F0 span (intonation flattening/expansion);
    voiced_lost: share of a's voiced frames no longer voiced in b (breathy/whispered or removed)."""
    n = min(len(f0_a), len(f0_b))
    a, b = f0_a[:n], f0_b[:n]
    both = ~np.isnan(a) & ~np.isnan(b)
    nan = float("nan")
    if both.sum() < 20:
        return {"f0_shift_cents": nan, "f0_p90_abs_cents": nan, "f0_octave_frac": nan, "f0_range_delta_st": nan,
                "f0_voiced_lost": nan}
    c = 1200 * np.log2(b[both] / a[both])
    ok = np.abs(c) < 600
    span = lambda v: 12 * np.log2(np.percentile(v, 90) / np.percentile(v, 10))  # noqa: E731
    return {"f0_shift_cents": float(np.median(c)),
            "f0_p90_abs_cents": float(np.percentile(np.abs(c[ok]), 90)) if ok.any() else nan,
            "f0_octave_frac": float(1 - ok.mean()),
            "f0_range_delta_st": float(span(b[both][ok]) - span(a[both][ok])) if ok.sum() > 20 else nan,
            "f0_voiced_lost": float(1 - both.sum() / max(1, (~np.isnan(a)).sum()))}
