"""Voice-identity preservation: speaker similarity, F0 shift, spectral tilt change.

Compares the raw take (ref) with a processed version of the SAME take (cand).
Thresholds are calibrated in eval/tools/calibrate_voice.py; see THRESHOLDS.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .audio import resample

ECAPA_DIR = Path(__file__).resolve().parents[2] / "models" / "ecapa"
ECAPA_REPO = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_REV = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"  # pinned HF commit
ECAPA_SHA256 = {
    "hyperparams.yaml": "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41",
    "embedding_model.ckpt": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
    "mean_var_norm_emb.ckpt": "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33",
    "label_encoder.txt": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
}

# Pass/fail limits (justification in calibrate_voice.py output / report).
# Calibrated on real laptop-mic and AirPods recordings (eval/tools/calibrate_voice.py):
#  f0 shift: benign processing (gain, EQ, HPF, noise, LPF) moved the median by
#    <= 0.1 st (one pYIN bin); +/-1 and +/-2 st shifts read within 0.1 st.
#    0.3 st = 3x the benign maximum, below the ~0.5 st a listener notices A/B.
#  f0 range (p10-p90): benign <= 0.3 st. 1.0 st flags prosody flattening/
#    exaggeration while leaving 3x margin.
#  tilt: benign broadcast EQ (HPF + presence, -6 dB low shelf) changed tilt by
#    0.50-0.63 dB/oct; laptop mic vs AirPods (different device) differ by ~0.6.
#    1.0 dB/oct (= 5.6 dB across 100 Hz-5 kHz) allows that, fails timbre reshaping.
#  speaker similarity (ECAPA cosine, same take): benign processing >= 0.955
#    (0.928 for +12 dB gain that hard-clipped an AirPods take); -0.5 st
#    phase-vocoder shift 0.80-0.87, +1 st 0.67-0.73, +/-2 st 0.43-0.55.
#    Same speaker on a different take/device: 0.67-0.84. 0.90 separates.
#    ECAPA is channel-invariant by design: a +2 dB/oct EQ tilt still scores
#    0.996, which is why tilt is a separate check.
THRESHOLDS: dict[str, float] = {
    "speaker_similarity_min": 0.90,
    "f0_shift_max_st": 0.3,
    "f0_range_change_max_st": 1.0,
    "tilt_change_max_db_oct": 1.0,
}


# ------------------------------------------------------------------ speaker


@lru_cache(maxsize=1)
def _ecapa():  # type: ignore[no-untyped-def]
    import logging

    logging.getLogger("speechbrain").setLevel(logging.ERROR)
    from speechbrain.inference.speaker import EncoderClassifier

    import hashlib

    if not all((ECAPA_DIR / f).exists() for f in ECAPA_SHA256):
        from huggingface_hub import hf_hub_download

        for f in ECAPA_SHA256:
            hf_hub_download(ECAPA_REPO, f, revision=ECAPA_REV, local_dir=ECAPA_DIR)
    for f, sha in ECAPA_SHA256.items():
        got = hashlib.sha256((ECAPA_DIR / f).read_bytes()).hexdigest()
        if got != sha:
            raise RuntimeError(f"{ECAPA_DIR / f}: sha256 {got} != pinned {sha}; delete it and retry")
    # SpeechBrain loads the .ckpt files with torch.load; torch >= 2.6 defaults
    # to weights_only=True, so a tampered pickle cannot execute code.
    return EncoderClassifier.from_hparams(source=str(ECAPA_DIR), savedir=str(ECAPA_DIR), run_opts={"device": "cpu"})


def speaker_embedding(x: np.ndarray, sr: int) -> np.ndarray:
    import torch

    y = resample(x, sr, 16_000).astype(np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    with torch.no_grad():
        e = _ecapa().encode_batch(torch.from_numpy(y / peak)[None, :])
    v = e.squeeze().cpu().numpy().astype(np.float64)
    return v / (np.linalg.norm(v) or 1.0)


def speaker_similarity(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Cosine similarity of ECAPA-TDNN (VoxCeleb) embeddings, 16 kHz."""
    return float(np.dot(speaker_embedding(a, sr), speaker_embedding(b, sr)))


# ------------------------------------------------------------------ F0


@dataclass
class F0Stats:
    median_hz: float
    p10_hz: float
    p90_hz: float
    range_st: float  # p90/p10 in semitones (intonation range)
    voiced_pct: float


F0_MIN, F0_MAX = 60.0, 500.0


def _pyin(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """pYIN F0 per 10 ms frame (NaN = unvoiced) and a speech-activity mask.

    Hardened against failure modes seen on real recordings:
      - raw laptop-mic files: low-frequency rumble tracked as "voiced" and pinned
        at fmin (60 Hz) in ~20 % of frames -> identical 4th-order 65 Hz
        high-pass on every input and drop estimates within 2 % of the limits;
      - only frames whose 300-3400 Hz energy is > p10 + 15 dB count (speech).
        For a same-take comparison the caller applies the REF mask to both,
        because a denoiser's silent pauses move the cand's own p10 by ~80 dB
        and would otherwise admit breaths/residue as "speech"."""
    import librosa
    from scipy.signal import butter, sosfilt

    y = resample(x, sr, 16_000)
    y = sosfilt(butter(4, 65, "hp", fs=16_000, output="sos"), y)
    hop = 160
    f0, voiced, _p = librosa.pyin(y, fmin=F0_MIN, fmax=F0_MAX, sr=16_000, frame_length=1024, hop_length=hop)
    band = sosfilt(butter(4, [300, 3400], "bp", fs=16_000, output="sos"), y)
    n = len(f0)
    padded = np.pad(band, (512, 512 + n * hop))
    e = np.array([np.mean(padded[i * hop : i * hop + 1024] ** 2) for i in range(n)])
    e_db = 10 * np.log10(e + 1e-20)
    speech = e_db > np.percentile(e_db, 10) + 15
    ok = voiced & (f0 > F0_MIN * 1.02) & (f0 < F0_MAX * 0.98)
    return np.where(ok, f0, np.nan), speech


def f0_stats(f0: np.ndarray) -> F0Stats:
    v = f0[np.isfinite(f0)]
    if len(v) < 10:
        return F0Stats(math.nan, math.nan, math.nan, math.nan, 100.0 * len(v) / max(1, len(f0)))
    p10, med, p90 = np.percentile(v, [10, 50, 90])
    return F0Stats(float(med), float(p10), float(p90), float(12 * np.log2(p90 / p10)), 100.0 * len(v) / len(f0))


# ------------------------------------------------------------------ tilt


def spectral_tilt(x: np.ndarray, sr: int, fmin: float = 100.0, fmax: float = 5000.0) -> float:
    """Long-term spectral slope of speech-active frames, dB/octave.

    Average power spectrum of frames > p10 + 15 dB, summed into 1/3-octave
    bands from fmin to fmax (so each octave gets equal weight), then a
    least-squares line of band level vs log2(frequency).
    fmax=5 kHz keeps the fit inside every device's bandwidth (BT 8 kHz,
    AirPods 11.5 kHz), so bandwidth extension does not register as tilt.
    """
    nfft = 2048
    hop = 1024
    if len(x) < nfft * 4:
        return math.nan
    idx = np.arange(0, len(x) - nfft + 1, hop)
    fr = np.lib.stride_tricks.sliding_window_view(x, nfft)[idx] * np.hanning(nfft)
    e = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-30)
    act = e > np.percentile(e, 10) + 15
    if act.sum() < 4:
        act = e >= np.percentile(e, 70)
    P = np.mean(np.abs(np.fft.rfft(fr[act], axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    centers = fmin * 2 ** (np.arange(0, np.log2(fmax / fmin) + 1e-9, 1 / 3))
    lv, lc = [], []
    for c in centers:
        m = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        if m.any():
            lv.append(10 * np.log10(P[m].mean() + 1e-30))
            lc.append(np.log2(c))
    if len(lv) < 4:
        return math.nan
    slope, _ = np.polyfit(np.array(lc), np.array(lv), 1)
    return float(slope)


# ------------------------------------------------------------------ verdict


@dataclass
class VoiceResult:
    speaker_similarity: float | None
    f0_ref: F0Stats
    f0_cand: F0Stats
    f0_shift_st: float  # 12*log2(median F0 cand / median F0 ref), pYIN voiced frames
    f0_range_change_st: float
    tilt_ref_db_oct: float
    tilt_cand_db_oct: float
    tilt_change_db_oct: float
    checks: dict[str, bool]
    verdict: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def voice_check(ref: np.ndarray, cand: np.ndarray, sr: int, *, with_speaker: bool = True) -> VoiceResult:
    from .intrusive import align

    same_take = abs(len(ref) - len(cand)) <= sr
    if same_take:  # align/crop; else (e.g. speed-changed) compare whole files
        r, c, _lag = align(ref, cand, sr)
    else:
        r, c = ref, cand
    # Distribution statistics, not frame-by-frame, so a time-stretched or
    # resampled (speed-changed) candidate is still measured correctly.
    f0r, mr = _pyin(r, sr)
    f0c, mc = _pyin(c, sr)
    if same_take:
        n = min(len(f0r), len(f0c))
        f0r, f0c, mr, mc = f0r[:n], f0c[:n], mr[:n], mr[:n]
    sr_, sc_ = f0_stats(np.where(mr, f0r, np.nan)), f0_stats(np.where(mc, f0c, np.nan))
    shift = float(12 * np.log2(sc_.median_hz / sr_.median_hz))
    rng = sc_.range_st - sr_.range_st
    tr, tc = spectral_tilt(r, sr), spectral_tilt(c, sr)
    dt = tc - tr
    sim = speaker_similarity(r, c, sr) if with_speaker else None

    T = THRESHOLDS
    checks: dict[str, bool] = {
        "f0_shift": bool(math.isfinite(shift) and abs(shift) <= T["f0_shift_max_st"]),
        "f0_range": bool(math.isfinite(rng) and abs(rng) <= T["f0_range_change_max_st"]),
        "tilt": bool(math.isfinite(dt) and abs(dt) <= T["tilt_change_max_db_oct"]),
    }
    if sim is not None:
        checks["speaker_similarity"] = sim >= T["speaker_similarity_min"]
    reasons = []
    if not checks["f0_shift"]:
        reasons.append(f"F0 shifted {shift:+.2f} st (limit +/-{T['f0_shift_max_st']})")
    if not checks["f0_range"]:
        reasons.append(f"F0 range changed {rng:+.2f} st (limit +/-{T['f0_range_change_max_st']})")
    if not checks["tilt"]:
        reasons.append(f"spectral tilt changed {dt:+.2f} dB/oct (limit +/-{T['tilt_change_max_db_oct']})")
    if sim is not None and not checks["speaker_similarity"]:
        reasons.append(f"speaker similarity {sim:.3f} < {T['speaker_similarity_min']}")
    return VoiceResult(sim, sr_, sc_, shift, rng, tr, tc, dt, checks, "FAIL" if reasons else "PASS", reasons)
