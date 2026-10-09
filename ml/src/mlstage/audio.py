"""WAV I/O, resampling and level helpers shared by every backend.

Interchange format for the whole project: WAV, 48 kHz, mono, float32.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

OUT_SR = 48_000


class AudioError(RuntimeError):
    pass


def read_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read any WAV soundfile can parse (PCM 8/16/24/32, float, extensible), downmix to mono f32."""
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"input file not found: {path}")
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # soundfile raises LibsndfileError / RuntimeError subclasses
        raise AudioError(f"cannot read {path}: {e}") from e
    if x.shape[0] == 0:
        raise AudioError(f"input has no samples: {path}")
    x = x.mean(axis=1, dtype=np.float32) if x.shape[1] > 1 else x[:, 0]
    if not np.all(np.isfinite(x)):
        raise AudioError(f"input contains NaN/Inf samples: {path}")
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def write_48k_f32(path: str | Path, x: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), OUT_SR, subtype="FLOAT")


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear-phase, delay-compensated resampling (soxr VHQ). Output length = ceil(n * ratio)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    return soxr.resample(x, sr_in, sr_out, quality="VHQ").astype(np.float32, copy=False)


def fix_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def out_len(n_in: int, sr_in: int, sr_out: int) -> int:
    return int(round(n_in * sr_out / sr_in))
