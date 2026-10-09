"""WAV loading with strict validation.

Every metric in this package goes through `load_wav`, so this is the single
place where untrusted files are checked: existence, size, container format,
channel count, sample rate, length, and non-finite samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_SR = 48_000
# Hard limits so a hostile or corrupt header cannot make us allocate gigabytes.
MAX_FILE_BYTES = 2 * 1024**3  # 2 GiB on disk
MAX_SECONDS = 3 * 3600.0  # 3 h of audio
MIN_SECONDS = 0.05
WAV_FORMATS = {"WAV", "WAVEX", "RF64", "W64"}
# Uncompressed / 1-byte-per-sample codecs only, so header length is checkable against file size.
WAV_SUBTYPES = {"PCM_U8", "PCM_S8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE", "ULAW", "ALAW"}
MIN_SR, MAX_SR = 8_000, 384_000


class AudioError(Exception):
    """Raised for any input file we refuse to measure. Message is user-facing."""


@dataclass
class Audio:
    samples: np.ndarray  # float64, mono, at `sr`
    sr: int
    path: Path
    source_sr: int
    source_channels: int
    subtype: str
    nonfinite_samples: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr


def load_wav(
    path: str | Path,
    *,
    target_sr: int | None = TARGET_SR,
    downmix: bool = False,
) -> Audio:
    """Load a WAV file as mono float64.

    - Missing / unreadable / non-WAV / empty / oversized files raise AudioError.
    - Multichannel files raise AudioError unless `downmix=True` (then mean of channels).
    - Sample rate != target_sr is resampled (soxr VHQ) and a warning is recorded.
    - NaN/Inf samples are counted, replaced with 0, and recorded as a warning,
      so a broken processing stage is visible instead of crashing every metric.
    """
    p = Path(path)
    if not p.exists():
        raise AudioError(f"{p}: file not found")
    if not p.is_file():
        raise AudioError(f"{p}: not a regular file")
    size = p.stat().st_size
    if size == 0:
        raise AudioError(f"{p}: file is empty (0 bytes)")
    if size > MAX_FILE_BYTES:
        raise AudioError(f"{p}: file is {size / 1e9:.1f} GB, limit is {MAX_FILE_BYTES / 1e9:.1f} GB")

    try:
        info = sf.info(str(p))
    except Exception as e:  # libsndfile raises a generic RuntimeError/LibsndfileError
        raise AudioError(f"{p}: not a readable audio file ({e})") from e

    if info.format not in WAV_FORMATS:
        raise AudioError(f"{p}: container is {info.format}, expected WAV")
    if info.subtype not in WAV_SUBTYPES:
        raise AudioError(f"{p}: WAV codec {info.subtype} not supported; use PCM 16/24/32-bit or float")
    if not (MIN_SR <= info.samplerate <= MAX_SR):
        raise AudioError(f"{p}: implausible sample rate {info.samplerate} Hz")
    if info.channels < 1:
        raise AudioError(f"{p}: header reports {info.channels} channels")
    if info.channels > 1 and not downmix:
        raise AudioError(
            f"{p}: has {info.channels} channels; this pipeline is mono. "
            "Re-export as mono or pass --downmix to average channels."
        )
    if info.frames <= 0:
        raise AudioError(f"{p}: contains no audio frames")
    # Bound the header-declared length against both a time limit and the real
    # file size, so a lying header cannot trigger a huge allocation.
    if info.frames / info.samplerate > MAX_SECONDS:
        raise AudioError(f"{p}: {info.frames / info.samplerate / 3600:.1f} h exceeds limit of {MAX_SECONDS / 3600:.0f} h")
    if info.frames * info.channels > size:  # every format here is >= 1 byte/sample
        raise AudioError(f"{p}: header claims {info.frames} frames but file is only {size} bytes")

    try:
        data, sr = sf.read(str(p), dtype="float64", always_2d=True)
    except Exception as e:
        raise AudioError(f"{p}: failed to decode audio data ({e})") from e

    if data.shape[0] == 0:
        raise AudioError(f"{p}: contains no audio frames")

    warnings: list[str] = []
    x = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if data.shape[1] > 1:
        warnings.append(f"downmixed {data.shape[1]} channels to mono")

    bad = ~np.isfinite(x)
    n_bad = int(bad.sum())
    if n_bad:
        warnings.append(f"{n_bad} NaN/Inf samples replaced with 0")
        x = np.where(bad, 0.0, x)

    if len(x) / sr < MIN_SECONDS:
        raise AudioError(f"{p}: only {len(x) / sr * 1000:.1f} ms of audio, need >= {MIN_SECONDS * 1000:.0f} ms")

    out_sr = sr
    if target_sr is not None and sr != target_sr:
        x = resample(x, sr, target_sr)
        warnings.append(f"resampled {sr} Hz -> {target_sr} Hz")
        out_sr = target_sr

    return Audio(
        samples=np.ascontiguousarray(x, dtype=np.float64),
        sr=int(out_sr),
        path=p,
        source_sr=int(sr),
        source_channels=int(data.shape[1]),
        subtype=str(info.subtype),
        nonfinite_samples=n_bad,
        warnings=warnings,
    )


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    return soxr.resample(x, sr_in, sr_out, quality="VHQ")


def db(x: float, floor: float = -math.inf) -> float:
    """20*log10 for amplitudes, returning -inf (or floor) for zero."""
    if x <= 0:
        return floor
    return 20.0 * math.log10(x)


def write_wav(path: str | Path, x: np.ndarray, sr: int, subtype: str = "FLOAT") -> None:
    sf.write(str(path), np.asarray(x, dtype=np.float32 if subtype == "FLOAT" else np.float64), sr, subtype=subtype)
