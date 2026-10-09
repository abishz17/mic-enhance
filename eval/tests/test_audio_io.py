"""Loader behaviour on bad / hostile input: every case is a clear AudioError or a sane load."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, sine
from mic_eval.audio import AudioError, load_wav
from mic_eval.signal_metrics import integrated_lufs


def _wav_header(n_channels: int, sr: int, bits: int, data_bytes: int, fmt_tag: int = 1) -> bytes:
    block = n_channels * bits // 8
    fmt = struct.pack("<HHIIHH", fmt_tag, n_channels, sr, sr * block, block, bits)
    return b"RIFF" + struct.pack("<I", min(36 + data_bytes, 0xFFFFFFFF)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", data_bytes)


def test_missing_file() -> None:
    with pytest.raises(AudioError, match="not found"):
        load_wav("/nonexistent/x.wav")


def test_directory(tmp_path: Path) -> None:
    with pytest.raises(AudioError, match="not a regular file"):
        load_wav(tmp_path)


def test_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "e.wav"
    p.write_bytes(b"")
    with pytest.raises(AudioError, match="empty"):
        load_wav(p)


def test_text_file_named_wav(tmp_path: Path) -> None:
    p = tmp_path / "t.wav"
    p.write_text("hello, not audio")
    with pytest.raises(AudioError, match="not a readable audio file"):
        load_wav(p)


def test_flac_named_wav(tmp_path: Path) -> None:
    p = tmp_path / "f.wav"
    sf.write(p, sine(440, -6, 1), SR, format="FLAC")
    with pytest.raises(AudioError, match="expected WAV"):
        load_wav(p)


def test_header_only_zero_frames(tmp_path: Path) -> None:
    p = tmp_path / "z.wav"
    p.write_bytes(_wav_header(1, SR, 16, 0))
    with pytest.raises(AudioError):
        load_wav(p)


def test_header_lies_about_huge_data_chunk(tmp_path: Path) -> None:
    # claims ~4 GB of data, actually 1 s: must load the 1 s (or refuse), never allocate 4 GB
    pcm = (sine(440, -6, 1) * 32767).astype("<i2").tobytes()
    p = tmp_path / "h.wav"
    p.write_bytes(_wav_header(1, SR, 16, 0xFFFFFFF0) + pcm)
    a = load_wav(p)
    assert a.duration == pytest.approx(1.0, abs=0.01)


def test_absurd_sample_rate(tmp_path: Path) -> None:
    p = tmp_path / "sr.wav"
    pcm = np.zeros(1000, "<i2").tobytes()
    p.write_bytes(_wav_header(1, 2_000_000_000, 16, len(pcm)) + pcm)
    with pytest.raises(AudioError):
        load_wav(p)


def test_too_short(tmp_path: Path) -> None:
    p = tmp_path / "s.wav"
    sf.write(p, np.zeros(100), SR)
    with pytest.raises(AudioError, match="ms of audio"):
        load_wav(p)


def test_stereo_refused_unless_downmix(tmp_path: Path) -> None:
    p = tmp_path / "st.wav"
    x = sine(440, -6, 1)
    sf.write(p, np.stack([x, x], axis=1), SR, subtype="PCM_24")
    with pytest.raises(AudioError, match="2 channels"):
        load_wav(p)
    a = load_wav(p, downmix=True)
    assert a.samples.ndim == 1 and a.source_channels == 2
    assert np.max(np.abs(a.samples)) == pytest.approx(10 ** (-6 / 20), abs=1e-3)


def test_wrong_rate_is_resampled_and_level_preserved(tmp_path: Path) -> None:
    p = tmp_path / "r.wav"
    sf.write(p, sine(997, -20, 5, sr=44_100), 44_100, subtype="PCM_24")
    a = load_wav(p)
    assert a.sr == SR and a.source_sr == 44_100
    assert any("resampled" in w for w in a.warnings)
    assert integrated_lufs(a.samples, a.sr) == pytest.approx(-23.01, abs=0.05)


def test_nan_and_inf_samples_are_counted_and_zeroed(tmp_path: Path) -> None:
    x = sine(440, -6, 1).astype(np.float32)
    x[100:110] = np.nan
    x[200] = np.inf
    p = tmp_path / "n.wav"
    sf.write(p, x, SR, subtype="FLOAT")
    a = load_wav(p)
    assert a.nonfinite_samples == 11
    assert np.all(np.isfinite(a.samples))


def test_24bit_pcm_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "p.wav"
    x = sine(440, -6, 1)
    sf.write(p, x, SR, subtype="PCM_24")
    assert np.max(np.abs(load_wav(p).samples - x)) < 2e-7
