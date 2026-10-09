"""Voice-identity checks on the smoke clip (skipped when testdata/ is absent).
ECAPA weights download on first run (~83 MB, pinned revision + sha256)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from conftest import SR, sine
from mic_eval.audio import load_wav, resample
from mic_eval.voice import _pyin, f0_stats, spectral_tilt, voice_check


@pytest.fixture(scope="module")
def voice(smoke):  # type: ignore[no-untyped-def]
    return load_wav(smoke / "clean.wav").samples


def test_f0_of_synthetic_harmonic_tone() -> None:
    t = np.arange(3 * SR) / SR
    x = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in range(1, 10)) * 0.1
    x[: SR // 2] = 0  # a "pause" so the speech-activity gate has a floor
    f0, speech = _pyin(x, SR)
    s = f0_stats(np.where(speech, f0, np.nan))
    assert s.median_hz == pytest.approx(120, rel=0.01)


def test_tilt_of_known_slope() -> None:
    rng = np.random.default_rng(0)
    white = rng.standard_normal(10 * SR) * 0.05
    pink_ish = sosfilt(butter(1, 50, fs=SR, output="sos"), white)  # -6 dB/oct above 50 Hz
    assert spectral_tilt(white, SR) == pytest.approx(0.0, abs=0.3)
    assert spectral_tilt(pink_ish, SR) == pytest.approx(-6.0, abs=0.5)


def test_same_file_similarity_is_one_and_passes(voice) -> None:  # type: ignore[no-untyped-def]
    v = voice_check(voice, voice.copy(), SR)
    assert v.speaker_similarity == pytest.approx(1.0, abs=1e-4)
    assert v.f0_shift_st == 0.0 and v.tilt_change_db_oct == pytest.approx(0.0, abs=1e-9)
    assert v.verdict == "PASS"


def test_benign_eq_and_gain_pass(voice) -> None:  # type: ignore[no-untyped-def]
    y = 0.5 * (sosfilt(butter(2, 80, "hp", fs=SR, output="sos"), voice))
    v = voice_check(voice, y, SR)
    assert v.verdict == "PASS", v.reasons


@pytest.mark.parametrize("semitones", [2.0, -2.0])
def test_pitch_shift_by_resampling_fails(voice, semitones: float) -> None:  # type: ignore[no-untyped-def]
    y = resample(voice, int(SR * 2 ** (semitones / 12)), SR)  # plays back 2 st higher/lower (and faster/slower)
    v = voice_check(voice, y, SR)
    assert v.verdict == "FAIL"
    assert v.f0_shift_st == pytest.approx(semitones, abs=0.2)
    assert not v.checks["f0_shift"]
    assert v.speaker_similarity is not None and v.speaker_similarity < 0.90


def test_tilt_change_fails_even_though_speaker_embedding_does_not_notice(voice) -> None:  # type: ignore[no-untyped-def]
    import librosa

    f = np.maximum(np.fft.rfftfreq(2048, 1 / SR), 100)
    y = librosa.istft(librosa.stft(voice) * ((f / 1000) ** (2 / 6.02))[:, None], length=len(voice))  # +2 dB/oct
    v = voice_check(voice, y, SR)
    assert v.tilt_change_db_oct == pytest.approx(2.0, abs=0.3)
    assert v.verdict == "FAIL" and not v.checks["tilt"]


def test_pure_tone_input_does_not_crash() -> None:
    v = voice_check(sine(440, -20, 3), sine(440, -20, 3), SR, with_speaker=False)
    assert v.verdict in ("PASS", "FAIL")
