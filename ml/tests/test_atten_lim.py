import numpy as np
import pytest

from mlstage.dsp import auto_atten_lim_db, speech_snr_db

SR = 48_000


@pytest.mark.parametrize("snr,want", [(0, 45), (20, 45), (23, 45), (24.3, 43.7), (33.8, 34.2), (40, 28), (48, 20), (60, 20)])
def test_clamp(snr, want):
    assert auto_atten_lim_db(snr) == pytest.approx(want)


def test_unknown_snr_falls_back_to_30():
    assert auto_atten_lim_db(float("nan")) == 30.0
    assert np.isnan(speech_snr_db(np.zeros(SR), SR))  # all digital zero -> undefined


def _speech_plus_noise(snr_db, seconds=10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(seconds * SR) / SR
    on = (t % 1.0) < 0.5  # 50 % "speech" bursts
    speech = 0.1 * np.sin(2 * np.pi * 220 * t) * on
    p_speech = np.mean(speech[on] ** 2)
    noise = rng.standard_normal(len(t)) * np.sqrt(p_speech / 10 ** (snr_db / 10))
    return (speech + noise).astype(np.float32), on


def test_snr_estimate():
    x, _ = _speech_plus_noise(25)
    assert speech_snr_db(x, SR) == pytest.approx(25, abs=1.5)


def test_exact_zero_frames_are_ignored():
    """AirPods-style gating: silences replaced by exact zeros must not inflate the SNR."""
    x, on = _speech_plus_noise(25)
    gated = x.copy()
    gated[: 2 * SR] = 0.0  # 2 s of on-device gating (exact digital zeros)
    assert speech_snr_db(gated, SR) == pytest.approx(speech_snr_db(x[2 * SR:], SR), abs=0.5)
    assert np.isfinite(speech_snr_db(gated, SR))
