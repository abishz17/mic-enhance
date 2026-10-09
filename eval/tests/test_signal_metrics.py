from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, firwin, lfilter, sosfilt

from conftest import SR, sine
from mic_eval.audio import load_wav
from mic_eval.signal_metrics import clipping, effective_bandwidth, noise_stats


# ---------------------------------------------------------------- clipping


def test_clipped_sine_detected() -> None:
    x = np.clip(sine(1000, 6.0, 1.0), -1.0, 1.0)  # +6 dB sine hard-clipped at full scale
    c = clipping(x)
    expected = int(np.sum(np.abs(sine(1000, 6.0, 1.0)) >= 1.0))
    assert c.clipped_samples == pytest.approx(expected, rel=0.05)
    assert c.clip_runs >= 1000 * 2 * 0.95  # two flat tops per cycle


def test_clipping_below_full_scale_detected() -> None:
    # preamp clipped at 0.5 then attenuated: plateau is still the file's peak
    x = np.clip(sine(200, 0.0, 1.0), -0.5, 0.5) * 0.3
    assert clipping(x).clip_runs > 300


@pytest.mark.parametrize("freq", [50.0, 1000.0])
def test_clean_sine_not_flagged(freq: float) -> None:
    for x in (sine(freq, -1.0, 2.0), np.round(sine(freq, -1.0, 2.0) * 8388607) / 8388607):
        assert clipping(x).clipped_samples == 0


def test_float_over_full_scale_counted() -> None:
    x = sine(1000, 3.0, 0.1)
    assert clipping(x).over_fs_samples > 0


def test_smoke_files_not_clipped(smoke) -> None:  # type: ignore[no-untyped-def]
    for f in smoke.glob("*.wav"):
        assert clipping(load_wav(f).samples).clipped_samples == 0, f.name


# ---------------------------------------------------------------- noise floor


def _speechlike(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Noise bursts (syllables) with pauses, ~-20 dBFS RMS while active."""
    n = rng.standard_normal(int(seconds * SR)) * 0.1
    t = np.arange(len(n)) / SR
    env = (np.sin(2 * np.pi * 1.5 * t) > 0.2).astype(float)
    return n * env


def test_noise_floor_matches_known_level() -> None:
    rng = np.random.default_rng(1)
    floor = rng.standard_normal(20 * SR) * 10 ** (-60 / 20)  # -60 dBFS RMS white noise
    ns = noise_stats(_speechlike(20, rng) + floor, SR, lead_silence_s=None)
    assert ns.noise_floor_dbfs == pytest.approx(-60.0, abs=1.0)
    assert ns.snr_db == pytest.approx(40.0, abs=3.0)


def test_lead_silence_is_median_of_room_tone() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(10 * SR) * 10 ** (-50 / 20)
    x[1000:1100] = 0.9  # click in the room tone must not move the median
    assert noise_stats(x, SR, lead_silence_s=4.0).lead_noise_dbfs == pytest.approx(-50.0, abs=0.5)


def test_digital_silence_floor_is_minus_inf_not_nan() -> None:
    x = np.concatenate([np.zeros(5 * SR), sine(300, -10, 5)])
    ns = noise_stats(x, SR, lead_silence_s=4.0)
    assert ns.noise_floor_dbfs == -np.inf and ns.lead_noise_dbfs == -np.inf
    assert ns.digital_silence_pct > 40


# ---------------------------------------------------------------- bandwidth


def _codec_lowpass(x: np.ndarray, cut_hz: float) -> np.ndarray:
    """Steep linear-phase FIR (~400 Hz transition), like a codec resampler."""
    return lfilter(firwin(511, cut_hz, fs=SR), 1.0, x)


def _tilted(x: np.ndarray) -> np.ndarray:
    """Give white noise a speech-like -6 dB/oct tilt above 500 Hz."""
    return sosfilt(butter(1, 500, fs=SR, output="sos"), x) * 4


def test_bandwidth_bluetooth_smoke_is_about_8k(smoke) -> None:  # type: ignore[no-untyped-def]
    a = load_wav(smoke / "degraded_bluetooth.wav")
    bw = effective_bandwidth(a.samples, a.sr)
    assert 7500 <= bw.bandwidth_hz <= 9000, bw


def test_bandwidth_fullband_noise_reads_nyquist() -> None:
    rng = np.random.default_rng(3)
    x = _speechlike(20, rng) + rng.standard_normal(20 * SR) * 1e-4
    bw = effective_bandwidth(x, SR)
    assert bw.bandwidth_hz >= 23000 and bw.limited_by == "nyquist"


def test_natural_tilt_is_not_mistaken_for_a_cliff() -> None:
    rng = np.random.default_rng(6)
    x = _tilted(_speechlike(20, rng)) + rng.standard_normal(20 * SR) * 1e-5
    bw = effective_bandwidth(x, SR)
    assert bw.cliff_hz is None, bw


@pytest.mark.parametrize("cut", [4000.0, 8000.0, 12000.0, 16000.0])
def test_bandwidth_tracks_codec_lowpass(cut: float) -> None:
    rng = np.random.default_rng(4)
    x = _codec_lowpass(_tilted(_speechlike(20, rng)), cut) + rng.standard_normal(20 * SR) * 1e-5
    bw = effective_bandwidth(x, SR)
    assert bw.limited_by == "cliff", bw
    assert cut <= bw.bandwidth_hz <= cut + 600, bw


@pytest.mark.parametrize("gain_db", [0.0, -30.0])
def test_bandwidth_is_level_independent(gain_db: float) -> None:
    rng = np.random.default_rng(7)
    x = _codec_lowpass(_tilted(_speechlike(20, rng)), 8000) + rng.standard_normal(20 * SR) * 1e-5
    assert 8000 <= effective_bandwidth(x * 10 ** (gain_db / 20), SR).bandwidth_hz <= 8600


def test_bandwidth_airpods_standin_zero_gated_12k_with_image() -> None:
    """Regression for a real AirPods recording (stand-in, the original is private):
    (1) room tone contains exact digital zeros from on-device gating, so any
        noise PSD has -inf bins; (2) audio was coded at 24 kHz and upsampled,
        so the spectrum ends hard at ~11.5 kHz but a mirror image of 10.5-12 kHz
        leaks ~20 dB down into 12-13.5 kHz. A naive "speech > noise + 6 dB"
        estimator reads 24 kHz here; expect ~12 kHz."""
    rng = np.random.default_rng(5)
    secs = 34
    speech = np.concatenate([np.zeros(4 * SR), _tilted(_speechlike(secs - 4, rng))])
    noise = rng.standard_normal(secs * SR) * 10 ** (-55 / 20)
    x24 = _codec_lowpass(speech + noise, 11500)
    # imaging: (-1)^n modulation maps f -> 24 kHz - f, so 10.5-11.9 kHz lands on 12.1-13.5 kHz
    band = sosfilt(butter(8, [10500, 11900], btype="band", fs=SR, output="sos"), x24)
    sign = np.where(np.arange(len(band)) % 2 == 0, 1.0, -1.0)
    x = x24 + band * sign * 10 ** (-20 / 20)
    # on-device gate: random 10-60 ms runs of exact zeros, mostly in the room tone
    for _ in range(120):
        s0 = int(rng.integers(0, 4 * SR)) if rng.random() < 0.7 else int(rng.integers(0, len(x)))
        x[s0 : s0 + int(rng.integers(480, 2880))] = 0.0
    assert np.sum(x[: 4 * SR] == 0) > 0.1 * 4 * SR  # really contains digital zeros
    bw = effective_bandwidth(x, SR)
    assert bw.limited_by == "cliff", bw
    assert 11000 <= bw.bandwidth_hz <= 12500, bw
    assert bw.snr_edge_hz > 12500  # the image alone would have fooled an SNR-only rule


def test_bandwidth_all_zero_does_not_crash() -> None:
    bw = effective_bandwidth(np.zeros(10 * SR), SR)
    assert np.isnan(bw.bandwidth_hz)
