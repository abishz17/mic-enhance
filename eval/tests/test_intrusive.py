from __future__ import annotations

import numpy as np
import pytest

from conftest import SR
from mic_eval.audio import load_wav
from mic_eval.intrusive import align, intrusive, si_sdr


def test_si_sdr_known_snr() -> None:
    rng = np.random.default_rng(0)
    s = rng.standard_normal(SR * 5)
    n = rng.standard_normal(SR * 5)
    n = n - s * np.dot(n, s) / np.dot(s, s)  # make noise orthogonal so SI-SDR == SNR exactly
    n *= np.linalg.norm(s) / np.linalg.norm(n) * 10 ** (-10 / 20)
    assert si_sdr(s, s + n) == pytest.approx(10.0, abs=0.01)
    assert si_sdr(s, 0.1 * (s + n)) == pytest.approx(10.0, abs=0.01)  # scale invariant


def test_align_recovers_delay() -> None:
    rng = np.random.default_rng(1)
    s = rng.standard_normal(SR * 2)
    d = np.concatenate([np.zeros(123), s])
    r, dd, lag = align(s, d, SR)
    assert lag == 123
    assert np.allclose(r[:1000], dd[:1000])


def test_identical_speech_scores_max(smoke) -> None:  # type: ignore[no-untyped-def]
    c = load_wav(smoke / "clean.wav").samples
    delayed = np.concatenate([np.zeros(480), c]) * 0.5
    res = intrusive(c, delayed, SR)
    assert res.lag_samples == 480
    assert res.pesq_wb == pytest.approx(4.64, abs=0.01)
    assert res.stoi == pytest.approx(1.0, abs=1e-3)
    assert res.si_sdr_db > 60


def test_degradation_ranks_correctly(smoke) -> None:  # type: ignore[no-untyped-def]
    c = load_wav(smoke / "clean.wav").samples
    bt = intrusive(c, load_wav(smoke / "degraded_bluetooth.wav").samples, SR)
    room = intrusive(c, load_wav(smoke / "degraded_noise_hum_room.wav").samples, SR)
    assert bt.pesq_wb is not None and room.pesq_wb is not None
    assert 4.64 > bt.pesq_wb > room.pesq_wb
    assert bt.si_sdr_db > room.si_sdr_db
