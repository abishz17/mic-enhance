"""DNSMOS port parity: numbers recorded from Microsoft's own dnsmos_local.py
(DNS-Challenge master, non-personalized) on the smoke files. Our port uses
soxr instead of librosa's resampler, hence the small tolerance."""

from __future__ import annotations

import numpy as np
import pytest

from mic_eval.audio import load_wav
from mic_eval.dnsmos import dnsmos

REFERENCE = {  # SIG, BAK, OVRL, P808 from dnsmos_local.py
    "clean.wav": (3.611, 4.163, 3.387, 4.230),
    "degraded_bluetooth.wav": (3.557, 3.973, 3.247, 4.152),
    "degraded_noise_hum_room.wav": (3.533, 3.968, 3.184, 4.079),
}


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_matches_reference_implementation(smoke, name: str) -> None:  # type: ignore[no-untyped-def]
    a = load_wav(smoke / name)
    d = dnsmos(a.samples, a.sr)
    sig, bak, ovrl, p808 = REFERENCE[name]
    assert d.sig == pytest.approx(sig, abs=0.01)
    assert d.bak == pytest.approx(bak, abs=0.01)
    assert d.ovrl == pytest.approx(ovrl, abs=0.01)
    assert d.p808 == pytest.approx(p808, abs=0.03)


def test_short_clip_is_tiled_not_crashing() -> None:
    rng = np.random.default_rng(0)
    d = dnsmos(rng.standard_normal(48_000) * 0.05, 48_000)
    # 1 s is tiled to 16 s like the reference -> 7 windows of 9.01 s
    assert d.n_windows == 7 and 1.0 <= d.ovrl <= 5.0
