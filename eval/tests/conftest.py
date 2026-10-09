from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SR = 48_000
REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "testdata" / "smoke"


def sine(freq: float, dbfs_peak: float, seconds: float, sr: int = SR, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return 10 ** (dbfs_peak / 20) * np.sin(2 * np.pi * freq * t + phase)


@pytest.fixture(scope="session")
def smoke() -> Path:
    if not (SMOKE / "clean.wav").exists():
        pytest.skip("testdata/smoke/clean.wav not present (test recordings are not in the repo)")
    return SMOKE
