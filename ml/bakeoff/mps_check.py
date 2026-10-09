"""MPS correctness + speed: same model, same input, CPU vs MPS.

    uv run --project ml --group bakeoff python ml/bakeoff/mps_check.py MODEL [MODEL...]
Reports SNR of the MPS output against the CPU output (CPU taken as truth) and warm wall time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))
from mlstage import backends  # noqa: E402
from mlstage.audio import read_mono  # noqa: E402
from mlstage.cli import enhance_array  # noqa: E402

CLIPS = [ROOT / "testdata/smoke/degraded_noise_hum_room.wav", *sorted((ROOT / "corpus").glob("*.wav"))[:1]]

for m in sys.argv[1:]:
    cls = backends.get(m)
    outs = {}
    for dev in ("cpu", "mps"):
        try:
            be = cls(dev)
        except Exception as e:  # noqa: BLE001
            print(f"{m} {dev}: load failed: {e}")
            continue
        for c in CLIPS:
            x, sr = read_mono(c)
            enhance_array(be, x[: sr * 3], sr)  # warm-up (MPS kernel compilation)
            t = time.perf_counter()
            try:
                y = enhance_array(be, x, sr)
            except Exception as e:  # noqa: BLE001
                print(f"{m} {dev} {c.name}: FAILED {type(e).__name__}: {e}")
                continue
            dt = time.perf_counter() - t
            outs[(dev, c.name)] = y
            print(f"{m:16s} {dev:4s} {c.name:30s} warm rtf {dt / (len(x) / sr):.3f}", flush=True)
        be.close()
    for c in CLIPS:
        a, b = outs.get(("cpu", c.name)), outs.get(("mps", c.name))
        if a is None or b is None:
            continue
        snr = 10 * np.log10(np.sum(a.astype(np.float64) ** 2) / (np.sum((a - b).astype(np.float64) ** 2) + 1e-20))
        print(f"{m:16s} {c.name:30s} MPS-vs-CPU SNR {snr:6.1f} dB  max|diff| {np.abs(a - b).max():.2e}", flush=True)
