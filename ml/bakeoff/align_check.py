"""Timing check: run models through the real mlstage pipeline and measure output lag vs input.

    uv run --project ml --group bakeoff python ml/bakeoff/align_check.py MODEL [MODEL...] [--device cpu|mps]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from metrics import lag_ms  # noqa: E402
from mlstage import backends  # noqa: E402
from mlstage.audio import read_mono  # noqa: E402
from mlstage.cli import enhance_array  # noqa: E402

CLIPS = [ROOT / "testdata/smoke/clean.wav", *sorted((ROOT / "corpus").glob("*.wav"))]

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
for m in a.models:
    be = backends.get(m)(a.device)
    for c in CLIPS:
        x, sr = read_mono(c)
        y = enhance_array(be, x, sr)
        print(f"{m:18s} {c.name:24s} len_ok={len(y) == len(x)} lag={lag_ms(x, y):+.3f} ms", flush=True)
    be.close()
