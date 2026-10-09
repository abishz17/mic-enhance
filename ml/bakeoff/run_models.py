"""Run every model over every test clip through the exact mlstage pipeline (cli.enhance_array),
loading each model once. Outputs: ml/out/<model>/<set>/<clip>.wav (48 kHz mono f32).

    uv run --project ml --group bakeoff python ml/bakeoff/run_models.py [--models a,b] [--sets x,y] [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from common import ALL_MODELS, SETS, clips, out_path  # noqa: E402
from mlstage import backends  # noqa: E402
from mlstage.audio import read_mono, write_48k_f32  # noqa: E402
from mlstage.cli import _pick_device, enhance_array  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=",".join(ALL_MODELS))
ap.add_argument("--sets", default=",".join(SETS))
ap.add_argument("--device", default="auto")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

for m in a.models.split(","):
    todo = [(s, c) for s in a.sets.split(",") for c in clips(s) if a.force or not out_path(m, s, c).exists()]
    if not todo:
        print(f"{m}: up to date", flush=True)
        continue
    # "a+b" = chain (a's output feeds b); "model@0.9" = that model at --strength 0.9;
    # "model~tm" = with the timbre-match post-EQ
    stages = []
    try:
        for part in m.split("+"):
            part, tm = (part[:-3], True) if part.endswith("~tm") else (part, False)
            base, _, strength = part.partition("@")
            cls = backends.get(base)
            dev = _pick_device(a.device, cls.info.devices)
            stages.append((cls(dev), float(strength) if strength else 1.0, tm))
    except Exception:  # noqa: BLE001
        print(f"{m}: LOAD FAILED\n{traceback.format_exc()}", flush=True)
        continue
    print(f"{m}: loaded {len(stages)} stage(s); {len(todo)} clips", flush=True)
    for s, c in todo:
        x, sr = read_mono(c)
        t = time.perf_counter()
        try:
            y, ysr = x, sr
            for be, strength, tm in stages:
                y, ysr = enhance_array(be, y, ysr, strength, tm), 48_000
        except Exception as e:  # noqa: BLE001
            print(f"  {m} {s}/{c.name}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        write_48k_f32(out_path(m, s, c), y)
        print(f"  {m} {s}/{c.name}: warm rtf {(time.perf_counter() - t) / (len(x) / sr):.3f}", flush=True)
    for be, *_ in stages:
        be.close()
