"""Cold-start performance through the real CLI (`uv run --project ml mlstage ...`), one process per run.

    uv run --project ml --group bakeoff python ml/bakeoff/perf.py MODEL[,MODEL] [--long PATH] [--device auto]

Reports wall seconds (incl. model load, as the Rust CLI will see it), RTF, peak physical footprint
(macOS lifetime max, includes Metal allocations) and checks the stdout JSON contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ml" / "out" / "_perf"

ap = argparse.ArgumentParser()
ap.add_argument("models")
ap.add_argument("--clips", required=True, help="comma-separated WAV paths")
ap.add_argument("--device", default="auto")
a = ap.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
print("| model | clip | dur s | device | wall s | RTF | load s | RTF excl. load | peak MB | stdout ok | len ok |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m in a.models.split(","):
    for clip in a.clips.split(","):
        clip = Path(clip)
        out = OUT / f"{m}_{clip.stem}.wav"
        st = OUT / f"{m}_{clip.stem}.json"
        p = subprocess.run(["uv", "run", "--quiet", "--project", str(ROOT / "ml"), "mlstage", "--model", m,
                            "--in", str(clip), "--out", str(out), "--stats", str(st), "--device", a.device],
                           capture_output=True, text=True)
        lines = p.stdout.strip().splitlines()
        ok = p.returncode == 0 and len(lines) == 1 and set(json.loads(lines[0])) == {"model", "seconds", "rtf", "device"}
        if p.returncode != 0:
            print(f"| {m} | {clip.name} | FAILED rc={p.returncode}: {p.stderr.strip().splitlines()[-1][:120]} |")
            continue
        s = json.loads(st.read_text())
        n_in, n_out = sf.info(str(clip)).frames, sf.info(str(out)).frames
        print(f"| {m} | {clip.name} | {s['audio_seconds']:.1f} | {s['device']} | {s['seconds']:.1f} | {s['rtf']:.3f} | "
              f"{s['load_seconds']:.1f} | {s['rtf_excl_load']:.3f} | {s['peak_footprint_mb']:.0f} | {ok} | "
              f"{n_out == round(n_in * 48000 / sf.info(str(clip)).samplerate)} |", flush=True)
        out.unlink(missing_ok=True)
