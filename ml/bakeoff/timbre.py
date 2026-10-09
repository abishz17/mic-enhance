"""Timbre check the speaker embedding cannot do: per-octave-band level change of speech-active frames.

    uv run --project ml --group bakeoff python ml/bakeoff/timbre.py --models a,b [--clips STEM,...]

For each octave band (63 Hz .. 16 kHz) prints output-minus-input level (dB) over the frames where
the *input* is speech-active (top 30 % frame energy), and, on the input row, the band's speech-to-
noise margin (active level minus quiet-frame level). A band whose input margin is large (>15 dB)
is speech-dominated: any level change there is a change of the voice itself, not noise removal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT / "ml" / "src"))
from mlstage.audio import read_mono  # noqa: E402

CENTERS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def bands(x, act_mask=None):
    f, t, Z = ss.stft(x, 48_000, nperseg=4096, noverlap=3072)
    P = np.abs(Z) ** 2
    e = P.sum(axis=0)
    if act_mask is None:
        act_mask = e > np.percentile(e, 70)
        quiet = e < np.percentile(e, 10)
    else:
        quiet = None
    out_a, out_q = [], []
    for c in CENTERS:
        b = (f >= c / np.sqrt(2)) & (f < c * np.sqrt(2))
        out_a.append(10 * np.log10(P[b][:, act_mask[: P.shape[1]]].mean() + 1e-20))
        if quiet is not None:
            out_q.append(10 * np.log10(P[b][:, quiet].mean() + 1e-20))
    return np.array(out_a), (np.array(out_q) if quiet is not None else None), act_mask


ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--clips", default=",".join(p.stem for p in sorted((ROOT / "corpus").glob("*.wav"))),
                help="comma-separated corpus/ file stems (default: all)")
a = ap.parse_args()
hdr = " | ".join(f"{c if c < 1000 else str(c // 1000) + 'k'}" for c in CENTERS)
for clip in a.clips.split(","):
    x, _ = read_mono(ROOT / "corpus" / f"{clip}.wav")
    ia, iq, mask = bands(x)
    print(f"\n**{clip}** — octave-band level change on speech frames (dB, output − input)\n")
    print(f"| model | {hdr} |")
    print("|---|" + "---|" * len(CENTERS))
    print("| *input speech-to-noise margin* | " + " | ".join(f"*{m:.0f}*" for m in ia - iq) + " |")
    for m in a.models.split(","):
        p = OUT / m / "corpus" / f"{clip}.wav"
        if not p.exists():
            continue
        y, _ = read_mono(p)
        oa, _, _ = bands(y[: len(x)], mask)
        print(f"| {m} | " + " | ".join(f"{d:+.1f}" for d in oa - ia) + " |", flush=True)
