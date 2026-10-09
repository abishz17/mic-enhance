"""Can a generative output be dry/wet-mixed with the clean-mode output without comb filtering?

    uv run --project ml --group bakeoff python ml/bakeoff/mix_check.py --dry deepfilternet3 --wet sidon,lavasr

For each band: magnitude-squared coherence between wet and dry, and the 50/50-mix level relative to
the level a phase-coherent mix would have. ~0 dB = coherent (mixing is safe); toward -3 dB and with
large coherence dips = phase-incoherent (mixing makes comb filtering / phasiness).
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

BANDS = [(100, 500), (500, 2000), (2000, 5000), (5000, 8000), (8000, 12000), (12000, 20000)]

ap = argparse.ArgumentParser()
ap.add_argument("--dry", default="deepfilternet3")
ap.add_argument("--wet", required=True)
ap.add_argument("--clips", default=",".join(p.stem for p in sorted((ROOT / "corpus").glob("*.wav"))),
                help="comma-separated corpus/ file stems (default: all)")
a = ap.parse_args()
print("| wet model | clip | " + " | ".join(f"{lo // 1000 if lo >= 1000 else lo / 1000:g}-{hi // 1000}k coh / mix dB" for lo, hi in BANDS) + " |")
print("|---|---|" + "---|" * len(BANDS))
for w in a.wet.split(","):
    for c in a.clips.split(","):
        pd, pw = OUT / a.dry / "corpus" / f"{c}.wav", OUT / w / "corpus" / f"{c}.wav"
        if not (pd.exists() and pw.exists()):
            continue
        d, _ = read_mono(pd)
        x, _ = read_mono(pw)
        n = min(len(d), len(x))
        d, x = d[:n], x[:n]
        # level-match wet to dry so the test is about phase, not gain
        x = x * np.sqrt(np.sum(d ** 2) / (np.sum(x ** 2) + 1e-20))
        f, coh = ss.coherence(d, x, fs=48_000, nperseg=2048)
        _, _, D = ss.stft(d, 48_000, nperseg=2048)
        _, _, X = ss.stft(x, 48_000, nperseg=2048)
        cells = []
        for lo, hi in BANDS:
            b = (f >= lo) & (f < hi)
            mix = np.sum(np.abs(0.5 * D[b] + 0.5 * X[b]) ** 2)
            coherent = np.sum((0.5 * np.abs(D[b]) + 0.5 * np.abs(X[b])) ** 2)
            cells.append(f"{coh[b].mean():.2f} / {10 * np.log10(mix / (coherent + 1e-20) + 1e-20):+.1f}")
        print(f"| {w} | {c} | " + " | ".join(cells) + " |", flush=True)
