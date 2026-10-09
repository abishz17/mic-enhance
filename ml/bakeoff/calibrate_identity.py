"""Anchor the speaker-similarity scale: how far do *known* manipulations move the ECAPA cosine?

    uv run --project ml --group bakeoff python ml/bakeoff/calibrate_identity.py

Prints a table used in BAKEOFF.md to justify the identity pass threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from common import ROOT, TD  # noqa: E402

sys.path.insert(0, str(ROOT / "ml" / "src"))
from mlstage.audio import read_mono, resample  # noqa: E402


def load(p):
    x, sr = read_mono(p)
    return resample(x, sr, 48_000)


def tilt(x, db_per_oct):
    """Linear-phase EQ: gain = db_per_oct * log2(f / 1 kHz), applied in the FFT domain."""
    X = np.fft.rfft(x)
    f = np.maximum(np.fft.rfftfreq(len(x), 1 / 48_000), 20)
    return np.fft.irfft(X * 10 ** (db_per_oct * np.log2(f / 1000) / 20), n=len(x)).astype(np.float32)


def peq(x, f0, gain_db, q=1.0):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / 48_000
    al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return ss.filtfilt(b, a, x).astype(np.float32)


def pitch(x, cents):
    import librosa

    return librosa.effects.pitch_shift(x, sr=48_000, n_steps=cents / 100).astype(np.float32)


def lowpass(x, fc):
    sos = ss.butter(12, fc, fs=48_000, output="sos")
    return ss.sosfiltfilt(sos, x).astype(np.float32)


def noise(x, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(len(x)).astype(np.float32)
    act = x[np.abs(x) > 0.01 * np.abs(x).max()]
    p = np.mean(act ** 2)
    return (x + n * np.sqrt(p / 10 ** (snr_db / 10))).astype(np.float32)


TRANSFORMS = [
    ("identity", lambda x: x),
    ("gain -6 dB", lambda x: x * 0.5),
    ("lowpass 12 kHz", lambda x: lowpass(x, 12000)),
    ("lowpass 8 kHz", lambda x: lowpass(x, 8000)),
    ("tilt +1 dB/oct", lambda x: tilt(x, 1)),
    ("tilt -1 dB/oct", lambda x: tilt(x, -1)),
    ("tilt +3 dB/oct", lambda x: tilt(x, 3)),
    ("tilt -3 dB/oct", lambda x: tilt(x, -3)),
    ("presence +6 dB @3k", lambda x: peq(x, 3000, 6)),
    ("pitch +25 cents", lambda x: pitch(x, 25)),
    ("pitch -25 cents", lambda x: pitch(x, -25)),
    ("pitch +50 cents", lambda x: pitch(x, 50)),
    ("pitch +100 cents", lambda x: pitch(x, 100)),
    ("pitch -100 cents", lambda x: pitch(x, -100)),
    ("white noise SNR 30 dB", lambda x: noise(x, 30)),
    ("white noise SNR 20 dB", lambda x: noise(x, 20)),
    ("white noise SNR 10 dB", lambda x: noise(x, 10)),
]


def main():
    corpus = sorted((ROOT / "corpus").glob("*.wav"))
    srcs = {**{f"corpus {p.stem}": p for p in corpus[:1]}, "vb p232_011": TD / "vbref/p232_011.wav",
            "vb p257_011": TD / "vbref/p257_011.wav"}
    print("| manipulation | " + " | ".join(srcs) + " |")
    print("|---|" + "---|" * len(srcs))
    base = {k: load(v) for k, v in srcs.items()}
    embs = {k: M.spk_emb(v) for k, v in base.items()}
    for name, fn in TRANSFORMS:
        row = [f"{float(np.dot(embs[k], M.spk_emb(fn(base[k])))):.3f}" for k in srcs]
        print(f"| {name} | " + " | ".join(row) + " |", flush=True)
    # same speaker, different sentence / different speaker
    e = {p.stem: M.spk_emb(load(p)) for p in sorted((TD / "vbref").glob("*.wav"))}
    same = [float(np.dot(e[a], e[b])) for a in e for b in e if a < b and a[:4] == b[:4]]
    diff = [float(np.dot(e[a], e[b])) for a in e for b in e if a < b and a[:4] != b[:4]]
    print(f"\nVB same speaker, different sentence: mean {np.mean(same):.3f} (min {np.min(same):.3f}, n={len(same)})")
    print(f"VB different speaker (p232 vs p257): mean {np.mean(diff):.3f} (max {np.max(diff):.3f}, n={len(diff)})")
    c = {p.stem: M.spk_emb(load(p)) for p in corpus}
    print("corpus cross-file (same person, different mic/session): " + ", ".join(
        f"{a} vs {b}: {float(np.dot(c[a], c[b])):.3f}" for a in c for b in c if a < b))
    print("corpus speaker vs VB speakers: max " + f"{max(float(np.dot(c[k], v)) for k in c for v in e.values()):.3f}")


if __name__ == "__main__":
    main()
