"""Score every ml/out/<model>/<set>/<clip>.wav (+ the raw input as model "input").

    uv run --project ml --group bakeoff python ml/bakeoff/score.py [--models a,b] [--sets x,y] [--no-asr]

Per-file results are cached as JSON in ml/out/_scores/<model>/<set>/<clip>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from common import ALL_MODELS, OUT, SETS, clips, out_path, ref_for, ref_text  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlstage.audio import read_mono, resample  # noqa: E402

SCORES = OUT / "_scores"


@lru_cache(maxsize=64)
def load48(p: str) -> np.ndarray:
    x, sr = read_mono(p)
    return resample(x, sr, 48_000)


@lru_cache(maxsize=64)
def emb(p: str) -> np.ndarray:
    return M.spk_emb(load48(p))


@lru_cache(maxsize=64)
def f0(p: str) -> np.ndarray:
    return M.f0_stats(load48(p))


@lru_cache(maxsize=64)
def text_of(p: str) -> str:
    return M.transcribe(load48(p))


def zero_runs_mask(x: np.ndarray, min_len: int = 480) -> np.ndarray:
    """Samples inside runs of exact digital zeros >= min_len (on-device gating, e.g. AirPods)."""
    z = x == 0
    m = np.zeros_like(z)
    if not z.any():
        return m
    d = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
    for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        if e - s >= min_len:
            m[s:e] = True
    return m


def band_db(x48, lo, hi, active_only=True) -> float:
    import scipy.signal as ss

    f, t, Z = ss.stft(x48, 48_000, nperseg=2048, noverlap=1536)
    P = np.abs(Z) ** 2
    if active_only:
        e = P.sum(axis=0)
        P = P[:, e > np.percentile(e, 70)]
    return float(10 * np.log10(P[(f >= lo) & (f < hi)].mean() + 1e-20))


def score_one(model: str, set_name: str, clip: Path, asr: bool, old: dict | None = None) -> dict:
    inp = str(clip)
    outp = inp if model == "input" else str(out_path(model, set_name, clip))
    x = load48(inp)
    y = load48(outp)
    ref = ref_for(set_name, clip)
    lead_s = SETS[set_name][2]
    r: dict = {"model": model, "set": set_name, "clip": clip.stem}
    r.update({k: float(v) for k, v in M.dnsmos(y).items()})
    r["bw_hz"] = M.effective_bandwidth(y, 48_000)
    r["peak"] = float(np.abs(y).max())
    r["clip_frac"] = M.clip_frac(y)
    r["lag_ms"] = M.lag_ms(x, y)
    if lead_s:
        r["lead_db"] = M.lead_energy_db(y, lead_s)
    zm = zero_runs_mask(x)
    if zm.mean() > 0.001:
        r["zero_gap_db"] = float(10 * np.log10(np.mean(y[zm].astype(np.float64) ** 2) + 1e-20))
    # identity / timbre vs the input (what a listener hears changing)
    r["spk_cos_in"] = float(np.dot(emb(inp), emb(outp)))
    fi, fo = f0(inp), f0(outp)
    r["f0_med_hz"], _ = M.f0_summary(fo)
    r.update(M.f0_compare(fi, fo))
    r["tilt_db_oct"] = M.spectral_tilt(y)
    r["presence_db"] = M.band_ratio_db(y)  # 2-5 kHz vs 0.5-2 kHz
    r["hf_9_11_vs_5_8_db"] = band_db(y, 9000, 11000) - band_db(y, 5000, 8000)
    r["hf_12_20_db"] = band_db(y, 12000, 20000)
    if ref is not None:
        rr = load48(str(ref))
        n = min(len(rr), len(y))
        r["pesq_wb"] = M.pesq_wb(rr[:n], y[:n])
        # reverberant inputs lag the dry reference by the RIR's direct-path delay; align first
        lag = int(round(M.lag_ms(rr[:n], y[:n], max_ms=100) * 48))
        ya = y[lag:] if lag >= 0 else np.pad(y, (-lag, 0))
        m_ = min(n, len(ya))
        r["ref_lag_ms"] = lag / 48
        r["si_sdr"] = M.si_sdr(rr[:m_], ya[:m_])
        r["spk_cos_ref"] = float(np.dot(emb(str(ref)), emb(outp)))
        r["lsd_0_8k"] = M.lsd_band(rr[:m_], ya[:m_], 50, 8000)
        r["lsd_8_12k"] = M.lsd_band(rr[:m_], ya[:m_], 8000, 12000)
        r["lsd_12_20k"] = M.lsd_band(rr[:m_], ya[:m_], 12000, 20000)
        r["tilt_vs_ref"] = r["tilt_db_oct"] - M.spectral_tilt(rr)
        fr = f0(str(ref))
        r["f0_shift_vs_ref_cents"] = M.f0_compare(fr, fo)["f0_shift_cents"]
    if asr and old and "hyp_text" in old:  # audio unchanged since last scoring: reuse transcripts
        for k in ("ref_text", "hyp_text", "wer", "ins", "del", "sub"):
            r[k] = old[k]
    elif asr:
        rt = ref_text(set_name, clip)
        if rt is None:  # smoke: transcript of the clean TTS; corpus: transcript of the raw input
            rt = text_of(str(ref) if ref is not None else inp)
        hyp = text_of(outp)
        r["ref_text"], r["hyp_text"] = rt, hyp
        r.update(M.wer(rt, hyp))
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="input," + ",".join(ALL_MODELS))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for m in a.models.split(","):
        for s in a.sets.split(","):
            for c in clips(s):
                op = c if m == "input" else out_path(m, s, c)
                if not op.exists():
                    continue
                dst = SCORES / m / s / f"{c.stem}.json"
                fresh = dst.exists() and dst.stat().st_mtime > op.stat().st_mtime
                old = json.loads(dst.read_text()) if fresh else None
                if fresh and not a.force and (a.no_asr or "wer" in old):
                    continue
                t = time.perf_counter()
                r = score_one(m, s, c, asr=not a.no_asr, old=old)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(r, indent=1))
                print(f"{m:18s} {s:10s} {c.stem:28s} OVRL {r['OVRL']:.2f} spk {r['spk_cos_in']:.3f} "
                      f"wer {r.get('wer', float('nan')):.3f} ({time.perf_counter() - t:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
