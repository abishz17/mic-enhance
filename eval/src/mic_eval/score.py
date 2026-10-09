"""Aggregate per-file scoring used by `score` and `report`."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from .audio import Audio
from .signal_metrics import clipping, effective_bandwidth, integrated_lufs, noise_stats, sample_peak_dbfs, true_peak_dbtp

TARGET_LUFS = -14.0
TARGET_TP = -1.0

# How far to trust each number. "repeatability" = what changes if you
# re-measure the same file / a slightly different take.
RELIABILITY: dict[str, str] = {
    "lufs_i": "reliable: BS.1770-4, matches ffmpeg ebur128 within 0.06 LU on real and smoke recordings; deterministic",
    "true_peak_dbtp": "reliable: 4x oversampling, may under-read by <=0.7 dB only for content near Nyquist",
    "noise_floor_dbfs": "reliable for files with pauses; p10 of 50 ms RMS, biased high if speech is continuous. "
    "On denoised/gated outputs it reads the gate (-100..-155 dBFS) and SNR becomes meaningless",
    "lead_noise_dbfs": "reliable if the window really is room tone (median 20 ms RMS)",
    "bandwidth_hz": "reliable to ~+/-300 Hz for band-limited (codec) input; 'nyquist' = no limit found",
    "clipped_samples": "reliable for hard digital clipping; blind to soft/analog clipping",
    "voice identity": "speaker sim / F0 / tilt vs the raw take; thresholds from eval/tools/calibrate_voice.py (21/21 benign pass, 15/15 shifts fail on real recordings); a noisy raw take depresses similarity (25 dB SNR noise alone: -0.02..-0.045)",
    "words guard": "Whisper small.en diff; calibrated in eval/tools/calibrate_guard.py; misses edits to words Whisper itself was unsure of (p < 0.9, ~25-35 % of casual speech)",
    "dnsmos": "noisy: +/-0.1-0.2 between takes; 16 kHz model, blind above 8 kHz; trained on real noisy speech, odd on TTS",
}


def _clean(v: Any) -> Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None if math.isnan(v) else ("-inf" if v < 0 else "inf")
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def score_audio(
    a: Audio,
    *,
    lead_silence_s: float | None = None,
    with_dnsmos: bool = True,
) -> dict[str, Any]:
    x, sr = a.samples, a.sr
    r: dict[str, Any] = {
        "file": str(a.path),
        "duration_s": round(a.duration, 3),
        "source_sr": a.source_sr,
        "source_channels": a.source_channels,
        "subtype": a.subtype,
        "warnings": list(a.warnings),
    }
    lufs = integrated_lufs(x, sr)
    tp = true_peak_dbtp(x)
    r["lufs_i"] = lufs
    r["true_peak_dbtp"] = tp
    r["sample_peak_dbfs"] = sample_peak_dbfs(x)
    r["meets_target"] = bool(lufs is not None and math.isfinite(lufs) and abs(lufs - TARGET_LUFS) <= 1.0 and tp <= TARGET_TP)
    n = noise_stats(x, sr, lead_silence_s)
    r.update(asdict(n))
    bw = effective_bandwidth(x, sr)
    r["bandwidth_hz"] = bw.bandwidth_hz
    r["bandwidth_limited_by"] = bw.limited_by
    r["bandwidth_snr_edge_hz"] = bw.snr_edge_hz
    r["bandwidth_cliff_hz"] = bw.cliff_hz
    r["gated_frames_pct"] = round(bw.gated_frames_pct, 2)
    c = clipping(x)
    r.update(asdict(c))
    r["nonfinite_samples"] = a.nonfinite_samples

    problems: list[str] = []
    if a.nonfinite_samples:
        problems.append(f"{a.nonfinite_samples} NaN/Inf samples")
    if c.over_fs_samples:
        problems.append(f"{c.over_fs_samples} samples above full scale")
    if c.clipped_samples:
        problems.append(f"clipping: {c.clip_runs} flat-top runs")
    if lufs is None:
        problems.append("too short for loudness (<400 ms)")
    r["problems"] = problems

    if with_dnsmos:
        from .dnsmos import dnsmos

        d = dnsmos(x, sr)
        r.update({f"dnsmos_{k}": v for k, v in asdict(d).items()})
    return _clean(r)


def _f(v: Any, fmt: str) -> str:
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return format(v, fmt)


def format_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        ("file", "File", "s"),
        ("lufs_i", "LUFS", ".2f"),
        ("true_peak_dbtp", "dBTP", ".2f"),
        ("noise_floor_dbfs", "Floor", ".1f"),
        ("lead_noise_dbfs", "Lead", ".1f"),
        ("snr_db", "SNR", ".1f"),
        ("bandwidth_hz", "BW Hz", ".0f"),
        ("bandwidth_limited_by", "BW by", "s"),
        ("clipped_samples", "Clip", "d"),
        ("dnsmos_sig", "SIG", ".2f"),
        ("dnsmos_bak", "BAK", ".2f"),
        ("dnsmos_ovrl", "OVRL", ".2f"),
        ("dnsmos_p808", "P808", ".2f"),
    ]
    cols = [c for c in cols if any(c[0] in r for r in rows)]
    body = []
    for r in rows:
        cells = []
        for k, _h, fmt in cols:
            v = r.get(k)
            if k == "file":
                v = str(v).split("/")[-1]
            cells.append(_f(v, fmt))
        body.append(cells)
    heads = [h for _k, h, _f2 in cols]
    w = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(heads)]
    out = ["  ".join(h.rjust(w[i]) if i else h.ljust(w[i]) for i, h in enumerate(heads))]
    out.append("  ".join("-" * x for x in w))
    for b in body:
        out.append("  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(b)))
    for r in rows:
        name = str(r["file"]).split("/")[-1]
        for p in r.get("problems", []):
            out.append(f"! {name}: {p}")
        for wmsg in r.get("warnings", []):
            out.append(f"~ {name}: {wmsg}")
    return "\n".join(out)
