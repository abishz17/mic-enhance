"""Turn ml/out/_scores/**.json into the markdown tables pasted into ml/BAKEOFF.md.

    uv run --project ml --group bakeoff python ml/bakeoff/aggregate.py > ml/out/tables.md
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402

SC = OUT / "_scores"
rows = [json.loads(p.read_text()) for p in SC.glob("*/*/*.json")]
by = {}
for r in rows:
    by.setdefault((r["model"], r["set"]), []).append(r)

ORDER = ["input", "deepfilternet3", "deepfilternet3@0.9", "deepfilternet3@0.8", "dpdfnet8-48k", "dpdfnet8-48k@0.9",
         "dpdfnet2-48k", "mossformer2-48k", "resemble-denoise", "rnnoise", "gtcrn",
         "lavasr", "lavasr-dn", "sidon", "resemble-enhance", "voicefixer"]
models = [m for m in ORDER if any(k[0] == m for k in by)] + sorted({k[0] for k in by} - set(ORDER))


def f(v, fmt="{:.2f}"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return fmt.format(v)


def mean(rs, k):
    v = [r[k] for r in rs if k in r and r[k] is not None and not (isinstance(r[k], float) and math.isnan(r[k]))]
    return float(np.mean(v)) if v else None


def get_input(set_name, clip):
    for r in by.get(("input", set_name), []):
        if r["clip"] == clip:
            return r
    return {}


# --- corpus, per file ------------------------------------------------------------------------
print("### Corpus (real laptop-mic and AirPods recordings), per file\n")
print("Δ columns are output minus raw input. `spk` = ECAPA cosine(input, output). `room` = level of the "
      "first 4 s (room tone) in dBFS. `gaps` = level where the input is exact digital zero. "
      "WER = Whisper small.en vs the transcript of the raw input (ins/del/sub word counts).\n")
for clip in sorted({r["clip"] for r in by.get(("input", "corpus"), [])}):
    gaps = "zero_gap_db" in get_input("corpus", clip)  # input has exact-zero gating
    print(f"**{clip}**\n")
    print("| model | SIG | BAK | OVRL | spk | F0 Δ cents (p90 abs) | F0 range Δ st | tilt Δ dB/oct | presence Δ dB | room dBFS | "
          + ("gaps dBFS | " if gaps else "") + "bw kHz | >12k dB | WER (i/d/s) | lag ms |")
    print("|---" * (14 + gaps) + "|")
    for m in models:
        r = next((x for x in by.get((m, "corpus"), []) if x["clip"] == clip), None)
        if r is None:
            continue
        i = get_input("corpus", clip)
        ins = f"{r.get('ins', 0)}/{r.get('del', 0)}/{r.get('sub', 0)}" if "wer" in r else "–"
        rng = r.get("f0_range_delta_st") if m != "input" else None
        print(f"| {m} | {f(r['SIG'])} | {f(r['BAK'])} | {f(r['OVRL'])} | {f(r['spk_cos_in'], '{:.3f}')} | "
              f"{f(r['f0_shift_cents'], '{:+.1f}')} ({f(r.get('f0_p90_abs_cents'), '{:.0f}')}) | {f(rng, '{:+.2f}')} | "
              f"{f(r['tilt_db_oct'] - i['tilt_db_oct'] if m != 'input' else r['tilt_db_oct'], '{:+.2f}')} | "
              f"{f(r['presence_db'] - i['presence_db'] if m != 'input' else r['presence_db'], '{:+.1f}')} | "
              f"{f(r.get('lead_db'), '{:.0f}')} | "
              + (f"{f(r.get('zero_gap_db'), '{:.0f}')} | " if gaps else "")
              + f"{r['bw_hz'] / 1000:.1f} | {f(r['hf_12_20_db'], '{:.0f}')} | "
              f"{f(r.get('wer'), '{:.2f}')} ({ins}) | {f(r['lag_ms'], '{:+.2f}')} |")
    print()

# --- VoiceBank sets, mean over clips -------------------------------------------------------------
print("### Real speech with clean reference (VoiceBank-DEMAND family), mean over clips\n")
print("`spk→ref` = ECAPA cosine(clean reference, output); `Δspk` = that minus the same for the raw input "
      "(>0: output is closer to the true voice than the input was). WER vs the official transcript.\n")
for s, title in (("vbdemand", "noisy, SNR 2.5 dB (8 clips)"), ("vbreverb", "noisy + real-RIR reverb (8 clips)"),
                 ("vbbt_msbc", "Bluetooth HFP mSBC round trip, 8 kHz band (6 clips)"),
                 ("vbbt_eld", "AAC-ELD 24 kHz round trip, ~11 kHz band (6 clips)")):
    print(f"**{s}** — {title}\n")
    print("| model | SIG | BAK | OVRL | PESQ-WB | SI-SDR dB | spk→ref | Δspk | F0 Δ vs ref cents | tilt Δ vs ref | bw kHz | WER | ins | del |")
    print("|---" * 14 + "|")
    base = by.get(("input", s), [])
    b_spk = mean(base, "spk_cos_ref")
    for m in models:
        rs = by.get((m, s))
        if not rs:
            continue
        spk = mean(rs, "spk_cos_ref")
        print(f"| {m} | {f(mean(rs, 'SIG'))} | {f(mean(rs, 'BAK'))} | {f(mean(rs, 'OVRL'))} | {f(mean(rs, 'pesq_wb'))} | "
              f"{f(mean(rs, 'si_sdr'), '{:.1f}')} | {f(spk, '{:.3f}')} | "
              f"{f(spk - b_spk if spk is not None and b_spk is not None else None, '{:+.3f}')} | "
              f"{f(mean(rs, 'f0_shift_vs_ref_cents'), '{:+.1f}')} | {f(mean(rs, 'tilt_vs_ref'), '{:+.2f}')} | "
              f"{mean(rs, 'bw_hz') / 1000:.1f} | {f(mean(rs, 'wer'), '{:.3f}')} | "
              f"{sum(r.get('ins', 0) for r in rs)} | {sum(r.get('del', 0) for r in rs)} |")
    print()

# --- smoke ------------------------------------------------------------------------------------
print("### Smoke clips (synthetic, TTS)\n")
print("| model | clip | OVRL | PESQ-WB | lead-2s dBFS | bw kHz | WER | spk(in) |")
print("|---|---|---|---|---|---|---|---|")
for m in models:
    for r in sorted(by.get((m, "smoke"), []), key=lambda r: r["clip"]):
        print(f"| {m} | {r['clip']} | {f(r['OVRL'])} | {f(r.get('pesq_wb'))} | {f(r.get('lead_db'), '{:.0f}')} | "
              f"{r['bw_hz'] / 1000:.1f} | {f(r.get('wer'), '{:.2f}')} | {f(r['spk_cos_in'], '{:.3f}')} |")
print()

# --- hallucination log --------------------------------------------------------------------------
print("### Transcript differences (every clip where the output transcript differs from the reference "
      "and the input transcript did not have the same error)\n")
for m in models:
    if m == "input":
        continue
    for (mm, s), rs in sorted(by.items()):
        if mm != m:
            continue
        for r in sorted(rs, key=lambda r: r["clip"]):
            i = get_input(s, r["clip"])
            if "hyp_text" in r and r.get("wer", 0) > i.get("wer", 0) + 1e-9:
                print(f"- **{m}** {s}/{r['clip']}: WER {i.get('wer', 0):.2f} → {r['wer']:.2f}  \n"
                      f"  ref: “{r['ref_text']}”  \n  in:  “{i.get('hyp_text', '')}”  \n  out: “{r['hyp_text']}”")
