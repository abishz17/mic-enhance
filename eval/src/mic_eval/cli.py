"""mic-eval command line."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from .audio import AudioError, load_wav

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2

SCORE_EPILOG = """\
metrics and how far to trust them:
  LUFS / dBTP     BS.1770-4 integrated loudness, 4x-oversampled true peak. Exact.
  Floor           p10 of 50 ms frame RMS (dBFS). Lead = median 20 ms RMS of the
                  first --lead-silence seconds (e.g. 4 for takes that start with room tone).
  BW Hz / BW by   effective speech bandwidth; "cliff" = brickwall codec limit,
                  "snr" = speech meets noise, "nyquist" = no limit found.
  Clip            samples in flat-topped runs at the file's peak (digital clipping).
  SIG/BAK/OVRL    DNSMOS P.835 (16 kHz, blind above 8 kHz), P808 = DNSMOS P.808.
                  Noisy: treat differences < ~0.15 as noise.
"""


def _dump(obj: Any) -> str:
    def fix(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [fix(x) for x in v]
        return v

    return json.dumps(fix(obj), indent=2)


def cmd_score(a: argparse.Namespace) -> int:
    from .score import format_table, score_audio

    rows = []
    errors = 0
    for f in a.wavs:
        try:
            au = load_wav(f, downmix=a.downmix)
            rows.append(score_audio(au, lead_silence_s=a.lead_silence or None, with_dnsmos=not a.no_dnsmos))
        except AudioError as e:
            print(f"error: {e}", file=sys.stderr)
            errors += 1
    if a.json:
        print(_dump(rows))
    elif rows:
        print(format_table(rows))
    if errors:
        return EXIT_ERROR
    if a.strict and any(r["problems"] or not r["meets_target"] for r in rows):
        return EXIT_FAIL
    return EXIT_OK


def cmd_compare(a: argparse.Namespace) -> int:
    from dataclasses import asdict

    ref = load_wav(a.ref, downmix=a.downmix)
    cand = load_wav(a.cand, downmix=a.downmix)
    out: dict[str, Any] = {"ref": str(ref.path), "cand": str(cand.path)}
    verdicts: dict[str, str] = {}
    say = (lambda *_x: None) if a.json else print

    # 1) voice identity (always: keeping the speaker's voice is a hard requirement)
    if not a.no_voice:
        from .voice import THRESHOLDS, voice_check

        v = voice_check(ref.samples, cand.samples, ref.sr)
        out["voice"] = v.to_dict()
        verdicts["voice"] = v.verdict
        T = THRESHOLDS
        say("VOICE IDENTITY")
        sim = "-" if v.speaker_similarity is None else f"{v.speaker_similarity:.3f}"
        say(f"  speaker similarity (ECAPA cos)  {sim:>7s}   pass >= {T['speaker_similarity_min']}")
        say(f"  F0 median shift                 {v.f0_shift_st:+7.2f} st  pass |x| <= {T['f0_shift_max_st']}   (ref {v.f0_ref.median_hz:.1f} Hz, cand {v.f0_cand.median_hz:.1f} Hz)")
        say(f"  F0 range change (p10-p90)      {v.f0_range_change_st:+7.2f} st  pass |x| <= {T['f0_range_change_max_st']}   (ref {v.f0_ref.range_st:.1f} st, cand {v.f0_cand.range_st:.1f} st)")
        say(f"  spectral tilt change 100-5k Hz  {v.tilt_change_db_oct:+7.2f} dB/oct  pass |x| <= {T['tilt_change_max_db_oct']}   (ref {v.tilt_ref_db_oct:.2f}, cand {v.tilt_cand_db_oct:.2f})")
        say(f"  => {v.verdict}" + (f"  ({'; '.join(v.reasons)})" if v.reasons else ""))

    # 2) invented-word guard: ASR diff + ASR-free presence check
    if not a.no_asr:
        from .asr_guard import format_ops, guard_verdict, speech_presence_check, transcribe

        tr = transcribe(ref.samples, ref.sr, a.model)
        tc = transcribe(cand.samples, cand.sr, a.model)
        g = guard_verdict(tr, tc, max_wer=a.max_wer, conf=a.conf)
        pr = speech_presence_check(ref.samples, cand.samples, ref.sr)
        out["guard"] = g.to_dict()
        out["presence"] = asdict(pr)
        verdicts["words"] = g.verdict
        verdicts["presence"] = pr.verdict
        say(f"\nWORDS (Whisper {a.model}): ref {g.n_ref_words} words, cand {len(tc.words)} words")
        say(f"  WER {g.wer:.2%}  (S={g.substitutions} D={g.deletions} I={g.insertions})")
        if g.ops:
            say("  word diff ([-removed-] {+added+}; !! = confident edit, others = ASR uncertainty, review by ear):")
            say(format_ops(g.ops))
        else:
            say("  transcripts identical after normalization")
        say(f"  rule: FAIL if a raw word heard with p >= {a.conf:.2f} is deleted/changed, an added word has p >= {a.conf:.2f}, or WER > {a.max_wer:.0%}")
        say(f"  => {g.verdict}" + (f"  ({'; '.join(g.reasons)})" if g.reasons else ""))
        say("\nNEW SOUND IN PAUSES (ASR-free)")
        for e in pr.events:
            say(f"  !! {e.start:7.2f}-{e.end:.2f}s  +{e.rise_db:.1f} dB over raw")
        say(f"  duration delta {pr.duration_delta_s:+.3f} s, level offset {pr.gain_offset_db:+.1f} dB")
        say(f"  => {pr.verdict}" + (f"  ({'; '.join(pr.reasons)})" if pr.reasons else ""))

    if a.clean:
        from .intrusive import intrusive

        clean = load_wav(a.clean, downmix=a.downmix)
        res = {}
        for name, au in (("ref", ref), ("cand", cand)):
            res[name] = asdict(intrusive(clean.samples, au.samples, au.sr))
        out["intrusive_vs_clean"] = res
        say(f"\nINTRUSIVE vs clean reference {clean.path.name}:")
        say(f"  {'':5s} {'PESQ-WB':>8s} {'SI-SDR':>8s} {'STOI':>6s} {'ESTOI':>6s} {'lag':>6s}")
        for name in ("ref", "cand"):
            r = res[name]

            def f(v: Any, fmt: str) -> str:
                return "-" if v is None else format(v, fmt)

            say(f"  {name:5s} {f(r['pesq_wb'], '8.3f')} {f(r['si_sdr_db'], '8.2f')} {f(r['stoi'], '6.3f')} {f(r['estoi'], '6.3f')} {r['lag_samples']:6d}")

    overall = "FAIL" if "FAIL" in verdicts.values() else "PASS"
    out["verdicts"] = verdicts
    out["verdict"] = overall
    if a.json:
        print(_dump(out))
    else:
        print(f"\nOVERALL: {overall}  " + "  ".join(f"{k}={v}" for k, v in verdicts.items()))
    return EXIT_OK if overall == "PASS" else EXIT_FAIL


def cmd_report(a: argparse.Namespace) -> int:
    from .report import build_report

    out = Path(a.output)
    rows = build_report(
        a.inputs,
        out,
        clean_ref=a.clean,
        lead_silence_s=a.lead_silence or None,
        with_dnsmos=not a.no_dnsmos,
        downmix=a.downmix,
        ref_variant=a.ref_variant,
        with_voice=not a.no_voice,
        with_guard=a.guard,
    )
    print(f"wrote {out.resolve()} ({len(rows)} files, {out.stat().st_size / 1e6:.1f} MB)")
    return EXIT_OK


def cmd_abx(a: argparse.Namespace) -> int:
    from .abx import run_session

    run_session(
        a.a,
        a.b,
        trials=a.trials,
        mode=a.mode,
        start=a.start,
        duration=a.duration,
        seed=a.seed,
        log=Path(a.log) if a.log else None,
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mic-eval", description="Measurement harness for mic-enhance (offline, nothing is uploaded).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--downmix", action="store_true", help="average multichannel input to mono instead of erroring")

    s = sub.add_parser("score", help="per-file metrics", epilog=SCORE_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("wavs", nargs="+")
    s.add_argument("--json", action="store_true")
    s.add_argument("--lead-silence", type=float, default=0.0, metavar="SEC", help="measure room tone over the first SEC seconds")
    s.add_argument("--no-dnsmos", action="store_true")
    s.add_argument("--strict", action="store_true", help="exit 1 if any file has problems or misses -14 LUFS / -1 dBTP")
    common(s)
    s.set_defaults(fn=cmd_score)

    c = sub.add_parser("compare", help="voice identity + invented-word guard (+ intrusive metrics with --clean); exit 1 on FAIL")
    c.add_argument("--ref", required=True, help="raw / unprocessed recording")
    c.add_argument("--cand", required=True, help="processed output")
    c.add_argument("--clean", help="clean reference for PESQ / SI-SDR / STOI (synthetic tests only)")
    c.add_argument("--model", default="small.en", help="faster-whisper model (default small.en)")
    c.add_argument("--max-wer", type=float, default=0.25, help="gross-failure WER limit (default 0.25)")
    c.add_argument("--conf", type=float, default=0.90, help="word probability above which an edit is 'confident' (default 0.90)")
    c.add_argument("--no-asr", action="store_true", help="skip Whisper and the presence check")
    c.add_argument("--no-voice", action="store_true", help="skip the voice-identity check")
    c.add_argument("--json", action="store_true")
    common(c)
    c.set_defaults(fn=cmd_compare)

    r = sub.add_parser("report", help="local HTML report with table, spectrograms, players")
    r.add_argument("inputs", nargs="+", help="wav files (one group) or directories (grouped by file stem)")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--clean", help="clean reference for intrusive metrics")
    r.add_argument("--lead-silence", type=float, default=0.0, metavar="SEC")
    r.add_argument("--no-dnsmos", action="store_true")
    r.add_argument("--ref-variant", help="label (parent dir name) of the raw input in each group (default: raw/corpus/input/first)")
    r.add_argument("--no-voice", action="store_true", help="skip voice-identity columns")
    r.add_argument("--guard", action="store_true", help="also run the Whisper word guard + presence check per variant")
    common(r)
    r.set_defaults(fn=cmd_report)

    x = sub.add_parser("abx", help="blind loudness-matched ABX / AB preference test (afplay)")
    x.add_argument("a")
    x.add_argument("b")
    x.add_argument("--trials", type=int, default=10)
    x.add_argument("--mode", choices=["abx", "ab"], default="abx")
    x.add_argument("--start", type=float, default=0.0, help="excerpt start (s)")
    x.add_argument("--duration", type=float, default=None, help="excerpt length (s)")
    x.add_argument("--seed", type=int, default=None)
    x.add_argument("--log", help="append results as JSON lines to this file")
    x.set_defaults(fn=cmd_abx)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
