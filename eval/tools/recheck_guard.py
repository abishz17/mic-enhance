"""Re-check the word guard on real processed outputs vs the raw corpus take.

    uv run --project eval python eval/tools/recheck_guard.py RAW.wav CAND.wav [CAND.wav ...]
Prints verdict + every edit (!! = confident) for each candidate.
"""

from __future__ import annotations

import sys

from mic_eval.asr_guard import format_ops, guard_verdict, transcribe
from mic_eval.audio import load_wav


def main(raw: str, cands: list[str]) -> None:
    r = load_wav(raw)
    ref = transcribe(r.samples, r.sr)
    for c in cands:
        a = load_wav(c)
        g = guard_verdict(ref, transcribe(a.samples, a.sr))
        print(f"--- {raw.split('/')[-1]} vs {c}: {g.verdict}  WER {g.wer:.1%}  {'; '.join(g.reasons)}")
        if g.ops:
            print(format_ops(g.ops))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
