"""Calibrate the invented-word guard on real recordings.

For each input file: transcribe the original, then (a) benign transforms that
must PASS (gain, EQ, added noise, 8 kHz lowpass, 16 kHz round trip) and
(b) word edits that must FAIL: splice a word out (changes length), paste a
word into the longest pause (length-preserving overwrite), replace a word with
another word's audio (length-preserving). Prints, per case, the ASR guard
(WER, confident edits) and the ASR-free presence check separately.

    uv run --project eval python eval/tools/calibrate_guard.py corpus/*.wav
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.signal import butter, sosfilt

from mic_eval.asr_guard import guard_verdict, speech_presence_check, transcribe
from mic_eval.audio import load_wav, resample

SR = 48_000


def xfade_cut(x: np.ndarray, a: int, b: int, fade: int = 240) -> np.ndarray:
    w = np.linspace(1, 0, fade)
    head, tail = x[:a].copy(), x[b:].copy()
    head[-fade:] = head[-fade:] * w + tail[:fade] * (1 - w)
    return np.concatenate([head, tail[fade:]])


def main(paths: list[str]) -> None:
    rng = np.random.default_rng(0)
    for p in paths:
        x = load_wav(p).samples
        ref = transcribe(x, SR)
        rms = np.sqrt(np.mean(x**2))
        benign = {
            "gain +12 dB": np.clip(x * 4, -1, 1),
            "HPF 100 Hz + presence": sosfilt(butter(2, 100, "hp", fs=SR, output="sos"), x)
            + 0.5 * sosfilt(butter(2, [3000, 6000], "bp", fs=SR, output="sos"), x),
            "noise @ 25 dB SNR": x + rng.standard_normal(len(x)) * rms * 10 ** (-25 / 20),
            "LPF 8 kHz": sosfilt(butter(8, 8000, fs=SR, output="sos"), x),
            "16 kHz round trip": resample(resample(x, SR, 16000), 16000, SR)[: len(x)],
        }
        cands = [w for w in ref.words if w.prob >= 0.9 and len(w.text) >= 4]
        picks = [cands[i] for i in rng.choice(len(cands), size=min(3, len(cands)), replace=False)]
        edits: dict[str, np.ndarray] = {}
        for w in picks:
            edits[f"splice out '{w.text}'"] = xfade_cut(x, int(w.start * SR), int(w.end * SR))
        W = ref.words
        if len(W) > 8 and picks:
            gap, gi = max((W[i + 1].start - W[i].end, i) for i in range(2, len(W) - 2))
            w = picks[0]
            seg = x[int(w.start * SR) : int(w.end * SR)]
            mid = int((W[gi].end + W[gi + 1].start) / 2 * SR)
            y = x.copy()
            a0 = max(0, mid - len(seg) // 2)
            y[a0 : a0 + len(seg)] = seg[: len(y) - a0]
            edits[f"paste '{w.text}' into {gap:.2f}s pause"] = y
            # substitution: overwrite word k with another word's audio, same length
            k, d = picks[1] if len(picks) > 1 else picks[0], picks[-1]
            if k is not d:
                y = x.copy()
                ks, ke = int(k.start * SR), int(k.end * SR)
                src = x[int(d.start * SR) : int(d.end * SR)]
                rep = np.zeros(ke - ks)
                rep[: min(len(src), len(rep))] = src[: len(rep)]
                y[ks:ke] = rep
                edits[f"replace '{k.text}' with '{d.text}'"] = y
        print(f"\n=== {p}: {len(ref.words)} words")
        for kind, cases in (("benign (want PASS)", benign), ("edit (want FAIL)", edits)):
            for name, y in cases.items():
                g = guard_verdict(ref, transcribe(y, SR))
                pr = speech_presence_check(x, y, SR)
                nconf = sum(o.confident for o in g.ops)
                both = "FAIL" if "FAIL" in (g.verdict, pr.verdict) else "PASS"
                ops = "; ".join(f"{o.kind}:{' '.join(o.ref)}->{' '.join(o.cand)}(p{o.min_prob:.2f})" for o in g.ops)
                print(f"  {kind:18s} {name:34s} WER {g.wer:6.1%} conf {nconf} ASR {g.verdict}  presence {pr.verdict} ({len(pr.events)} ev, dur {pr.duration_delta_s:+.2f}s) => {both}   {ops}")


if __name__ == "__main__":
    main(sys.argv[1:])
