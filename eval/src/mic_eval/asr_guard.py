"""Invented-word guard: transcribe raw and processed audio, diff the words.

Decision rule (see `guard_verdict`), thresholds from eval/tools/calibrate_guard.py
on real recordings:
  FAIL if any single edit is "confident": a deleted/substituted raw word that
  Whisper heard with p >= conf=0.90, or an inserted word heard with p >= 0.90; OR if WER > max_wer=25 % (gross
  failure only: benign gain/EQ/noise/LPF changes alone produced up to 16 % WER
  on casual laptop-mic speech, all of it on words with p < 0.9).
Plus an ASR-free check, `speech_presence_check`, because Whisper drops
isolated words that were pasted into pauses.
Rationale: the reference transcript comes from the *raw* (noisy) file, so
some disagreement is ASR noise, which concentrates on words Whisper itself
was unsure of. A word deleted/changed/added by a generative model in clearly
audible speech shows up as a high-confidence edit, and must fail even in a
long file where it is only 0.1 % WER.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from .audio import resample

WHISPER_DIR = Path(__file__).resolve().parents[2] / "models" / "whisper"
DEFAULT_MODEL = "small.en"
DEFAULT_MAX_WER = 0.25
DEFAULT_CONF = 0.90


@dataclass
class Word:
    text: str  # normalized
    raw: str
    start: float
    end: float
    prob: float


@dataclass
class Transcript:
    text: str
    words: list[Word]
    model: str


@dataclass
class EditOp:
    kind: str  # "sub" | "del" | "ins"
    ref: list[str]
    cand: list[str]
    ref_time: tuple[float, float] | None
    cand_time: tuple[float, float] | None
    min_prob: float  # min Whisper word prob over all words involved
    ref_max_prob: float  # most confident raw-side word (0 for insertions)
    cand_min_prob: float  # least confident cand-side word (0 for deletions)
    confident: bool


@dataclass
class GuardResult:
    wer: float
    n_ref_words: int
    substitutions: int
    deletions: int
    insertions: int
    ops: list[EditOp]
    verdict: str  # "PASS" | "FAIL"
    reasons: list[str]
    ref_text: str
    cand_text: str
    model: str
    max_wer: float
    conf: float
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ------------------------------------------------------------------ text

_APOS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_word(n: int) -> str:
    """Hand-written speller for 0..9999 (enough for spoken numbers in a take)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + " hundred" + ("" if r == 0 else " " + _num_word(r))
    th, r = divmod(n, 1000)
    return _num_word(th) + " thousand" + ("" if r == 0 else " " + _num_word(r))


def _spell_digits(d: str) -> str:
    if len(d) <= 4 and not (len(d) > 1 and d.startswith("0")):
        return _num_word(int(d))
    return " ".join(_ONES[int(c)] for c in d)  # phone numbers, codes: digit by digit


def normalize_word(w: str) -> str:
    """Normalize one ASR token to zero or more comparison words:
    NFKC + lowercase; hyphens/underscores -> space; split letter/digit
    boundaries ("3blue1prons" -> "3 blue 1 prons"); spell digits ("3" ->
    "three"); strip punctuation except in-word apostrophes. So Whisper's
    formatting choices ("3Blue1Brown" vs "three blue one brown") are not
    counted as word changes."""
    w = unicodedata.normalize("NFKC", w).translate(_APOS).lower()
    w = re.sub(r"[-_/]+", " ", w)
    w = re.sub(r"(?<=\d)(?=[^\W\d])|(?<=[^\W\d])(?=\d)", " ", w)
    out: list[str] = []
    for tok in w.split():
        tok = re.sub(r"[^\w']+", "", tok).strip("'")
        if not tok:
            continue
        out.append(_spell_digits(tok) if tok.isdigit() else tok)
    return " ".join(out)


# ------------------------------------------------------------------ ASR


@lru_cache(maxsize=2)
def _model(name: str):  # type: ignore[no-untyped-def]
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8", download_root=str(WHISPER_DIR))


def transcribe(x: np.ndarray, sr: int, model: str = DEFAULT_MODEL) -> Transcript:
    """Deterministic decoding: beam 5, temperature 0 only (no sampling
    fallback), no conditioning on previous text (limits hallucination
    cascades), no VAD (VAD would hide words invented inside "silence")."""
    audio = resample(x, sr, 16_000).astype(np.float32)
    segs, _info = _model(model).transcribe(
        audio,
        language="en",
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=True,
        vad_filter=False,
    )
    words: list[Word] = []
    for s in segs:
        for w in s.words or []:
            for piece in w.word.split():
                for t in normalize_word(piece).split():
                    words.append(Word(t, piece, float(w.start), float(w.end), float(w.probability)))
    return Transcript(" ".join(w.text for w in words), words, model)


# ------------------------------------------------------------------ diff


def _span(ws: list[Word]) -> tuple[float, float] | None:
    return (round(ws[0].start, 2), round(ws[-1].end, 2)) if ws else None


def _near(word: Word, others: list[Word], window_s: float) -> bool:
    """Same normalized word within +/- window_s on the other side?"""
    return any(o.text == word.text and abs(o.start - word.start) <= window_s for o in others)


_NEGATIONS = {"no", "not", "never", "none", "nothing", "nobody", "nor", "neither", "cannot"}
_VOICING = str.maketrans({"b": "p", "d": "t", "g": "k", "c": "k", "q": "k", "z": "s", "v": "f", "x": "k"})


def _sound_fold(t: str) -> str:
    """Crude sound-alike key: fold voiced/unvoiced consonant pairs, keep the
    first letter, drop later vowels ("bronze" -> "prns" == "prons" -> "prns")."""
    t = t.replace(" ", "").translate(_VOICING)
    return t[:1] + re.sub(r"[aeiouyhw]", "", t[1:])


def _char_similarity(a: str, b: str) -> float:
    """max(spelling similarity, sound-alike similarity), 0..1."""
    plain = Levenshtein.normalized_similarity(a.replace(" ", ""), b.replace(" ", ""))
    return float(max(plain, Levenshtein.normalized_similarity(_sound_fold(a), _sound_fold(b))))


def _overlaps(w: Word, spans: list[tuple[float, float]], pad: float) -> bool:
    return any(w.start - pad < e and s0 < w.end + pad for s0, e in spans)


def diff_words(
    ref: list[Word],
    cand: list[Word],
    conf: float = DEFAULT_CONF,
    window_s: float = 1.0,
    overlap_pad_s: float = 0.15,
    sound_alike: float = 0.8,
) -> list[EditOp]:
    """Word-level Levenshtein diff (on normalized words). An edit is
    "confident" (= evidence of a real word change) when:
    - del: a raw word heard with p >= conf is missing from the candidate
      (not found within +/- window_s) AND its time span (+/- 150 ms) does not
      overlap a substituted/inserted candidate word. Overlap means Whisper
      re-segmented the same sound ("three blue one bronze" -> "3blue1prons"),
      not that the word was removed.
    - sub: a raw word with p >= conf is missing from the candidate, unless the
      candidate side is a low-confidence (p < conf) sound-alike of the raw
      side (spelling or crude sound-alike similarity >= 0.8, e.g. bronze ->
      prons) and no negation was added/removed ("can" -> "can't" always counts): that is the
      ASR re-hearing the same audio. A confident or dissimilar replacement
      (specific -> bad, seashells -> seashore at p 0.90) still counts.
    - ins: every added word has p >= conf and none is in the raw transcript
      within +/- window_s (a mumbled word made clearer is not invented).
    """
    rt = [w.text for w in ref]
    ct = [w.text for w in cand]
    opcodes = [op for op in Levenshtein.opcodes(rt, ct) if op.tag != "equal"]
    changed_cand_spans = [
        (w.start, w.end) for op in opcodes if op.tag in ("replace", "insert") for w in cand[op.dest_start : op.dest_end]
    ]
    ops: list[EditOp] = []
    for op in opcodes:
        rw = ref[op.src_start : op.src_end]
        cw = cand[op.dest_start : op.dest_end]
        kind = {"replace": "sub", "delete": "del", "insert": "ins"}[op.tag]
        probs = [w.prob for w in rw + cw]
        mp = min(probs) if probs else 0.0
        rmax = max((w.prob for w in rw), default=0.0)
        cmin = min((w.prob for w in cw), default=0.0)
        if kind == "del":
            confident = any(
                w.prob >= conf and not _near(w, cand, window_s) and not _overlaps(w, changed_cand_spans, overlap_pad_s) for w in rw
            )
        elif kind == "sub":
            lost = any(w.prob >= conf and not _near(w, cand, window_s) for w in rw)
            # compare with the op's raw words AND with all raw words under the
            # cand span in time, so a merged token ("threeblueoneprons") is
            # matched against the whole phrase it re-hears.
            c_txt = " ".join(w.text for w in cw)
            under = [w.text for w in ref if _overlaps(w, [(cw[0].start, cw[-1].end)], overlap_pad_s)]
            sim = max(_char_similarity(" ".join(w.text for w in rw), c_txt), _char_similarity(" ".join(under), c_txt) if under else 0.0)
            # never excuse a change in negation ("can" -> "can't", "do" -> "don't")
            neg = lambda ws: {w.text for w in ws if w.text in _NEGATIONS or w.text.endswith("n't")}  # noqa: E731
            rehearing = cmin < conf and sim >= sound_alike and neg(rw) == neg(cw)
            confident = lost and not rehearing
        else:
            confident = cmin >= conf and not any(_near(w, ref, window_s) for w in cw)
        ops.append(
            EditOp(kind, [w.text for w in rw], [w.text for w in cw], _span(rw), _span(cw), round(mp, 3), round(rmax, 3), round(cmin, 3), confident)
        )
    return ops


def guard_verdict(
    ref: Transcript,
    cand: Transcript,
    max_wer: float = DEFAULT_MAX_WER,
    conf: float = DEFAULT_CONF,
) -> GuardResult:
    rt = [w.text for w in ref.words]
    ct = [w.text for w in cand.words]
    ops = diff_words(ref.words, cand.words, conf)
    ed = Levenshtein.editops(rt, ct)
    s = sum(1 for e in ed if e.tag == "replace")
    d = sum(1 for e in ed if e.tag == "delete")
    i = sum(1 for e in ed if e.tag == "insert")
    n = len(rt)
    if n == 0:
        wer = 0.0 if not ct else float("inf")
    else:
        wer = (s + d + i) / n
    reasons: list[str] = []
    if wer > max_wer:
        reasons.append(f"WER {wer:.1%} > {max_wer:.1%}")
    conf_ops = [o for o in ops if o.confident]
    if conf_ops:
        reasons.append(f"{len(conf_ops)} confident word edit(s) (p >= {conf:.2f})")
    return GuardResult(
        wer=wer,
        n_ref_words=n,
        substitutions=s,
        deletions=d,
        insertions=i,
        ops=ops,
        verdict="FAIL" if reasons else "PASS",
        reasons=reasons,
        ref_text=ref.text,
        cand_text=cand.text,
        model=ref.model,
        max_wer=max_wer,
        conf=conf,
    )


def format_ops(ops: list[EditOp]) -> str:
    lines = []
    for o in ops:
        t = o.ref_time or o.cand_time
        at = f"{t[0]:7.2f}s" if t else "      ?"
        mark = "!!" if o.confident else "  "
        if o.kind == "sub":
            body = f"[-{' '.join(o.ref)}-] {{+{' '.join(o.cand)}+}}"
        elif o.kind == "del":
            body = f"[-{' '.join(o.ref)}-]"
        else:
            body = f"{{+{' '.join(o.cand)}+}}"
        p = f"raw p={o.ref_max_prob:.2f}" if o.kind != "ins" else f"new p={o.cand_min_prob:.2f}"
        lines.append(f"  {mark} {at}  {o.kind:3s}  {body}   ({p})")
    return "\n".join(lines)


# ------------------------------------------------------------------ non-ASR check


@dataclass
class PresenceEvent:
    start: float
    end: float
    rise_db: float  # how much louder cand is than raw there, after level matching


@dataclass
class PresenceResult:
    events: list[PresenceEvent]
    lag_samples: int
    duration_delta_s: float
    gain_offset_db: float
    verdict: str
    reasons: list[str]


def speech_presence_check(
    ref: np.ndarray,
    cand: np.ndarray,
    sr: int,
    rise_db: float = 12.0,
    min_event_s: float = 0.10,
) -> PresenceResult:
    """Flag new speech-band sound where the raw file has a pause.

    Whisper's language model tends to drop an isolated, out-of-context word
    (measured: a single word pasted into a pause is not transcribed), so the
    ASR diff cannot be the only guard against invented audio. This check is
    ASR-free and assumes processing preserves timing (both our modes do):
      1. align by cross-correlation, 20 ms frames of 300-3400 Hz energy;
      2. level-match by the median cand-raw difference over raw speech frames;
      3. a frame is flagged if raw is in a pause (< raw p10 + 10 dB), cand is
         clearly active (> cand p10 + 20 dB) and cand exceeds raw by `rise_db`;
      4. runs of flagged frames >= `min_event_s` are events -> FAIL.
    """
    from scipy.signal import butter, sosfilt

    from .intrusive import align

    r, c, lag = align(ref, cand, sr)
    dur_delta = (len(cand) - len(ref)) / sr
    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    hop = int(0.02 * sr)

    def frames_db(x: np.ndarray) -> np.ndarray:
        y = sosfilt(sos, x)
        n = len(y) // hop
        e = np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1)
        return 10 * np.log10(e + 1e-20)

    rd, cd = frames_db(r), frames_db(c)
    r_floor, c_floor = np.percentile(rd, 10), np.percentile(cd, 10)
    speech = rd > r_floor + 20
    offset = float(np.median(cd[speech] - rd[speech])) if speech.sum() >= 10 else float(np.median(cd - rd))
    flag = (rd < r_floor + 10) & (cd > c_floor + 20) & (cd - offset > rd + rise_db)
    events: list[PresenceEvent] = []
    min_frames = max(1, int(round(min_event_s / 0.02)))
    i = 0
    while i < len(flag):
        if flag[i]:
            j = i
            while j + 1 < len(flag) and (flag[j + 1] or (j + 2 < len(flag) and flag[j + 2])):
                j += 1
            if j - i + 1 >= min_frames:
                rise = float(np.max(cd[i : j + 1] - offset - rd[i : j + 1]))
                events.append(PresenceEvent(round(i * 0.02, 2), round((j + 1) * 0.02, 2), round(rise, 1)))
            i = j + 1
        else:
            i += 1
    reasons = []
    if events:
        reasons.append(f"{len(events)} region(s) of new speech-band sound where raw is a pause")
    if abs(dur_delta) > 0.05:
        reasons.append(f"duration changed by {dur_delta:+.3f} s (processing should preserve timing)")
    return PresenceResult(events, lag, round(dur_delta, 4), round(offset, 2), "FAIL" if reasons else "PASS", reasons)
