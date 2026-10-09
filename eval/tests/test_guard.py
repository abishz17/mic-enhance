"""Invented-word guard. Uses faster-whisper small.en (downloaded on first run
into eval/models/whisper, ~480 MB)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import SR
from mic_eval.asr_guard import Transcript, Word, guard_verdict, normalize_word, speech_presence_check, transcribe
from mic_eval.audio import load_wav


def _t(words: list[tuple[str, float]]) -> Transcript:
    return Transcript(" ".join(w for w, _ in words), [Word(w, w, i * 0.5, i * 0.5 + 0.4, p) for i, (w, p) in enumerate(words)], "x")


# ---------------------------------------------------------------- pure logic


def test_normalization() -> None:
    assert normalize_word("Don’t,") == "don't"
    assert normalize_word("3") == "three" and normalize_word("42") == "forty two"
    assert normalize_word("...") == ""


def test_rule_low_confidence_disagreement_passes() -> None:
    base = [(f"w{i}", 0.99) for i in range(10)]
    ref = _t(base + [("manin", 0.30)])
    cand = _t(base + [("manim", 0.40)])
    g = guard_verdict(ref, cand)
    assert g.verdict == "PASS" and g.substitutions == 1


def test_rule_confident_word_change_fails_even_at_tiny_wer() -> None:
    words = [(f"w{i}", 0.99) for i in range(200)]
    cand = list(words)
    cand[100] = ("cannot", 0.40)  # raw word was heard at p=0.99
    g = guard_verdict(_t(words), _t(cand))
    assert g.wer == pytest.approx(0.005)
    assert g.verdict == "FAIL"


def test_rule_words_in_silence_fail() -> None:
    g = guard_verdict(_t([]), _t([("thanks", 0.95)]))
    assert g.verdict == "FAIL"


# ---------------------------------------------------------------- with Whisper


@pytest.fixture(scope="module")
def clean_take(smoke):  # type: ignore[no-untyped-def]
    x = load_wav(smoke / "clean.wav").samples
    return x, transcribe(x, SR)


def test_identical_audio_passes(clean_take) -> None:  # type: ignore[no-untyped-def]
    x, ref = clean_take
    g = guard_verdict(ref, transcribe(x.copy(), SR))
    assert g.verdict == "PASS" and g.wer == 0.0
    assert speech_presence_check(x, x, SR).verdict == "PASS"


def test_benign_gain_passes(clean_take) -> None:  # type: ignore[no-untyped-def]
    x, ref = clean_take
    y = x * 0.25
    assert guard_verdict(ref, transcribe(y, SR)).verdict == "PASS"
    assert speech_presence_check(x, y, SR).verdict == "PASS"


def test_spliced_out_word_fails(clean_take) -> None:  # type: ignore[no-untyped-def]
    x, ref = clean_take
    w = next(w for w in ref.words if w.text == "rainbow")
    y = np.concatenate([x[: int(w.start * SR)], x[int(w.end * SR) :]])
    g = guard_verdict(ref, transcribe(y, SR))
    assert g.verdict == "FAIL", g.ops
    assert any(o.kind in ("del", "sub") and "rainbow" in o.ref for o in g.ops)


def test_word_pasted_into_pause_fails_presence_check(clean_take) -> None:  # type: ignore[no-untyped-def]
    # Whisper tends to drop an isolated pasted word; the ASR-free check must catch it.
    x, ref = clean_take
    w = next(w for w in ref.words if w.text == "prism")
    seg = x[int(w.start * SR) : int(w.end * SR)]
    y = x.copy()
    y[int(0.5 * SR) : int(0.5 * SR) + len(seg)] = seg  # leading 2 s of clean.wav is silence
    pr = speech_presence_check(x, y, SR)
    assert pr.verdict == "FAIL" and len(pr.events) == 1
    assert 0.4 <= pr.events[0].start <= 0.7


# ---------------------------------------------------------------- re-segmentation regressions
# Seen on a real laptop-mic recording vs the same file after an 80 Hz HPF.
# Raw: "three blue one bronze" (p 0.95); filtered: "3blue1prons" (p 0.46),
# previously reported as sub + confident deletion of "blue one bronze".


def _timed(words: list[tuple[str, float, float, float]]) -> Transcript:
    out = []
    for raw, s0, e, p in words:
        for t in normalize_word(raw).split():
            out.append(Word(t, raw, s0, e, p))
    return Transcript(" ".join(w.text for w in out), out, "x")


CTX_BEFORE = [(f"pre{i}", 20 + 0.4 * i, 20.3 + 0.4 * i, 0.99) for i in range(10)]
CTX_AFTER = [(f"post{i}", 27 + 0.4 * i, 27.3 + 0.4 * i, 0.99) for i in range(10)]
RAW_3B1B = CTX_BEFORE + [("three", 25.0, 25.3, 0.95), ("blue", 25.3, 25.6, 0.95), ("one", 25.6, 25.8, 0.95), ("bronze", 25.8, 26.2, 0.95)] + CTX_AFTER


def test_merged_number_token_normalizes_to_same_words() -> None:
    assert normalize_word("3Blue1Brown") == "three blue one brown"
    assert normalize_word("3blue1prons") == "three blue one prons"


def test_whisper_resegmentation_3blue1prons_is_not_confident() -> None:
    cand = _timed(CTX_BEFORE + [("3blue1prons", 25.0, 26.2, 0.46)] + CTX_AFTER)
    g = guard_verdict(_timed(RAW_3B1B), cand)
    assert g.verdict == "PASS", [(o.kind, o.ref, o.cand, o.confident) for o in g.ops]


def test_unsplittable_merged_token_deletions_overlap_and_pass() -> None:
    # same failure without digits: one fused token covering the 4 raw words
    cand = _timed(CTX_BEFORE + [("threeblueoneprons", 25.0, 26.2, 0.46)] + CTX_AFTER)
    g = guard_verdict(_timed(RAW_3B1B), cand)
    assert any(o.kind == "del" for o in g.ops) or any(len(o.ref) > 1 for o in g.ops)
    assert g.verdict == "PASS", [(o.kind, o.ref, o.cand, o.confident) for o in g.ops]


def test_real_deletion_without_overlap_still_fails() -> None:
    cand = _timed(CTX_BEFORE + [("three", 25.0, 25.3, 0.95), ("blue", 25.3, 25.6, 0.95), ("one", 25.6, 25.8, 0.95)] + CTX_AFTER)
    assert guard_verdict(_timed(RAW_3B1B), cand).verdict == "FAIL"  # "bronze" gone, nothing in its place


def test_confident_or_dissimilar_substitution_still_fails() -> None:
    swap = lambda w, p: _timed(CTX_BEFORE + [("three", 25.0, 25.3, 0.95), ("blue", 25.3, 25.6, 0.95), ("one", 25.6, 25.8, 0.95), (w, 25.8, 26.2, p)] + CTX_AFTER)  # noqa: E731
    assert guard_verdict(_timed(RAW_3B1B), swap("brown", 0.95)).verdict == "FAIL"  # clearly heard different word
    assert guard_verdict(_timed(RAW_3B1B), swap("kettle", 0.40)).verdict == "FAIL"  # unsure but not a sound-alike
    assert guard_verdict(_timed(RAW_3B1B), swap("prons", 0.40)).verdict == "PASS"  # unsure sound-alike = re-hearing


def test_negation_change_is_never_excused_as_rehearing() -> None:
    raw = _timed(CTX_BEFORE + [("i", 25.0, 25.2, 0.95), ("can", 25.2, 25.5, 0.95), ("go", 25.5, 25.8, 0.95)] + CTX_AFTER)
    cand = _timed(CTX_BEFORE + [("i", 25.0, 25.2, 0.95), ("can't", 25.2, 25.5, 0.50), ("go", 25.5, 25.8, 0.95)] + CTX_AFTER)
    assert guard_verdict(raw, cand).verdict == "FAIL"
