"""Blind ABX / AB-preference listening test in the terminal (macOS afplay).

Both files are loudness-matched (BS.1770 integrated) and cropped to the same
length before any trial, so neither level nor duration gives the answer away.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.stats import binomtest

from .audio import AudioError, load_wav
from .signal_metrics import integrated_lufs, true_peak_dbtp

MAX_TP = -1.0


@dataclass
class Matched:
    a: np.ndarray
    b: np.ndarray
    sr: int
    lufs_a_before: float
    lufs_b_before: float
    lufs_after: float
    gain_a_db: float
    gain_b_db: float


def loudness_match(a: np.ndarray, b: np.ndarray, sr: int, max_tp: float = MAX_TP) -> Matched:
    """Gain both to the quieter one's integrated loudness (only attenuation,
    so no new clipping); if either would exceed `max_tp` dBTP, lower both."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    la, lb = integrated_lufs(a, sr), integrated_lufs(b, sr)
    if la is None or lb is None or not (math.isfinite(la) and math.isfinite(lb)):
        raise AudioError("cannot loudness-match: one excerpt is silent or shorter than 400 ms")
    target = min(la, lb)
    ga, gb = target - la, target - lb
    tp = max(true_peak_dbtp(a) + ga, true_peak_dbtp(b) + gb)
    if tp > max_tp:
        ga -= tp - max_tp
        gb -= tp - max_tp
    a2 = a * 10 ** (ga / 20)
    b2 = b * 10 ** (gb / 20)
    after = integrated_lufs(a2, sr)
    assert after is not None
    return Matched(a2, b2, sr, la, lb, after, ga, gb)


@dataclass
class Trial:
    n: int
    hidden: str  # ABX: which file X was ("A"/"B"); AB: which file was option 1
    answer: str  # ABX: "A"/"B"; AB: the file preferred ("A"/"B")
    correct: bool | None
    plays: list[str]
    seconds: float


@dataclass
class AbxResult:
    mode: str
    file_a: str
    file_b: str
    trials: list[Trial] = field(default_factory=list)
    gain_a_db: float = 0.0
    gain_b_db: float = 0.0
    lufs_after: float = 0.0

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def k(self) -> int:
        if self.mode == "abx":
            return sum(1 for t in self.trials if t.correct)
        return sum(1 for t in self.trials if t.answer == "A")

    @property
    def p_value(self) -> float:
        if self.n == 0:
            return 1.0
        if self.mode == "abx":  # H0: guessing; H1: can hear a difference
            return float(binomtest(self.k, self.n, 0.5, alternative="greater").pvalue)
        return float(binomtest(self.k, self.n, 0.5, alternative="two-sided").pvalue)

    def summary(self) -> str:
        if self.mode == "abx":
            s = f"ABX: {self.k}/{self.n} correct, one-sided binomial p = {self.p_value:.4f}"
            verdict = "audible difference (p < 0.05)" if self.p_value < 0.05 else "no reliable difference shown (p >= 0.05)"
        else:
            s = f"AB preference: A preferred {self.k}/{self.n}, B {self.n - self.k}/{self.n}, two-sided binomial p = {self.p_value:.4f}"
            if self.p_value < 0.05:
                verdict = "significant preference for " + ("A" if self.k > self.n / 2 else "B")
            else:
                verdict = "no significant preference"
        return f"{s}\n=> {verdict}\nA = {self.file_a}\nB = {self.file_b}"


def afplay(path: Path) -> None:
    if shutil.which("afplay") is None:
        raise RuntimeError("afplay not found (macOS only); use `mic-eval report` players instead")
    try:
        subprocess.run(["afplay", str(path)], check=False)
    except KeyboardInterrupt:  # Ctrl-C stops playback, not the test
        pass


def run_session(
    file_a: str,
    file_b: str,
    *,
    trials: int = 10,
    mode: str = "abx",
    start: float = 0.0,
    duration: float | None = None,
    seed: int | None = None,
    log: Path | None = None,
    player: Callable[[Path], None] = afplay,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> AbxResult:
    if mode not in ("abx", "ab"):
        raise ValueError("mode must be 'abx' or 'ab'")
    if trials < 1:
        raise ValueError("trials must be >= 1")
    A = load_wav(file_a)
    B = load_wav(file_b)
    i0 = int(start * A.sr)
    i1 = None if duration is None else i0 + int(duration * A.sr)
    m = loudness_match(A.samples[i0:i1], B.samples[i0:i1], A.sr)
    say(
        f"Loudness-matched: A {m.lufs_a_before:.2f} LUFS ({m.gain_a_db:+.2f} dB), "
        f"B {m.lufs_b_before:.2f} LUFS ({m.gain_b_db:+.2f} dB) -> both {m.lufs_after:.2f} LUFS, "
        f"{len(m.a) / m.sr:.1f} s each."
    )
    rng = random.Random(seed)
    res = AbxResult(mode, str(file_a), str(file_b), gain_a_db=m.gain_a_db, gain_b_db=m.gain_b_db, lufs_after=m.lufs_after)
    with tempfile.TemporaryDirectory(prefix="mic-abx-") as td:
        pa, pb = Path(td) / "a.wav", Path(td) / "b.wav"
        sf.write(pa, m.a, m.sr, subtype="PCM_24")
        sf.write(pb, m.b, m.sr, subtype="PCM_24")
        files = {"A": pa, "B": pb}
        for t in range(1, trials + 1):
            hidden = rng.choice("AB")
            plays: list[str] = []
            t0 = time.monotonic()
            if mode == "abx":
                say(f"\nTrial {t}/{trials}: [a] play A  [b] play B  [x] play X  then answer [A] or [B] (X is ...)  [q] quit")
                keys = {"a": files["A"], "b": files["B"], "x": files[hidden]}
            else:
                other = "B" if hidden == "A" else "A"
                say(f"\nTrial {t}/{trials}: [1] play first  [2] play second  then answer [P1] or [P2] (preferred)  [q] quit")
                keys = {"1": files[hidden], "2": files[other]}
            answer = ""
            while True:
                c = ask("> ").strip()
                if c.lower() == "q":
                    break
                if c.lower() in keys and not (mode == "abx" and c in ("A", "B")):
                    plays.append(c.lower())
                    player(keys[c.lower()])
                    continue
                if mode == "abx" and c in ("A", "B"):
                    answer = c
                    break
                if mode == "ab" and c.upper() in ("P1", "P2"):
                    answer = hidden if c.upper() == "P1" else ("B" if hidden == "A" else "A")
                    break
                say("  ? unknown input")
            if not answer:
                say("Stopped early.")
                break
            correct = (answer == hidden) if mode == "abx" else None
            res.trials.append(Trial(t, hidden, answer, correct, plays, round(time.monotonic() - t0, 2)))
            say("  recorded.")  # no feedback on correctness: avoids learning cues mid-test
    say("\n" + res.summary())
    if log is not None:
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**{k: v for k, v in asdict(res).items()}, "k": res.k, "n": res.n, "p_value": res.p_value}) + "\n")
        say(f"Logged to {log}")
    return res
