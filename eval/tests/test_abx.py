from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, sine
from mic_eval.abx import AbxResult, Trial, loudness_match, run_session
from mic_eval.signal_metrics import integrated_lufs, true_peak_dbtp


def test_loudness_match_equalizes_and_never_boosts_over_ceiling() -> None:
    a = sine(440, -1.0, 5)  # loud
    b = sine(440, -30.0, 5)  # quiet
    m = loudness_match(a, b, SR)
    la, lb = integrated_lufs(m.a, SR), integrated_lufs(m.b, SR)
    assert la == pytest.approx(lb, abs=0.05)
    assert max(true_peak_dbtp(m.a), true_peak_dbtp(m.b)) <= -1.0 + 1e-6
    assert m.gain_b_db <= 0.0 + 1e-9  # quieter file is not boosted


def test_binomial_p_values() -> None:
    r = AbxResult("abx", "a", "b", [Trial(i, "A", "A", True, [], 0.0) for i in range(10)])
    assert r.p_value == pytest.approx(0.5**10)  # 10/10 correct
    r = AbxResult("abx", "a", "b", [Trial(i, "A", "A" if i < 5 else "B", i < 5, [], 0.0) for i in range(10)])
    assert r.p_value == pytest.approx(0.623, abs=0.001)  # 5/10 = guessing


def test_scripted_abx_session(tmp_path: Path) -> None:
    pa, pb = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(pa, sine(440, -6, 3), SR)
    sf.write(pb, sine(660, -20, 3.5), SR)  # different level AND length
    lengths: list[int] = []
    hidden: list[str] = []

    def player(p: Path) -> None:
        x, _ = sf.read(p)
        lengths.append(len(x))
        hidden.append("A" if abs(np.argmax(np.abs(np.fft.rfft(x))) / len(x) * SR - 440) < 5 else "B")

    step = {"n": 0}

    def ask(_prompt: str) -> str:
        # perfect listener: play X, then name the file X turned out to be
        step["n"] += 1
        return "x" if step["n"] % 2 == 1 else hidden[-1]

    log = tmp_path / "log.jsonl"
    res = run_session(str(pa), str(pb), trials=8, seed=1, log=log, player=player, ask=ask, say=lambda _s: None)
    assert res.n == 8 and res.k == 8
    assert res.p_value == pytest.approx(0.5**8)
    assert log.read_text().count("\n") == 1
    # both temp files were cropped to the same length and loudness-matched
    assert set(lengths) == {3 * SR}
