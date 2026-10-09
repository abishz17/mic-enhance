from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from conftest import SR, sine
from mic_eval.cli import main


def test_score_json_on_good_file(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "s.wav"
    sf.write(p, sine(997, -20, 5), SR, subtype="PCM_24")
    assert main(["score", "--json", "--no-dnsmos", str(p)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert abs(rows[0]["lufs_i"] - (-23.01)) < 0.05


def test_bad_inputs_exit_2_with_message(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    t = tmp_path / "t.wav"
    t.write_text("nope")
    assert main(["score", "--no-dnsmos", str(t), str(tmp_path / "missing.wav")]) == 2
    err = capsys.readouterr().err
    assert "not a readable audio file" in err and "file not found" in err


def test_strict_fails_off_target(tmp_path: Path) -> None:
    p = tmp_path / "s.wav"
    sf.write(p, sine(997, -20, 5), SR)
    assert main(["score", "--strict", "--no-dnsmos", str(p)]) == 1  # -23 LUFS != -14


def test_report_writes_local_html(tmp_path: Path) -> None:
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(a, sine(220, -20, 3), SR)
    sf.write(b, sine(220, -14, 3), SR)
    out = tmp_path / "r.html"
    assert main(["report", "--no-dnsmos", "--no-voice", str(a), str(b), "-o", str(out)]) == 0
    html = out.read_text()
    assert html.count("<audio") == 2 and "data:audio/flac;base64," in html and "src=\"http" not in html
