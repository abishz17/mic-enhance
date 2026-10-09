"""Known-answer tests for BS.1770-4 loudness and true peak.

Reference values: BS.1770-4 sec. 2.1 / EBU Tech 3341 - a 0 dBFS-peak 997 Hz
sine in one channel reads -3.01 LKFS. Mono here is ONE channel (weight 1.0),
so a -20 dBFS-peak 1 kHz sine must read -23.01 LUFS. (Tools that treat mono
as dual-mono, e.g. ffmpeg ebur128 dualmono=true, read +3.01 dB higher.)
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, sine
from mic_eval.signal_metrics import integrated_lufs, sample_peak_dbfs, true_peak_dbtp


@pytest.mark.parametrize("level", [0.0, -20.0, -33.0])
def test_sine_1k_reads_level_minus_3_01(level: float) -> None:
    x = sine(997.0, level, 20.0)
    assert integrated_lufs(x, SR) == pytest.approx(level - 3.01, abs=0.05)


# EBU Tech 3341 Table 1 (stereo, per-channel dBFS, all expect -23.0 +/- 0.1 LUFS).
# Mono equivalent of a stereo pair at L dBFS per channel is L + 3.01 dBFS.
TECH3341 = {
    "case3": [(-36, 10), (-23, 60), (-36, 10)],
    "case4": [(-72, 10), (-36, 10), (-23, 60), (-36, 10), (-72, 10)],
    "case5": [(-26, 20), (-20, 20.1), (-26, 20)],
}


@pytest.mark.parametrize("case", sorted(TECH3341))
def test_gating_tech3341_mono_equivalent(case: str) -> None:
    x = np.concatenate([sine(1000, lvl + 3.0103, dur) for lvl, dur in TECH3341[case]])
    assert integrated_lufs(x, SR) == pytest.approx(-23.0, abs=0.1)


def test_absolute_gate_removes_sub_minus_70_content() -> None:
    # Room tone at -80 dBFS sits below the -70 LUFS absolute gate, so it must
    # read exactly like digital silence. (Both read ~-23.17, not -23.01: the
    # 400 ms blocks straddling the sine's edges are partly empty yet pass the
    # gates; that is correct BS.1770 behaviour.)
    rng = np.random.default_rng(0)
    tone = sine(1000, -20, 10)
    quiet = rng.standard_normal(10 * SR) * 10 ** (-80 / 20)
    a = integrated_lufs(np.concatenate([np.zeros(10 * SR), tone, np.zeros(10 * SR)]), SR)
    b = integrated_lufs(np.concatenate([quiet, tone, quiet]), SR)
    assert a == pytest.approx(b, abs=0.01)
    assert a == pytest.approx(-23.01, abs=0.2)


def test_too_short_returns_none_and_silence_is_minus_inf() -> None:
    assert integrated_lufs(np.zeros(1000), SR) is None
    assert integrated_lufs(np.zeros(SR), SR) == -np.inf


def test_true_peak_catches_intersample_peak() -> None:
    # fs/4 sine at 45 deg phase: every sample is +-0.7071, the waveform peaks at 1.0.
    x = sine(SR / 4, 0.0, 1.0, phase=np.pi / 4)
    x[:480] *= np.hanning(960)[:480]  # 10 ms fades: an abrupt start would add Gibbs overshoot
    x[-480:] *= np.hanning(960)[480:]
    assert sample_peak_dbfs(x) == pytest.approx(-3.01, abs=0.01)
    assert true_peak_dbtp(x) == pytest.approx(0.0, abs=0.05)


def test_true_peak_worst_case_phase_under_reads_at_most_0_7_db() -> None:
    worst = max(-true_peak_dbtp(sine(SR / 4 - 100, 0.0, 1.0, phase=ph)) for ph in np.linspace(0, np.pi / 2, 9))
    assert worst < 0.7


def test_true_peak_equals_sample_peak_for_low_freq() -> None:
    x = sine(100.0, -6.0, 1.0)
    assert true_peak_dbtp(x) == pytest.approx(-6.0, abs=0.01)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_matches_ffmpeg_ebur128(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    # speech-like: pink-ish noise bursts with pauses
    n = rng.standard_normal(30 * SR)
    n = np.convolve(n, np.ones(8) / 8, mode="same") * 0.1
    env = (np.sin(2 * np.pi * 0.7 * np.arange(len(n)) / SR) > 0).astype(float)
    x = n * env
    p = tmp_path / "x.wav"
    sf.write(p, x, SR, subtype="FLOAT")
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(p), "-af", "ebur128", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stderr
    summary = out[out.rindex("Summary:"):]
    ff = float(summary.split("I:")[1].split("LUFS")[0])
    assert integrated_lufs(x, SR) == pytest.approx(ff, abs=0.1)
