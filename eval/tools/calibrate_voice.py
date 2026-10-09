"""Calibrate voice-identity thresholds on real recordings.

Benign processing (must PASS) vs identity changes (must FAIL), per file:
speaker similarity (ECAPA cosine), F0 median shift / range change (st),
spectral tilt change (dB/oct). Also prints the cross-device same-speaker
similarity matrix as an anchor.

    uv run --project eval python eval/tools/calibrate_voice.py corpus/*.wav
"""

from __future__ import annotations

import sys
import warnings

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

from mic_eval.audio import load_wav, resample
from mic_eval.voice import speaker_embedding, voice_check

SR = 48_000
warnings.filterwarnings("ignore")


def main(paths: list[str]) -> None:
    rng = np.random.default_rng(0)
    emb = {p: speaker_embedding(load_wav(p).samples, SR) for p in paths}
    print("cross-file speaker similarity (different takes / devices):")
    for a in paths:
        print(f"  {a.split('/')[-1]:28s}" + " ".join(f"{float(np.dot(emb[a], emb[b])):6.3f}" for b in paths))
    for p in paths:
        x = load_wav(p).samples
        rms = np.sqrt(np.mean(x**2))
        sos = lambda *a, **k: butter(*a, fs=SR, output="sos", **k)  # noqa: E731
        benign = {
            "identical": x,
            "gain +12 dB": np.clip(x * 4, -1, 1),
            "HPF100 + presence": sosfilt(sos(2, 100, "hp"), x) + 0.5 * sosfilt(sos(2, [3000, 6000], "bp"), x),
            "low shelf -6 dB <300": x - 0.5 * sosfilt(sos(2, 300), x),
            "noise 25 dB SNR": x + rng.standard_normal(len(x)) * rms * 10 ** (-25 / 20),
            "LPF 8 kHz": sosfilt(sos(8, 8000), x),
            "16 kHz round trip": resample(resample(x, SR, 16000), 16000, SR)[: len(x)],
        }
        changes = {
            "resample +2 st": resample(x, int(SR * 2 ** (2 / 12)), SR),
            "resample -2 st": resample(x, int(SR * 2 ** (-2 / 12)), SR),
            "phase-vocoder +1 st": librosa.effects.pitch_shift(x, sr=SR, n_steps=1),
            "phase-vocoder -0.5 st": librosa.effects.pitch_shift(x, sr=SR, n_steps=-0.5),
            "formant-ish: tilt +2 dB/oct": librosa.istft(
                librosa.stft(x) * (np.maximum(np.fft.rfftfreq(2048, 1 / SR), 100) / 1000)[:, None] ** (2 / 6.02),
                length=len(x),
            ),
        }
        print(f"\n=== {p}")
        print(f"  {'case':28s} {'want':5s} {'spk sim':>8s} {'F0 st':>7s} {'range':>7s} {'tilt d':>7s}  verdict")
        for want, cases in (("PASS", benign), ("FAIL", changes)):
            for name, y in cases.items():
                v = voice_check(x, y, SR)
                ok = "ok" if v.verdict == want else "WRONG"
                print(
                    f"  {name:28s} {want:5s} {v.speaker_similarity:8.3f} {v.f0_shift_st:+7.2f} {v.f0_range_change_st:+7.2f} "
                    f"{v.tilt_change_db_oct:+7.2f}  {v.verdict} {ok}  {'; '.join(v.reasons)}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
