"""Sidon v0.1 (Nakata & Saito, 2025, arXiv 2509.17052): w2v-BERT 2.0 feature cleanser + DAC-style
vocoder. Fully generative: the 48 kHz output is re-synthesised from 16 kHz SSL features, so
nothing of the input waveform survives (words, timbre and timing must be checked).

Inference follows the official demo (huggingface.co/spaces/sarulab-speech/sidon_demo_beta):
peak-normalise to 0.9, 50 Hz high-pass, 16 kHz, SeamlessM4T log-mel features, decoder at 48 kHz.
The peak normalisation is done once per file here (not per chunk) and undone afterwards.
"""

from __future__ import annotations

import logging

import numpy as np

from ..audio import resample
from .base import Backend, ModelInfo

log = logging.getLogger("mlstage")
_REPO, _REV = "sarulab-speech/sidon-v0.1", "b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52"
_FE_SHA = "fd9abc906a9048b3c047bd3d246a25f6485d09c2d6bb85e098f527f844efc019"
_DEC_SHA = "34dcc80fab75bd1336369ba5b2063f6da475887aa824db94b495c4059c5fdea4"
_FE_REPO, _FE_REV = "facebook/w2v-bert-2.0", "da985ba0987f70aaeb84a80f2851cfac8c697a7b"


class Sidon(Backend):
    info = ModelInfo(
        name="sidon",
        mode="studio",
        in_sr=16_000,
        out_sr=48_000,
        license="MIT (weights, HF card); w2v-BERT 2.0 base MIT",
        weights=f"hf:{_REPO}@{_REV[:8]} feature_extractor_cpu.pt + decoder_cpu.pt (TorchScript archives)",
        # the official "*_cpu.pt" TorchScript graphs create CPU tensors internally -> fail on MPS
        # ("Passed CPU tensor to MPS op"); the "*_cuda.pt" graphs hard-code CUDA.
        devices=("cpu",),
        chunk_s=20.0,
        context_s=1.0,
    )

    # measured: output lags input by 7.8 ms (median over 9 clips in the 1-4 kHz band; the 10 ms
    # front pad minus ~2.2 ms of decoder look-back). Generated phase means +-2 ms clip-to-clip jitter.
    FRONT_TRIM = 374  # 48 kHz samples

    def __init__(self, device: str):
        super().__init__(device)
        import torch
        import transformers
        from ..weights import fetch

        self._torch = torch
        base = f"https://huggingface.co/{_REPO}/resolve/{_REV}"
        fe = fetch("sidon/feature_extractor_cpu.pt", f"{base}/feature_extractor_cpu.pt", _FE_SHA)
        dec = fetch("sidon/decoder_cpu.pt", f"{base}/decoder_cpu.pt", _DEC_SHA)
        self.fe = torch.jit.load(fe, map_location="cpu").to(device).eval()
        self.dec = torch.jit.load(dec, map_location="cpu").to(device).eval()
        self.pre = transformers.SeamlessM4TFeatureExtractor.from_pretrained(_FE_REPO, revision=_FE_REV)
        self.gain = 1.0

    def prepare(self, x_full: np.ndarray) -> None:
        peak = float(np.max(np.abs(x_full))) or 1.0
        self.gain = 0.9 / peak

    def process(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        import torchaudio.functional as AF

        n_out = int(round(len(x) * 3))
        w = torch.from_numpy(x * self.gain)[None]
        w = AF.highpass_biquad(w, 16_000, 50.0)
        w = torch.nn.functional.pad(w.view(-1), (0, 8000))  # tail pad as in the demo (decoder lookahead)
        inputs = self.pre(torch.nn.functional.pad(w, (160, 160)).numpy(), sampling_rate=16_000, return_tensors="pt")
        with torch.inference_mode():
            feat = self.fe(inputs["input_features"].to(self.device))["last_hidden_state"]
            y = self.dec(feat.transpose(1, 2)).reshape(-1).float().cpu().numpy()
        # the 160-sample (10 ms @16k) front pad is still in the decoder output: drop the measured latency
        y = y[self.FRONT_TRIM:self.FRONT_TRIM + n_out]
        if len(y) < n_out:
            y = np.pad(y, (0, n_out - len(y)))
        return (y / self.gain).astype(np.float32)
