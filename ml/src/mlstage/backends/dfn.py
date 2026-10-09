"""DeepFilterNet3 (Schroeter et al. 2023), 48 kHz full-band, mask + deep-filter (no generation)."""

from __future__ import annotations

import os

import numpy as np

from .base import Backend, ModelInfo


class DeepFilterNet3(Backend):
    info = ModelInfo(
        name="deepfilternet3",
        mode="clean",
        in_sr=48_000,
        out_sr=48_000,
        license="MIT OR Apache-2.0 (code + weights)",
        weights="github.com/Rikorose/DeepFilterNet release DeepFilterNet3.zip (torch pickle .ckpt)",
        devices=("cpu",),
        chunk_s=20.0,  # peak footprint: 37 s one-shot = 1.0 GB, so keep chunks short
        context_s=1.0,
    )

    def __init__(self, device: str):
        super().__init__(device)
        os.environ.setdefault("DF_LOG_LEVEL", "WARNING")
        import importlib

        import torch

        dfe = importlib.import_module("df.enhance")  # `df.enhance` attr is shadowed by the function
        init_df = dfe.init_df

        self._torch = torch
        self._dfe = dfe
        if device != "cpu":
            # df uses its own get_device() (cuda or cpu); route it to the requested device
            dev = torch.device(device)
            dfe.get_device = lambda: dev
            import df.utils

            df.utils.get_device = lambda: dev
        self.model, self.state, _ = init_df(log_level="WARNING", post_filter=False)
        self.model = self.model.to(device).eval()

    def process(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            # pad=True compensates the STFT/ISTFT algorithmic delay (n_fft - hop) -> sample aligned
            y = self._dfe.enhance(self.model, self.state, torch.from_numpy(x)[None], pad=True)
        return y[0].numpy().astype(np.float32)
