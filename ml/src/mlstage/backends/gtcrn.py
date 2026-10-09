"""GTCRN (Rong et al., ICASSP 2024): 48k-param causal mask model, 16 kHz only (no 48 kHz weights).

Runs the official streaming ONNX export frame by frame (onnxruntime, CPU). Output above 8 kHz
is necessarily empty, so this is only a candidate for already-narrowband input.
"""

from __future__ import annotations

import numpy as np

from ..weights import fetch
from .base import Backend, ModelInfo

_URL = ("https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/"
        "9202557caa577baf2ec2220bef82ba9f4b589dc1/stream/onnx_models/gtcrn_simple.onnx")
_SHA = "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87"
N_FFT, HOP = 512, 256


class GTCRN(Backend):
    info = ModelInfo(
        name="gtcrn",
        mode="clean",
        in_sr=16_000,
        out_sr=16_000,
        license="MIT (code + weights)",
        weights="github.com/Xiaobin-Rong/gtcrn stream/onnx_models/gtcrn_simple.onnx (ONNX, DNS3-trained)",
        devices=("cpu",),
        chunk_s=120.0,
        context_s=1.0,
    )

    def __init__(self, device: str):
        super().__init__(device)
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(fetch("gtcrn/gtcrn_simple.onnx", _URL, _SHA)), so,
                                         providers=["CPUExecutionProvider"])
        self.win = np.sqrt(np.hanning(N_FFT + 1)[:-1]).astype(np.float32)  # periodic sqrt-Hann

    def process(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        pad = N_FFT // 2
        xp = np.pad(x, (pad, pad + N_FFT), mode="reflect" if n > pad else "constant")
        n_frames = 1 + (len(xp) - N_FFT) // HOP
        conv = np.zeros([2, 1, 16, 16, 33], np.float32)
        tra = np.zeros([2, 3, 1, 1, 16], np.float32)
        inter = np.zeros([2, 1, 33, 16], np.float32)
        y = np.zeros(len(xp), np.float64)
        wsum = np.zeros(len(xp), np.float64)
        for t in range(n_frames):
            fr = xp[t * HOP: t * HOP + N_FFT] * self.win
            X = np.fft.rfft(fr)
            mix = np.stack([X.real, X.imag], -1).astype(np.float32)[None, :, None, :]
            enh, conv, tra, inter = self.sess.run(
                None, {"mix": mix, "conv_cache": conv, "tra_cache": tra, "inter_cache": inter})
            Y = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
            y[t * HOP: t * HOP + N_FFT] += np.fft.irfft(Y, N_FFT) * self.win
            wsum[t * HOP: t * HOP + N_FFT] += self.win ** 2
        y = y / np.maximum(wsum, 1e-8)
        return y[pad: pad + n].astype(np.float32)
