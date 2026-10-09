"""RNNoise (xiph, Valin 2018; v0.2 weights) via the librnnoise bundled in the `pyrnnoise` wheel."""

from __future__ import annotations

import ctypes

import numpy as np

from .base import Backend, ModelInfo


class RNNoise(Backend):
    info = ModelInfo(
        name="rnnoise",
        mode="clean",
        in_sr=48_000,
        out_sr=48_000,
        license="BSD-3-Clause (xiph rnnoise); wrapper pyrnnoise Apache-2.0",
        weights="compiled into librnnoise (no checkpoint file)",
        devices=("cpu",),
        chunk_s=120.0,
        context_s=1.0,
    )
    # librnnoise processes 480-sample frames in place with an overlap-add STFT (960-pt window):
    # output lags input by 2 frames = 960 samples = 20 ms (measured with bakeoff/align_check.py).
    DELAY = 960

    def __init__(self, device: str):
        super().__init__(device)
        from pyrnnoise.rnnoise import FRAME_SIZE, lib

        self._lib = lib
        self._fs = FRAME_SIZE

    def process(self, x: np.ndarray) -> np.ndarray:
        fs, lib, d = self._fs, self._lib, self.DELAY
        n = len(x)
        # rnnoise expects int16-scaled floats; pad tail so delayed output covers the input
        buf = np.zeros(((n + d + fs - 1) // fs) * fs, dtype=np.float32)
        buf[:n] = x * 32768.0
        st = lib.rnnoise_create(None)
        try:
            for i in range(0, len(buf), fs):
                fr = buf[i:i + fs]  # contiguous view, processed in place
                p = fr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                lib.rnnoise_process_frame(st, p, p)
        finally:
            lib.rnnoise_destroy(st)
        return (buf[d:d + n] / 32768.0).astype(np.float32)
