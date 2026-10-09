"""DPDFNet (Ceva, arXiv 2512.16420, Dec 2025): DeepFilterNet2 + dual-path RNN encoder blocks.
48 kHz "HR" variants, causal mask/deep-filter model (no generation), official ONNX on CPU.
"""

from __future__ import annotations

import numpy as np

from .base import Backend, ModelInfo

_REPO, _REV = "Ceva-IP/DPDFNet", "dd6818d00f50c836fed43a6243ebe49116de5964"


class _DPDFNet(Backend):
    onnx_name: str

    def __init__(self, device: str):
        super().__init__(device)
        from dpdfnet.onnx_backend import build_runtime_model
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(_REPO, f"onnx/{self.onnx_name}", revision=_REV)
        self.rt = build_runtime_model(path)

    def process(self, x: np.ndarray) -> np.ndarray:
        from dpdfnet.api import _enhance_with_runtime  # dpdfnet==0.6.0, pinned in pyproject

        return _enhance_with_runtime(x, 48_000, runtime=self.rt, model_sample_rate=48_000)


def _info(name, onnx):
    return ModelInfo(name=name, mode="clean", in_sr=48_000, out_sr=48_000,
                     license="Apache-2.0 (code + weights)",
                     weights=f"hf:{_REPO}@{_REV[:8]} onnx/{onnx} (ONNX)", devices=("cpu",),
                     chunk_s=60.0, context_s=1.0)


class DPDFNet8HR(_DPDFNet):
    onnx_name = "dpdfnet8_48khz_hr.onnx"
    info = _info("dpdfnet8-48k", onnx_name)


class DPDFNet2HR(_DPDFNet):
    onnx_name = "dpdfnet2_48khz_hr.onnx"
    info = _info("dpdfnet2-48k", onnx_name)
