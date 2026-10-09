"""LavaSR v2 (Sharma, Interspeech 2026, arXiv 2603.07285): Vocos-based bandwidth extension
(+ optional UL-UNAS 16 kHz denoiser).

Difference from upstream inference: upstream always bandlimits the input to 16 kHz and keeps
only < 8 kHz of the *resampled* signal. Here the original 48 kHz signal is kept below the
input's measured speech bandwidth and only the band above it is taken from the model, so a
full-band laptop-mic recording passes through untouched and an 8 kHz AirPods recording keeps its
real low band sample-for-sample (only the band above the codec cliff is generated).
The cliff is found with dsp.transport_bandwidth (highest band with any energy, speech or noise).
"""

from __future__ import annotations

import logging

import numpy as np

from ..audio import resample
from ..dsp import band_merge, transport_bandwidth
from .base import Backend, ModelInfo

log = logging.getLogger("mlstage")
_REPO, _REV = "YatharthS/LavaSR", "b98dc8be472da45ab7b6346ad7997e1dfeb5911d"
# above this the input is treated as full-band: no generation at all
FULLBAND_HZ = 15_000.0


class LavaSR(Backend):
    info = ModelInfo(
        name="lavasr",
        mode="studio",
        in_sr=48_000,
        out_sr=48_000,
        license="Apache-2.0 (code + weights)",
        weights=f"hf:{_REPO}@{_REV[:8]} enhancer_v2/pytorch_model.bin (torch pickle, loaded weights_only)",
        devices=("cpu", "mps"),
        chunk_s=30.0,
        context_s=1.0,
    )
    denoise = False

    def __init__(self, device: str):
        super().__init__(device)
        import torch
        import yaml
        from huggingface_hub import snapshot_download

        from ..vendor.vocos.feature_extractors import MelSpectrogramFeatures
        from ..vendor.vocos.heads import ISTFTHead
        from ..vendor.vocos.models import VocosBackbone

        self._torch = torch
        root = snapshot_download(_REPO, revision=_REV, allow_patterns=["enhancer_v2/*", "denoiser/*"])
        cfg = yaml.safe_load(open(f"{root}/enhancer_v2/config.yaml"))
        self.fe = MelSpectrogramFeatures(**cfg["feature_extractor"]["init_args"])
        self.backbone = VocosBackbone(**cfg["backbone"]["init_args"])
        self.head = ISTFTHead(**cfg["head"]["init_args"])
        sd = torch.load(f"{root}/enhancer_v2/pytorch_model.bin", map_location="cpu", weights_only=True)
        for prefix, mod in (("feature_extractor.", self.fe), ("backbone.", self.backbone), ("head.", self.head)):
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            mod.load_state_dict(sub)
            mod.to(device).eval()
        if self.denoise:
            from ..vendor.lavasr.ulunas import ULUNAS

            self.dn = ULUNAS().eval()
            self.dn.load_state_dict(torch.load(f"{root}/denoiser/denoiser.bin", map_location="cpu", weights_only=True))
            self.dn.to(device)
        self.cutoff = 8_000.0

    def prepare(self, x_full: np.ndarray) -> None:
        bw = transport_bandwidth(x_full, 48_000)
        self.cutoff = min(bw, 8_000.0) if self.denoise else bw
        log.info("lavasr: input transport bandwidth %.0f Hz -> keep original below %.0f Hz%s", bw, self.cutoff,
                 " (full-band: passthrough)" if self.cutoff >= FULLBAND_HZ and not self.denoise else "")

    def _head(self, x):  # upstream's monkey-patched ISTFTHead.forward (mag clip instead of exp only)
        torch = self._torch
        h = self.head
        x = h.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e3)
        return h.istft(mag * (torch.cos(p) + 1j * torch.sin(p)))

    def process(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        if not self.denoise and self.cutoff >= FULLBAND_HZ:
            return x
        x16 = resample(x, 48_000, 16_000)
        if self.denoise:
            with torch.inference_mode():
                x16 = self.dn(torch.from_numpy(x16)[None].to(self.device))[0].cpu().numpy()
            base = resample(x16, 16_000, 48_000)[: len(x)]
        else:
            base = x
        bwe_in = resample(x16, 16_000, 48_000)[: len(x)]
        with torch.inference_mode():
            t = torch.from_numpy(np.ascontiguousarray(bwe_in))[None].to(self.device)
            pred = self._head(self.backbone(self.fe(t)))[0, : len(x)].float().cpu().numpy()
        if len(pred) < len(x):
            pred = np.pad(pred, (0, len(x) - len(pred)))
        # transition band sits just below the measured cutoff so no "hole" is left
        width = 600.0
        return band_merge(base, pred, 48_000, cutoff=self.cutoff - width / 2, width=width)


class LavaSRDenoise(LavaSR):
    info = ModelInfo(**{**LavaSR.info.__dict__, "name": "lavasr-dn",
                        "devices": ("cpu",),  # UL-UNAS on MPS diverges from CPU (18-23 dB SNR), see BAKEOFF.md
                        "notes": "UL-UNAS 16 kHz denoiser then BWE; discards original content above 8 kHz"})
    denoise = True
