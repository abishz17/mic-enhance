"""Model registry. Imports are lazy so `mlstage --list` never loads torch."""

from __future__ import annotations

import importlib

# name -> (module, class)
REGISTRY: dict[str, tuple[str, str]] = {
    "deepfilternet3": ("dfn", "DeepFilterNet3"),
    "dpdfnet8-48k": ("dpdfnet", "DPDFNet8HR"),
    "dpdfnet2-48k": ("dpdfnet", "DPDFNet2HR"),
    "rnnoise": ("rnnoise", "RNNoise"),
    "gtcrn": ("gtcrn", "GTCRN"),
    "lavasr": ("lavasr", "LavaSR"),
    "lavasr-dn": ("lavasr", "LavaSRDenoise"),
    "sidon": ("sidon", "Sidon"),
    # models whose dependencies conflict with this env run in their own uv env (ml/envs/<env>)
    "mossformer2-48k": ("external", "MossFormer2"),
    "resemble-denoise": ("external", "ResembleDenoise"),
    "resemble-enhance": ("external", "ResembleEnhance"),
    "voicefixer": ("external", "VoiceFixer"),
}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'; available: {', '.join(sorted(REGISTRY))}")
    mod, cls = REGISTRY[name]
    return getattr(importlib.import_module(f".{mod}", __name__), cls)
