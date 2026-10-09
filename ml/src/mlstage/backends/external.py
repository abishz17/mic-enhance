"""Backends whose Python dependencies conflict with the main env (pinned old torch/librosa,
deepspeed, ...). Each runs in its own uv project under ml/envs/<env>/ as a long-lived worker
process, so the model loads once per mlstage call even when the file is chunked.

Worker protocol (newline-delimited JSON):
  launch : uv run --project ml/envs/<env> python -I ml/envs/<env>/runner.py --model <m> --device <d>
  worker -> {"ready": true}                       once the model is loaded
  us     -> {"in": "<wav>", "out": "<wav>"}       one job per chunk (48 kHz mono f32 input)
  worker -> {"ok": true} | {"ok": false, "error": "..."}
Worker output WAVs may be at any rate / length; this side resamples to 48 kHz, removes the
model's measured latency (DELAY_S) and trims to length. Worker stdout is protocol only; its
logs go to stderr (inherited, so they show up in mlstage's stderr).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio import fix_length, resample
from .base import Backend, ModelInfo

log = logging.getLogger("mlstage")
ENVS = Path(__file__).resolve().parents[3] / "envs"


class External(Backend):
    env: str
    worker_model: str
    DELAY_S: float = 0.0  # measured model latency, removed here

    def __init__(self, device: str):
        super().__init__(device)
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("`uv` not found on PATH (needed to run the isolated model env)")
        env = ENVS / self.env
        if not (env / "pyproject.toml").is_file():
            raise RuntimeError(f"model env missing: {env}")
        self.tmp = Path(tempfile.mkdtemp(prefix=f"mlstage-{self.worker_model}-"))
        self.proc = subprocess.Popen(
            [uv, "run", "--quiet", "--project", str(env), "python", "-I", str(env / "runner.py"),
             "--model", self.worker_model, "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)
        msg = self._recv()
        if not msg.get("ready"):
            raise RuntimeError(f"worker failed to start: {msg}")
        self.k = 0

    def _recv(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.worker_model} worker exited (code {self.proc.poll()})")
        return json.loads(line)

    def process(self, x: np.ndarray) -> np.ndarray:
        self.k += 1
        src, dst = self.tmp / f"in{self.k}.wav", self.tmp / f"out{self.k}.wav"
        sf.write(src, x, 48_000, subtype="FLOAT")
        self.proc.stdin.write(json.dumps({"in": str(src), "out": str(dst)}) + "\n")
        self.proc.stdin.flush()
        msg = self._recv()
        if not msg.get("ok"):
            raise RuntimeError(f"{self.worker_model}: {msg.get('error')}")
        y, sr = sf.read(dst, dtype="float32", always_2d=True)
        y = resample(y.mean(axis=1), sr, 48_000)
        d = int(round(self.DELAY_S * 48_000))
        y = y[d:] if d >= 0 else np.pad(y, (-d, 0))
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
        return fix_length(y, len(x))

    def close(self) -> None:
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        shutil.rmtree(getattr(self, "tmp", ""), ignore_errors=True)


def _info(name, mode, lic, weights, devices=("cpu",), chunk_s=30.0, notes=""):
    return ModelInfo(name=name, mode=mode, in_sr=48_000, out_sr=48_000, license=lic, weights=weights,
                     devices=devices, chunk_s=chunk_s, context_s=1.0, notes=notes)


class MossFormer2(External):
    env, worker_model = "clearvoice", "MossFormer2_SE_48K"
    info = _info("mossformer2-48k", "clean", "Apache-2.0 (ClearerVoice-Studio code + weights)",
                 "hf:alibabasglab/MossFormer2_SE_48K last_best_checkpoint.pt (torch pickle, weights_only load)",
                 # MPS verified vs CPU: SNR 72 dB (15.7 s smoke), 51 dB (31 s real recording); ~4x faster
                 devices=("mps", "cpu"))


class ResembleDenoise(External):
    env, worker_model = "resemble", "denoise"
    info = _info("resemble-denoise", "clean", "MIT (code + weights)",
                 "hf:ResembleAI/resemble-enhance enhancer_stage2 (DeepSpeed torch pickle)",
                 # MPS verified vs CPU: SNR 137 dB (smoke), 125 dB (31 s real recording); ~4x faster
                 devices=("mps", "cpu"),
                 notes="44.1 kHz; mask-based denoiser module of Resemble Enhance")


class ResembleEnhance(External):
    env, worker_model = "resemble", "enhance"
    info = _info("resemble-enhance", "studio", "MIT (code + weights)",
                 "hf:ResembleAI/resemble-enhance enhancer_stage2 (DeepSpeed torch pickle)",
                 # MPS is NOT correct for the CFM enhancer (torch 2.8): output ~24 dB low and
                 # spectrally unrelated to CPU (log-mel diff 32 dB vs 11 dB between CPU seeds)
                 devices=("cpu",),
                 notes="44.1 kHz; latent CFM enhancer (generative)")


class VoiceFixer(External):
    env, worker_model = "voicefixer", "voicefixer"
    info = _info("voicefixer", "studio", "MIT (code + weights)",
                 "zenodo (official links in haoheliu/voicefixer): vf.ckpt + 44.1k HiFi-GAN (torch pickle)",
                 notes="44.1 kHz; analysis-synthesis via mel + neural vocoder")
