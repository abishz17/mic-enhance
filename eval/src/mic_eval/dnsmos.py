"""DNSMOS P.835 (SIG/BAK/OVRL) + P.808 MOS, using Microsoft's official ONNX models.

Port of DNS-Challenge/DNSMOS/dnsmos_local.py (non-personalized), same
segmenting (9.01 s windows, 1 s hop), same mel features for P.808, same
polynomial calibration. Models are downloaded from the official GitHub repo
into eval/models/dnsmos/ and verified by SHA-256.

IMPORTANT: DNSMOS works at 16 kHz. Anything above 8 kHz is discarded before
scoring, so DNSMOS cannot see the AirPods bandwidth problem at all.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np

from .audio import resample

SR = 16_000
INPUT_LENGTH = 9.01
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "dnsmos"
BASE_URL = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/"
MODELS = {
    "sig_bak_ovr.onnx": "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    "model_v8.onnx": "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91",
}
# Non-personalized calibration polynomials from dnsmos_local.py
P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])


@dataclass
class DnsmosScore:
    sig: float
    bak: float
    ovrl: float
    p808: float
    n_windows: int
    sig_std: float  # spread across 9 s windows: a per-file noise indicator
    ovrl_std: float


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_models(model_dir: Path = MODEL_DIR) -> dict[str, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, sha in MODELS.items():
        p = model_dir / name
        if not p.exists():
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(BASE_URL + name, tmp)  # noqa: S310 (fixed https URL)
            tmp.rename(p)
        got = _sha256(p)
        if got != sha:
            raise RuntimeError(f"DNSMOS model {p} has sha256 {got}, expected {sha}; delete it and retry")
        out[name] = p
    return out


@lru_cache(maxsize=1)
def _sessions():  # type: ignore[no-untyped-def]
    import onnxruntime as ort

    paths = ensure_models()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    prov = ["CPUExecutionProvider"]
    return (
        ort.InferenceSession(str(paths["sig_bak_ovr.onnx"]), so, providers=prov),
        ort.InferenceSession(str(paths["model_v8.onnx"]), so, providers=prov),
    )


def _melspec(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=321, hop_length=160, n_mels=120)
    return ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T


def dnsmos(x: np.ndarray, sr: int) -> DnsmosScore:
    sess, p808 = _sessions()
    audio = resample(x, sr, SR).astype(np.float64)
    seg_len = int(INPUT_LENGTH * SR)
    while len(audio) < seg_len:  # same as reference: tile short clips
        audio = np.concatenate([audio, audio])
    num_hops = int(np.floor(len(audio) / SR) - INPUT_LENGTH) + 1
    sig, bak, ovr, p8 = [], [], [], []
    for i in range(num_hops):
        seg = audio[i * SR : i * SR + seg_len]
        if len(seg) < seg_len:
            continue
        feat = seg.astype(np.float32)[None, :]
        mel = _melspec(seg[:-160]).astype(np.float32)[None, :, :]
        p8.append(float(p808.run(None, {"input_1": mel})[0][0][0]))
        s, b, o = sess.run(None, {"input_1": feat})[0][0]
        sig.append(float(P_SIG(s)))
        bak.append(float(P_BAK(b)))
        ovr.append(float(P_OVR(o)))
    return DnsmosScore(
        sig=float(np.mean(sig)),
        bak=float(np.mean(bak)),
        ovrl=float(np.mean(ovr)),
        p808=float(np.mean(p8)),
        n_windows=len(sig),
        sig_std=float(np.std(sig)),
        ovrl_std=float(np.std(ovr)),
    )
