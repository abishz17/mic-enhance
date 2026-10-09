from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "testdata"
OUT = ROOT / "ml" / "out"

# set name -> (input dir, reference kind, leading-silence seconds to check for babble)
SETS = {
    "corpus": (ROOT / "corpus", "none", 4.0),
    "smoke": (TD / "smoke", "smoke", 2.0),
    "vbdemand": (TD / "vbdemand", "vb", 0.0),
    "vbreverb": (TD / "vbreverb", "vb", 0.0),
    "vbbt_msbc": (TD / "vbbt_msbc", "vb", 0.0),
    "vbbt_eld": (TD / "vbbt_eld", "vb", 0.0),
}

CLEAN_MODELS = ["deepfilternet3", "dpdfnet8-48k", "dpdfnet2-48k", "rnnoise", "gtcrn", "mossformer2-48k", "resemble-denoise"]
STUDIO_MODELS = ["lavasr", "lavasr-dn", "sidon", "resemble-enhance", "voicefixer"]
ALL_MODELS = CLEAN_MODELS + STUDIO_MODELS


def clips(set_name: str) -> list[Path]:
    d = SETS[set_name][0]
    return sorted(p for p in d.glob("*.wav"))


def ref_for(set_name: str, clip: Path) -> Path | None:
    kind = SETS[set_name][1]
    if kind == "smoke":
        return TD / "smoke" / "clean.wav"
    if kind == "vb":
        return TD / "vbref" / clip.name
    return None


def ref_text(set_name: str, clip: Path) -> str | None:
    if SETS[set_name][1] == "vb":
        return (TD / "vbref" / (clip.stem + ".txt")).read_text().strip()
    return None


def out_path(model: str, set_name: str, clip: Path) -> Path:
    return OUT / model / set_name / clip.name
