from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("mlstage")


@dataclass
class ModelInfo:
    name: str
    mode: str  # "clean" (mask/filter only) or "studio" (generative)
    in_sr: int  # rate the backend's process() expects
    out_sr: int  # rate process() returns
    license: str
    weights: str  # where weights come from + format
    devices: tuple[str, ...] = ("cpu",)  # devices verified correct, preferred first
    chunk_s: float = 30.0
    context_s: float = 1.0
    notes: str = ""
    extra: dict = field(default_factory=dict)


class Backend:
    """A model behind the mlstage contract.

    process() receives float32 mono audio at info.in_sr and must return float32 mono audio at
    info.out_sr, *time-aligned to the input* (the backend compensates its own latency) and of
    length ~len(x) * out_sr / in_sr (the caller pads/trims the last few samples).
    """

    info: ModelInfo

    def __init__(self, device: str):
        self.device = device

    def prepare(self, x_full: np.ndarray) -> None:
        """Called once with the whole (resampled) input before chunking, e.g. for global gain."""

    def process(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def finish(self, y_full: np.ndarray) -> np.ndarray:
        """Called once with the whole output (at out_sr) after chunking."""
        return y_full

    def close(self) -> None:
        pass
