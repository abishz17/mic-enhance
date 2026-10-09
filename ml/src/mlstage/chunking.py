"""Bounded-memory processing of long files: overlapping chunks + complementary crossfade.

Each chunk is the "core" region [s, s+L) padded with `context` seconds of real audio on both
sides (so recurrent / attention models see what comes before and after). The model output for
the padded segment is cropped back to the core plus half a crossfade on each side, and
neighbouring chunks are blended with sin^2/cos^2 fades that sum to exactly 1. The chunk
outputs are estimates of the same signal, so a linear-sum (coherent) crossfade is correct.

    input   |---ctx---|========= core L =========|---ctx---|
    kept               <xf/2>|===== core =====|<xf/2>
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from .audio import fix_length

log = logging.getLogger("mlstage")


def _fade(n: int) -> tuple[np.ndarray, np.ndarray]:
    t = (np.arange(n, dtype=np.float64) + 0.5) / n
    fade_in = np.sin(0.5 * np.pi * t) ** 2
    return fade_in.astype(np.float32), (1.0 - fade_in).astype(np.float32)


def process_chunked(
    x: np.ndarray,
    sr_in: int,
    sr_out: int,
    fn: Callable[[np.ndarray], np.ndarray],
    chunk_s: float,
    context_s: float,
    xfade_s: float = 0.5,
) -> np.ndarray:
    """Run `fn` (sr_in samples -> sr_out samples, time-aligned) over `x` in chunks.

    Returns exactly round(len(x) * sr_out / sr_in) samples.
    """
    n = len(x)
    ratio = sr_out / sr_in
    n_out = int(round(n * ratio))
    L = int(round(chunk_s * sr_in))
    C = int(round(context_s * sr_in))
    xf = int(round(xfade_s * sr_out))
    if xfade_s > 2 * context_s:
        raise ValueError("crossfade must fit inside the context on both sides")

    if n <= L + C:  # short file: one shot, no seams
        return fix_length(fn(x), n_out)

    # chunk boundaries; a short tail (< 1 s) is folded into the previous chunk so every
    # core is long enough to hold a full crossfade
    bounds = list(range(0, n, L)) + [n]
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < sr_in:
        del bounds[-2]
    n_chunks = len(bounds) - 1
    log.info("chunked: %d chunks of %.1fs (+%.1fs context, %.2fs crossfade)", n_chunks, chunk_s, context_s, xfade_s)
    y = np.zeros(n_out, dtype=np.float32)
    fin, fout = _fade(xf)
    for i in range(n_chunks):
        s, e = bounds[i], bounds[i + 1]
        a = max(0, s - C)
        b = min(n, e + C)
        seg_out = fix_length(fn(x[a:b]), int(round((b - a) * ratio)))
        # core region in output samples, widened by half a crossfade on interior seams
        cs = int(round(s * ratio)) - (xf // 2 if i > 0 else 0)
        ce = (int(round(e * ratio)) + (xf - xf // 2) if i < n_chunks - 1 else n_out)
        ce = min(ce, n_out)
        off = int(round(a * ratio))
        piece = seg_out[cs - off: ce - off].copy()
        if len(piece) != ce - cs:
            piece = fix_length(piece, ce - cs)
        if i > 0:
            piece[:xf] *= fin
        if i < n_chunks - 1:
            piece[-xf:] *= fout
        y[cs:ce] += piece
    return y
