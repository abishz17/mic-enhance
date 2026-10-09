"""Small DSP utilities: speech bandwidth estimate and a linear-phase band merge."""

from __future__ import annotations

import numpy as np


def effective_bandwidth(x: np.ndarray, sr: int, margin_db: float = 10.0, range_db: float = 60.0) -> float:
    """Highest frequency (Hz) where *speech* carries energy.

    A band counts as "carrying speech" when the mean power of loud (speech-active) frames is
    more than `margin_db` above the quiet (noise-floor) frames AND within `range_db` of the
    loudest band. Stationary noise above a codec cutoff therefore does not count as bandwidth,
    which is what we want for deciding whether bandwidth extension is needed.
    """
    n_fft, hop = 2048, 512
    if len(x) < n_fft * 4:
        return sr / 2
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]  # view, no copy
    n_frames = len(frames)
    blk = 1024

    def spectra(i):
        return np.abs(np.fft.rfft(frames[i:i + blk] * win, axis=1)) ** 2

    # pass 1: frame energies (Parseval: time-domain energy of the windowed frame)
    e = np.concatenate([np.sum((frames[i:i + blk] * win) ** 2, axis=1) for i in range(0, n_frames, blk)])
    order = np.argsort(e)
    quiet = np.zeros(n_frames, bool)
    quiet[order[: max(1, n_frames // 10)]] = True
    active = np.zeros(n_frames, bool)
    active[order[int(n_frames * 0.7):]] = True
    # pass 2: mean power spectrum of the quiet (bottom 10 %) and active (top 30 %) frames
    acc_q = np.zeros(n_fft // 2 + 1)
    acc_a = np.zeros(n_fft // 2 + 1)
    for i in range(0, n_frames, blk):
        P = spectra(i)
        acc_q += P[quiet[i:i + blk]].sum(axis=0)
        acc_a += P[active[i:i + blk]].sum(axis=0)

    def band_db(acc, count):
        s = np.convolve(acc / max(1, count), np.ones(4) / 4, mode="same")  # 94 Hz smoothing
        return 10 * np.log10(s + 1e-20)

    act, quiet = band_db(acc_a, active.sum()), band_db(acc_q, quiet.sum())
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    ok = (act > quiet + margin_db) & (act > act.max() - range_db) & (freqs > 100)
    if not ok.any():
        return 0.0
    return float(freqs[np.nonzero(ok)[0][-1]])


def band_merge(low_src: np.ndarray, high_src: np.ndarray, sr: int, cutoff: float, width: float = 1000.0) -> np.ndarray:
    """Zero-phase crossover: `low_src` below `cutoff`, `high_src` above, raised-cosine transition
    of `width` Hz centred on cutoff. Both inputs must be time-aligned and equal length."""
    n = len(low_src)
    L = np.fft.rfft(low_src.astype(np.float64))
    H = np.fft.rfft(high_src.astype(np.float64)[:n], n=n)
    f = np.fft.rfftfreq(n, 1 / sr)
    t = np.clip((f - (cutoff - width / 2)) / width, 0.0, 1.0)
    w_hi = 0.5 - 0.5 * np.cos(np.pi * t)  # 0 below band, 1 above
    return np.fft.irfft(L * (1 - w_hi) + H * w_hi, n=n).astype(np.float32)


def transport_bandwidth(x: np.ndarray, sr: int, range_db: float = 70.0) -> float:
    """Highest frequency (Hz) where the signal has *any* energy (speech or noise) within
    `range_db` of its strongest band: finds the hard cliff left by a codec / resampler
    (AirPods transport ~12 kHz, HFP mSBC 8 kHz). This, not the speech bandwidth, is what
    decides where bandwidth extension may start: below the cliff there is real content
    (even if noisy) that must not be replaced by generated audio."""
    n_fft, hop = 2048, 512
    if len(x) < n_fft * 4:
        return sr / 2
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    acc = np.zeros(n_fft // 2 + 1)
    for i in range(0, len(frames), 1024):
        acc += (np.abs(np.fft.rfft(frames[i:i + 1024] * win, axis=1)) ** 2).sum(axis=0)
    s = 10 * np.log10(np.convolve(acc / len(frames), np.ones(8) / 8, mode="same") + 1e-30)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    ok = (s > s[freqs > 100].max() - range_db) & (freqs > 100)
    return float(freqs[np.nonzero(ok)[0][-1]]) if ok.any() else 0.0


def _band_stats(x: np.ndarray, n_fft: int, hop: int, sel_from: np.ndarray | None = None):
    """Mean power spectrum over the top-30 % (active) and bottom-10 % (quiet) energy frames.
    Frame selection can be taken from another signal (`sel_from`, same length) so input and
    output are compared on exactly the same frames. Block-wise: memory independent of length."""
    win = np.hanning(n_fft).astype(np.float32)
    src = x if sel_from is None else sel_from
    fx = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
    fs = np.lib.stride_tricks.sliding_window_view(src, n_fft)[::hop]
    n = min(len(fx), len(fs))
    e = np.concatenate([np.sum((fs[i:i + 1024] * win) ** 2, axis=1) for i in range(0, n, 1024)])
    order = np.argsort(e)
    act = np.zeros(n, bool)
    act[order[int(n * 0.7):]] = True
    quiet = np.zeros(n, bool)
    quiet[order[: max(1, n // 10)]] = True
    acc_a = np.zeros(n_fft // 2 + 1)
    acc_q = np.zeros(n_fft // 2 + 1)
    for i in range(0, n, 1024):
        P = np.abs(np.fft.rfft(fx[i:min(n, i + 1024)] * win, axis=1)) ** 2
        acc_a += P[act[i:i + 1024]].sum(axis=0)
        acc_q += P[quiet[i:i + 1024]].sum(axis=0)
    return acc_a / max(1, act.sum()), acc_q / max(1, quiet.sum())


def timbre_match(x_in: np.ndarray, y_out: np.ndarray, sr: int = 48_000, f_lo: float = 2000.0,
                 max_boost_db: float = 12.0, margin_lo: float = 12.0, margin_hi: float = 20.0,
                 taps: int = 2047) -> tuple[np.ndarray, dict]:
    """Restore speech-band level that a denoiser removed from *speech-dominated* bands.

    Per 1/3-octave band above `f_lo`: loss = input speech-frame level - output level on the same
    frames. Where the input's speech-to-noise margin in that band is >= `margin_hi` the loss is
    the voice's own energy (not noise) and is restored fully; below `margin_lo` it is noise removal
    and left alone; linear ramp in between; boost capped at `max_boost_db`, never cut. Applied as
    one static linear-phase FIR (delay-compensated, sample aligned): a fixed EQ, so it cannot
    invent audio and does not pump. Returns (y, {band_hz: gain_db}).
    """
    n_fft, hop = 4096, 1024
    a_in, q_in = _band_stats(x_in, n_fft, hop)
    a_out, _ = _band_stats(y_out, n_fft, hop, sel_from=x_in)
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    centers = f_lo * 2 ** (np.arange(0, np.log2((sr / 2 * 0.95) / f_lo), 1 / 3))
    gains = {}
    for c in centers:
        b = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        loss = 10 * np.log10((a_in[b].sum() + 1e-20) / (a_out[b].sum() + 1e-20))
        margin = 10 * np.log10((a_in[b].sum() + 1e-20) / (q_in[b].sum() + 1e-20))
        w = np.clip((margin - margin_lo) / (margin_hi - margin_lo), 0.0, 1.0)
        gains[float(c)] = float(np.clip(loss, 0.0, max_boost_db) * w)
    # gain curve on a fine grid: 0 dB below f_lo, interpolate band gains in log-frequency
    grid = np.linspace(0, sr / 2, taps // 2 + 2)
    lc, lg = np.log2(list(gains)), np.array(list(gains.values()))
    g_db = np.where(grid < f_lo * 2 ** (-1 / 6), 0.0, np.interp(np.log2(np.maximum(grid, 1)), lc, lg))
    import scipy.signal as ss

    h = ss.firwin2(taps, grid, 10 ** (g_db / 20), fs=sr)
    y = ss.oaconvolve(y_out, h)[taps // 2: taps // 2 + len(y_out)].astype(np.float32)
    return y, gains


# Residual noise floor target, this far below active speech *at mlstage's output*. 68, not the
# ~55 dB we want in the final file, because the Rust post chain's compressor narrows the
# speech-to-noise gap by ~15 dB. Known coupling: the long-term fix is to target the final
# output floor directly (iteration 2).
ATTEN_FLOOR_TARGET_DB = 68.0
ATTEN_LIM_RANGE = (20.0, 45.0)


def speech_snr_db(x: np.ndarray, sr: int, frame_s: float = 0.02) -> float:
    """Speech-to-noise ratio (dB): mean power of speech-active frames (top 30 % by energy) over
    non-speech frames (bottom 20 %). Frames that are exact digital zero (on-device gating, e.g.
    AirPods) are excluded: they are not the room's noise floor and would make SNR infinite."""
    n = int(frame_s * sr)
    k = len(x) // n
    if k < 10:
        return float("nan")
    fr = x[: k * n].reshape(k, n).astype(np.float64)
    e = np.mean(fr ** 2, axis=1)
    e = e[np.any(fr != 0, axis=1)]  # drop frames that are entirely exact zero
    if len(e) < 10:
        return float("nan")
    e = np.sort(e)
    noise = e[: max(1, int(len(e) * 0.2))].mean()
    speech = e[int(len(e) * 0.7):].mean()
    return float(10 * np.log10((speech + 1e-20) / (noise + 1e-20)))


def auto_atten_lim_db(snr_db: float) -> float:
    """clamp(68 - SNR, 20, 45): leave the residual floor ~68 dB below speech, never remove less
    than 20 dB or more than 45 dB. Unknown SNR (silent / too short input) -> 30 dB."""
    lo, hi = ATTEN_LIM_RANGE
    if not np.isfinite(snr_db):
        return 30.0
    return float(min(hi, max(lo, ATTEN_FLOOR_TARGET_DB - snr_db)))
