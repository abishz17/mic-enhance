"""Deterministic signal metrics: loudness, true peak, clipping, noise floor, bandwidth.

These are pure DSP (no ML) and are covered by known-answer tests in tests/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pyloudnorm
from scipy.signal import resample_poly

from .audio import db

# ---------------------------------------------------------------- loudness


def integrated_lufs(x: np.ndarray, sr: int) -> float | None:
    """ITU-R BS.1770-4 integrated loudness (K-weighting, 400 ms blocks, 75 %
    overlap, -70 LUFS absolute + -10 LU relative gate), via pyloudnorm.

    Mono is measured as ONE channel (channel weight 1.0), i.e. a 0 dBFS-peak
    1 kHz sine reads -3.01 LUFS. Returns None if shorter than one 400 ms block,
    -inf if every block is gated out (silence).
    """
    if len(x) < int(0.4 * sr):
        return None
    meter = pyloudnorm.Meter(sr)  # BS.1770-4 K-weighting
    return float(meter.integrated_loudness(x))


def true_peak_dbtp(x: np.ndarray, oversample: int = 4) -> float:
    """BS.1770-4 Annex 2 true peak: oversample (default 4x) and take max |x|.

    Uses a polyphase Kaiser-windowed FIR (scipy resample_poly) which is longer
    / flatter than the 48-tap example filter in the annex, so it reads at
    least as high. Residual under-read for 4x is at most ~0.7 dB for content
    right at Nyquist and negligible for speech.
    """
    if len(x) == 0:
        return -math.inf
    up = resample_poly(x, oversample, 1)
    return db(float(np.max(np.abs(up))))


def sample_peak_dbfs(x: np.ndarray) -> float:
    return db(float(np.max(np.abs(x)))) if len(x) else -math.inf


# ---------------------------------------------------------------- clipping


@dataclass
class ClipStats:
    clipped_samples: int  # samples inside flat-topped runs (digital clipping)
    clipped_pct: float
    clip_runs: int
    over_fs_samples: int  # |x| > 1.0 (float files: will clip on integer export)


def clipping(x: np.ndarray, min_run: int = 3, peak_window_db: float = 0.1, min_level: float = 0.01) -> ClipStats:
    """Detect digital clipping as runs of >= `min_run` consecutive samples with
    *exactly* equal magnitude, within `peak_window_db` of the file's own peak
    (and the peak >= `min_level`, i.e. -40 dBFS).

    Why "near the file's peak": a hard clipper flattens the loudest crests, so
    the plateau IS the maximum, wherever the ceiling was (0 dBFS, or a lower
    ADC/preamp ceiling followed by linear gain). Requiring this rejects 16-bit
    content, where low-frequency crests well below peak often repeat a value
    (seen on testdata/smoke/clean.wav, which is 16-bit-quantized TTS).

    Why exact equality: a band-limited signal never sits flat for 3 samples,
    whereas a hard clipper (at 0 dBFS or at any lower ceiling, e.g. a clipped
    preamp/ADC before gain) produces identical values. Tolerance-based
    detectors false-trigger on low-frequency sines (50 Hz hum: adjacent samples
    near the crest differ by only ~1e-5).
    Blind spot: soft/analog clipping (rounded tops) is not detected.
    """
    n = len(x)
    over = int(np.sum(np.abs(x) > 1.0))
    if n < min_run:
        return ClipStats(0, 0.0, 0, over)
    a = np.abs(x)
    peak = float(a.max())
    if peak < min_level:
        return ClipStats(0, 0.0, 0, over)
    level = peak * 10 ** (-peak_window_db / 20)
    same = (a[1:] == a[:-1]) & (a[1:] >= level)
    # run-length encode `same` (pairs); a run of k equal pairs = k+1 samples
    padded = np.concatenate(([False], same, [False])).astype(np.int8)
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    lengths = ends - starts + 1  # samples per run
    keep = lengths >= min_run
    clipped = int(lengths[keep].sum())
    return ClipStats(clipped, 100.0 * clipped / n, int(keep.sum()), over)


# ---------------------------------------------------------------- noise / level


FRAME_S = 0.050
HOP_S = 0.025


def frame_rms_db(x: np.ndarray, sr: int, frame_s: float = FRAME_S, hop_s: float = HOP_S) -> np.ndarray:
    """RMS (dBFS, sine-peak convention NOT applied: full-scale DC = 0 dB) per frame."""
    f = max(1, int(round(frame_s * sr)))
    h = max(1, int(round(hop_s * sr)))
    if len(x) < f:
        frames = x[None, :]
    else:
        idx = np.arange(0, len(x) - f + 1, h)
        frames = np.lib.stride_tricks.sliding_window_view(x, f)[idx]
    ms = np.mean(frames**2, axis=1)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(ms)


SILENCE_DB = -200.0


def _pct(r: np.ndarray, q: float) -> float:
    """Percentile that treats -inf (digital zero) as a value, not NaN."""
    v = float(np.percentile(np.maximum(r, SILENCE_DB), q))
    return -math.inf if v <= SILENCE_DB else v


@dataclass
class NoiseStats:
    noise_floor_dbfs: float  # 10th percentile of 50 ms frame RMS
    speech_level_dbfs: float  # mean power of "active" frames (> floor + 15 dB)
    snr_db: float | None
    digital_silence_pct: float  # % of frames that are exactly zero
    lead_noise_dbfs: float | None  # median 20 ms RMS of an explicit leading room-tone window


def noise_stats(x: np.ndarray, sr: int, lead_silence_s: float | None = None) -> NoiseStats:
    r = frame_rms_db(x, sr)
    zero_pct = 100.0 * float(np.mean(np.isneginf(r)))
    floor = _pct(r, 10) if len(r) else -math.inf
    active_thr = (floor if math.isfinite(floor) else -100.0) + 15.0
    act = r[r > active_thr]
    if len(act):
        speech = float(10.0 * np.log10(np.mean(10.0 ** (act / 10.0))))
    else:
        speech = float(np.max(r)) if len(r) else -math.inf
    snr = speech - floor if math.isfinite(floor) and math.isfinite(speech) else None
    lead = None
    if lead_silence_s:
        # median of 20 ms frame RMS over the leading room-tone window: robust to
        # a start click or a breath; digital-zero frames count as -inf.
        seg = x[: int(lead_silence_s * sr)]
        if len(seg) >= int(0.02 * sr):
            lead = _pct(frame_rms_db(seg, sr, 0.02, 0.02), 50)
    return NoiseStats(floor, speech, snr, zero_pct, lead)


# ---------------------------------------------------------------- bandwidth


@dataclass
class BandwidthStats:
    bandwidth_hz: float  # min(SNR-based edge, brickwall cliff); NaN if undeterminable
    limited_by: str  # "snr" | "cliff" | "nyquist" | reason it is NaN
    snr_edge_hz: float  # highest band >= excess_db over the noise floor
    cliff_hz: float | None  # -10 dB point of a brickwall cutoff, if one exists
    gated_frames_pct: float  # frames excluded because they contain digital zeros


def effective_bandwidth(
    x: np.ndarray,
    sr: int,
    excess_db: float = 10.0,
    dynamic_range_db: float = 70.0,
    band_hz: float = 100.0,
    cliff_drop_db: float = 20.0,
    cliff_window_hz: float = 1000.0,
    cliff_min_hz: float = 3000.0,
) -> BandwidthStats:
    """Highest frequency where speech has meaningful energy above the noise.

    Spectra: Hann 2048-pt frames (~43 ms, 50 % hop) summed into `band_hz` bands.
    Frames with > 5 % exact-zero samples are dropped first (AirPods / VoIP
    noise gates write digital zeros; those would make any noise PSD -inf).

    Per band b, over the remaining frames:
      speech[b] = 90th percentile of band level over speech-active frames
                  (broadband level > p10 + 15 dB). p90 rather than mean so a
                  handful of broadband plosive/click frames cannot dominate;
                  p90 rather than median so fricatives (a minority of frames,
                  and the only HF speech energy) still count.
      noise[b]  = 10th percentile of band level over all kept frames
                  (minimum statistics; needs pauses in the speech).
    SNR edge: highest band with speech-noise >= `excess_db` and speech within
      `dynamic_range_db` of the strongest band, after a 5-band majority vote.
    Cliff (filter edge): the first band j >= 3 kHz whose speech level is
      >= `cliff_drop_db` (20 dB) below the strongest band in the preceding
      `cliff_window_hz` (1 kHz), i.e. a slope of >= 20 dB/kHz, and above which
      speech never returns to within 10 dB of that level. Measured p90 speech
      spectra above 3 kHz fall at most ~6 dB/kHz (laptop-mic recordings), codec
      edges fall 20-45 dB within 600 Hz (AirPods recording, BT smoke file), so only
      a filter triggers this; the "never returns" rule ignores notches.
      Energy above a cliff is aliasing/imaging (AirPods: 24 kHz codec upsampled
      to 48 kHz leaves a mirror image at 12-14 kHz ~20 dB down), not bandwidth.
      Reported value is where the filter has fallen 20 dB.
    Result = min(SNR edge, cliff). "nyquist" means no limit was found.
    """
    nfft = 2048
    hop = nfft // 2
    nan = math.nan
    if len(x) < nfft * 8:
        return BandwidthStats(nan, "too short", nan, None, 0.0)
    idx = np.arange(0, len(x) - nfft + 1, hop)
    raw = np.lib.stride_tricks.sliding_window_view(x, nfft)[idx]
    gated = np.mean(raw == 0.0, axis=1) > 0.05
    gated_pct = 100.0 * float(gated.mean())
    raw = raw[~gated]
    if len(raw) < 8:
        return BandwidthStats(nan, "mostly digital silence", nan, None, gated_pct)
    frames = raw * np.hanning(nfft)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    tiny = 1e-30  # -300 dB: guards log(0) for bands that are exactly zero
    fr_db = 10.0 * np.log10(np.mean(frames**2, axis=1) + tiny)
    active = fr_db > np.percentile(fr_db, 10) + 15.0
    if active.sum() < 4:  # no speech/noise contrast: use loudest 30 %
        active = fr_db >= np.percentile(fr_db, 70)

    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    nb = int(round(sr / 2 / band_hz))
    edges = np.arange(nb + 1) * band_hz
    b = np.clip((freqs // band_hz).astype(int), 0, nb - 1)
    band_pow = np.zeros((len(frames), nb))
    for k in range(nb):  # sum bins into bands per frame
        band_pow[:, k] = spec[:, b == k].sum(axis=1)
    lv = 10.0 * np.log10(band_pow + tiny)
    s_db = np.percentile(lv[active], 90, axis=0)
    n_db = np.percentile(lv, 10, axis=0)

    ok = ((s_db - n_db) >= excess_db) & (s_db >= s_db.max() - dynamic_range_db)
    okf = np.convolve(ok.astype(float), np.ones(5) / 5, mode="same") > 0.5
    hits = np.flatnonzero(okf)
    snr_edge = float(edges[hits[-1] + 1]) if len(hits) else nan

    cliff = None
    win = max(2, int(round(cliff_window_hz / band_hz)))
    run_max = np.array([s_db[max(0, k - win) : k + 1].max() for k in range(nb)])
    # stop 1 kHz below Nyquist: every ADC / resampler anti-alias filter (and
    # bilinear-transform IIRs) rolls off there; that is not a band limit.
    top = nb - int(round(1000.0 / band_hz))
    for j in range(int(cliff_min_hz / band_hz), top):
        if s_db[j] < run_max[j] - cliff_drop_db and float(np.max(s_db[j:])) < run_max[j] - 10.0:
            cliff = float(edges[j])
            break

    if not len(hits) and cliff is None:
        return BandwidthStats(nan, "no band with speech above noise", nan, None, gated_pct)
    candidates = [(snr_edge, "snr")] if len(hits) else []
    if cliff is not None:
        candidates.append((cliff, "cliff"))
    bw, why = min(candidates)
    if why == "snr" and bw >= edges[-1] - 1000.0:
        why = "nyquist"
    return BandwidthStats(bw, why, snr_edge, cliff, gated_pct)
