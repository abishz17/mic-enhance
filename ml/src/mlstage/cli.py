"""mlstage: run one enhancement model on a WAV file.

    mlstage --model <name> --in a.wav --out b.wav [--strength 0..1] [--device auto|cpu|mps]

Contract (relied on by the Rust `mic-enhance` CLI):
  * reads any WAV soundfile can parse (any rate, any channel count -> mono downmix)
  * writes WAV 48 kHz mono float32 with exactly round(n_in * 48000 / sr_in) samples,
    time-aligned with the input (model latency compensated)
  * stdout carries exactly one JSON line: {"model", "seconds", "rtf", "device", "atten_lim_db"}
    (atten_lim_db: noise-attenuation cap actually used, dB, or null); everything
    else (logs, library chatter) goes to stderr. stdout fd is redirected to stderr while
    models run, so even C-level prints cannot corrupt the JSON line.
  * exit 0 on success; 2 on bad arguments / unreadable input; 1 on model failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

log = logging.getLogger("mlstage")


def _peak_footprint_bytes() -> int | None:
    """macOS lifetime-max physical footprint of this process (includes Metal/GPU allocations)."""
    if sys.platform != "darwin":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    import ctypes

    class RUsageInfoV4(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16), ("v", ctypes.c_uint64 * 40)]

    info = RUsageInfoV4()
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        return None
    return int(info.v[28])  # ri_lifetime_max_phys_footprint


def _pick_device(req: str, allowed: tuple[str, ...]) -> str:
    import importlib.util

    def mps_ok() -> bool:
        if importlib.util.find_spec("torch") is None:
            return False
        import torch

        return torch.backends.mps.is_available()

    if req == "auto":
        for d in allowed:
            if d == "cpu" or (d == "mps" and mps_ok()):
                return d
        return "cpu"
    if req not in allowed:
        raise ValueError(f"device '{req}' not verified for this model (verified: {', '.join(allowed)})")
    if req == "mps" and not mps_ok():
        raise ValueError("MPS requested but not available")
    return req


def enhance_array(be, x, sr: int, strength: float = 1.0, timbre: bool = False,
                  atten_lim_db: float | None = None):
    """The whole model stage on an in-memory signal (used by the CLI and the bake-off).

    Order: model -> attenuation limit -> timbre match -> strength mix.
    atten_lim_db: cap on how much noise is removed (DeepFilterNet's `atten_lim_db`, applied
    model-independently): y = g*x + (1-g)*y with g = 10^(-lim/20). For a masking model this is
    exactly DF's spectral formula (both the mask output and the input are sample-aligned and the
    STFT is linear), so residual noise sits `lim` dB below the original noise and keeps its natural
    character instead of dropping to digital silence. Only valid for phase-coherent (clean) models.
    """
    import numpy as np

    from .audio import OUT_SR, fix_length, out_len, resample
    from .chunking import process_chunked

    info = be.info
    n48 = out_len(len(x), sr, OUT_SR)
    xin = resample(x, sr, info.in_sr)
    be.prepare(xin)
    y = process_chunked(xin, info.in_sr, info.out_sr, be.process, info.chunk_s, info.context_s)
    y = be.finish(y)
    y = fix_length(resample(y, info.out_sr, OUT_SR), n48)
    need_dry = strength < 1.0 or timbre or atten_lim_db is not None
    dry = fix_length(resample(x, sr, OUT_SR), n48) if need_dry else None
    if atten_lim_db is not None:
        if info.mode != "clean":
            raise ValueError("--atten-lim-db only applies to clean (masking) models")
        g = 10 ** (-abs(atten_lim_db) / 20)
        y = g * dry + (1 - g) * y
    if timbre:
        from .dsp import timbre_match

        y, gains = timbre_match(dry, y)
        log.info("timbre-match EQ (dB): %s", " ".join(f"{c / 1000:.1f}k:{g:+.1f}" for c, g in gains.items()))
    if strength < 1.0:
        y = strength * y + (1.0 - strength) * dry
    if not np.all(np.isfinite(y)):
        raise RuntimeError("model produced NaN/Inf samples")
    return y.astype(np.float32)


def run(args) -> dict:
    import numpy as np

    from . import backends
    from .audio import read_mono, write_48k_f32

    t0 = time.perf_counter()
    cls = backends.get(args.model)
    info = cls.info
    x, sr = read_mono(args.inp)
    dur = len(x) / sr
    device = _pick_device(args.device, info.devices)
    log.info("model=%s mode=%s device=%s input=%s (%.2fs @ %d Hz)", info.name, info.mode, device, args.inp, dur, sr)

    be = cls(device)
    try:
        t_load = time.perf_counter() - t0
        tm = args.timbre_match == "on" or (args.timbre_match == "auto" and info.mode == "clean")
        if args.atten_lim_db == "auto":
            lim = None
            if info.mode == "clean":
                from .dsp import auto_atten_lim_db, speech_snr_db

                snr = speech_snr_db(x, sr)
                lim = auto_atten_lim_db(snr)
                log.info("input speech-to-noise %.1f dB -> attenuation limit clamp(68 - SNR, 20, 45) = %.1f dB", snr, lim)
        else:
            lim = None if args.atten_lim_db == "off" else float(args.atten_lim_db)
        log.info("timbre-match %s, attenuation limit %s", "on" if tm else "off", f"{lim:g} dB" if lim else "off")
        y = enhance_array(be, x, sr, args.strength, tm, lim)
    finally:
        be.close()
    write_48k_f32(args.out, y)

    secs = time.perf_counter() - t0
    peak = _peak_footprint_bytes()
    log.info("done in %.2fs (load %.2fs), rtf %.3f, peak footprint %s MB", secs, t_load, secs / dur,
             f"{peak / 2**20:.0f}" if peak else "?")
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump({"model": info.name, "device": device, "seconds": secs, "load_seconds": t_load,
                       "audio_seconds": dur, "rtf": secs / dur, "rtf_excl_load": (secs - t_load) / dur,
                       "peak_footprint_mb": peak / 2**20 if peak else None, "out_peak": float(np.abs(y).max())}, f)
    return {"model": info.name, "seconds": round(secs, 3), "rtf": round(secs / dur, 4), "device": device,
            "atten_lim_db": None if lim is None else round(lim, 1)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mlstage", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="model name (see --list)")
    p.add_argument("--in", dest="inp", help="input WAV")
    p.add_argument("--out", help="output WAV (48 kHz mono f32)")
    p.add_argument("--strength", type=float, default=1.0,
                   help="wet fraction 0..1: out = s*model + (1-s)*input (default 1)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--timbre-match", default="auto", choices=["auto", "on", "off"],
                   help="restore speech-band level the model removed (static EQ, see dsp.timbre_match); "
                        "auto (default) = on for clean-mode models, off for generative")
    p.add_argument("--atten-lim-db", default="auto", metavar="DB|auto|off",
                   help="max noise attenuation in dB, keeps a natural noise floor; auto (default) = "
                        "clamp(68 - input SNR, 20, 45) for clean-mode models, off for generative")
    p.add_argument("--stats", help="also write timing/memory stats JSON here")
    p.add_argument("--list", action="store_true", help="list models and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if args.verbose else logging.INFO,
                        format="mlstage: %(levelname)s: %(message)s")
    for noisy in ("httpx", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if args.list:
        from .backends import REGISTRY

        print("\n".join(sorted(REGISTRY)))
        return
    if not (args.model and args.inp and args.out):
        p.error("--model, --in and --out are required")
    if not 0.0 <= args.strength <= 1.0:
        p.error("--strength must be in [0, 1]")
    if args.atten_lim_db not in ("auto", "off"):
        try:
            if not 0 < float(args.atten_lim_db) <= 120:
                raise ValueError
        except ValueError:
            p.error("--atten-lim-db must be a number in (0, 120], 'auto' or 'off'")

    # keep fd 1 clean: anything any library writes to stdout lands on stderr
    sys.stdout.flush()
    real_stdout = os.dup(1)
    os.dup2(2, 1)
    code, result = 0, None
    try:
        result = run(args)
    except (KeyError, ValueError, FileNotFoundError) as e:
        log.error("%s", e.args[0] if e.args else e)
        code = 2
    except Exception as e:  # noqa: BLE001 - top-level: report and exit nonzero
        from .audio import AudioError

        code = 2 if isinstance(e, AudioError) else 1
        log.error("%s: %s", type(e).__name__, e, exc_info=args.verbose)
    finally:
        sys.stdout.flush()
        os.dup2(real_stdout, 1)
        os.close(real_stdout)
    if result is not None:
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    sys.exit(code)


if __name__ == "__main__":
    main()
