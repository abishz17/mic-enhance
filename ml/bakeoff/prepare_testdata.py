"""Build the real-speech test sets under testdata/ (gitignored) from the Edinburgh DataShare zips.

Sources (CC BY 4.0, Valentini-Botinhao et al., 48 kHz):
  VoiceBank-DEMAND test set      https://datashare.ed.ac.uk/handle/10283/2791
  Noisy reverberant VoiceBank    https://datashare.ed.ac.uk/handle/10283/2826  (real RIRs: ACE, MIRD)
Download (≈500 MB, only the test zips; delete afterwards):
  see ml/BAKEOFF.md "Reproduce" for the curl lines.

    uv run --project ml --group bakeoff python ml/bakeoff/prepare_testdata.py --dl <dir with zips>

Sets written (48 kHz mono float32; reference = VoiceBank clean, transcript = official .txt):
  vbdemand/   8 noisy clips, SNR 2.5 dB (hardest condition), 4 per speaker
  vbreverb/   same 8 utterances from the noisy-reverberant set
  vbbt_msbc/  6 clips at SNR 12.5-17.5 dB through a Bluetooth HFP mSBC round trip (16 kHz -> 8 kHz bandwidth)
  vbbt_eld/   same 6 clips through AAC-ELD at 24 kHz / 64 kbps via macOS AudioToolbox
              (~11 kHz bandwidth with a hard cliff above, like an AirPods recording)
"""

from __future__ import annotations

import argparse
import io
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "testdata"


def read_zip_wav(z: zipfile.ZipFile, name: str) -> np.ndarray:
    path = next(n for n in z.namelist() if n.endswith(f"/{name}.wav") and "__MACOSX" not in n)
    x, sr = sf.read(io.BytesIO(z.read(path)), dtype="float32")
    assert sr == 48_000, (name, sr)
    return x


def write(p: Path, x: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p, x.astype(np.float32), 48_000, subtype="FLOAT")


def codec_roundtrip(x: np.ndarray, enc_args: list[str], ext: str, dec_args: list[str]) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        src, mid, dst = Path(d) / "in.wav", Path(d) / f"mid.{ext}", Path(d) / "out.wav"
        sf.write(src, x, 48_000, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *enc_args, str(mid)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", *dec_args, "-i", str(mid), "-ar", "48000", "-ac", "1",
                        "-c:a", "pcm_f32le", str(dst)], check=True)
        y, sr = sf.read(dst, dtype="float32")
    return align_to(x, y)


def align_to(ref: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Remove codec priming delay by cross-correlation so the reference stays sample-aligned."""
    import scipy.signal as ss

    n = min(len(ref), len(y), 48_000 * 3)
    c = ss.correlate(y[:n], ref[:n], mode="full", method="fft")
    lag = int(np.argmax(c[n - 1: n - 1 + 4800]))  # codecs only delay (0..100 ms)
    y = y[lag:]
    return np.pad(y, (0, max(0, len(ref) - len(y))))[: len(ref)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", required=True, type=Path)
    a = ap.parse_args()
    z_clean = zipfile.ZipFile(a.dl / "clean_testset_wav.zip")
    z_noisy = zipfile.ZipFile(a.dl / "noisy_testset_wav.zip")
    z_rev = zipfile.ZipFile(a.dl / "noisyreverb_testset_wav.zip")
    z_txt = zipfile.ZipFile(a.dl / "testset_txt.zip")
    log = {}
    for line in zipfile.ZipFile(a.dl / "logfiles.zip").read("log_testset.txt").decode().splitlines():
        k, noise, snr = line.split()
        log[k] = (noise, float(snr))
    rlog = {}
    for line in zipfile.ZipFile(a.dl / "logfiles_reverb.zip").read("logfiles/log_testset.txt").decode().splitlines():
        p = line.split()
        rlog[p[0]] = " ".join(p[1:])
    lens = {}
    for n in z_clean.namelist():
        if n.endswith(".wav") and "__MACOSX" not in n:
            k = Path(n).stem
            lens[k] = z_clean.getinfo(n).file_size / 2 / 48_000

    def pick(snrs, count_per_spk, exclude=()):
        out = []
        for spk in ("p232", "p257"):
            seen_noise = set()
            for k in sorted(log):
                noise, snr = log[k]
                if (k.startswith(spk) and snr in snrs and 3.0 <= lens.get(k, 0) <= 7.0 and k not in exclude
                        and noise not in seen_noise):
                    out.append(k)
                    seen_noise.add(noise)
                if sum(o.startswith(spk) for o in out) == count_per_spk:
                    break
        return out

    hard = pick({2.5}, 4)
    bt = pick({12.5, 17.5}, 3, exclude=set(hard))
    manifest = []
    for k in hard + bt:
        txt = z_txt.read(f"testset_txt/{k}.txt").decode().strip()
        clean = read_zip_wav(z_clean, k)
        sets = []
        if k in hard:
            write(TD / "vbdemand" / f"{k}.wav", read_zip_wav(z_noisy, k))
            write(TD / "vbreverb" / f"{k}.wav", read_zip_wav(z_rev, k))
            sets = ["vbdemand", "vbreverb"]
        else:
            noisy = read_zip_wav(z_noisy, k)
            msbc = codec_roundtrip(noisy, ["-ar", "16000", "-ac", "1", "-c:a", "sbc", "-msbc", "1"], "msbc", [])
            # profile 38 = AAC-ELD (AudioToolbox); at 24 kHz/64 kbps the decoded band ends ~11 kHz
            eld = codec_roundtrip(noisy, ["-ar", "24000", "-ac", "1", "-c:a", "aac_at", "-profile:a", "38",
                                          "-b:a", "64k"], "m4a", ["-c:a", "aac_at"])
            write(TD / "vbbt_msbc" / f"{k}.wav", msbc)
            write(TD / "vbbt_eld" / f"{k}.wav", eld)
            sets = ["vbbt_msbc", "vbbt_eld"]
        write(TD / "vbref" / f"{k}.wav", clean)
        (TD / "vbref" / f"{k}.txt").write_text(txt + "\n")
        manifest.append((k, ",".join(sets), log[k][0], log[k][1], rlog.get(k, ""), txt))
    with open(TD / "vbref" / "MANIFEST.tsv", "w") as f:
        f.write("id\tsets\tnoise\tsnr_db\treverb_condition\ttext\n")
        for m in manifest:
            f.write("\t".join(map(str, m)) + "\n")
    print(f"wrote {len(manifest)} utterances -> {TD}")


if __name__ == "__main__":
    main()
