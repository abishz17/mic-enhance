# mic-enhance

`mic-enhance` is a command-line tool for people who record voice-overs, tutorials or podcasts on a laptop mic or
AirPods. It removes background noise and mains hum, evens out the level, and delivers a file at YouTube loudness
(−14 LUFS) or podcast loudness (−16 LUFS). It does not change how you sound. Everything runs on your own computer.

```console
$ mic-enhance talk.wav
mic-enhance  talk.wav -> talk.enhanced.wav
  source     48000 Hz, 1 ch, 32-bit float, 15.7 s
  input      -23.51 LUFS, true peak -7.74 dBTP, sample peak -7.75 dBFS
  hum        50.00 Hz, notched harmonics 1x
  restore    deepfilternet3 via mlstage, 1.6 s (real-time factor 0.10)
  tone       natural (no tone shaping; --tone-strength 0..1 corrects mic colouring)
  expander   threshold -72.3 dBFS, 1 open/close changes; max 8.0 dB, mean 6.4 dB while active, active 13% of the time
  compressor max 8.5 dB, mean 4.7 dB while active, active 81% of the time; make-up +5.6 dB
  de-esser   max 6.0 dB, mean 2.4 dB while active, active 27% of the time
  loudness   +10.6 dB in 2 pass(es)
  limiter    max 2.0 dB, mean 1.0 dB while active, active 3% of the time
  output     -14.01 LUFS, true peak -1.04 dBTP, sample peak -1.05 dBFS
  time       2.54 s (6x real time)
```

## What it does not do

- **Room echo** is reduced only slightly. Moving closer to the mic fixes echo far better than any software.
- **Bluetooth bandwidth.** AirPods and other Bluetooth mics cut off around 8–12 kHz. The default mode does not
  invent the missing top end.
- **Clipping** in the original recording cannot be repaired.
- **Studio mode** (`--mode studio`, generative voice rebuilding) is experimental and not recommended. The models
  tested so far change the voice: pitch errors of up to 80 cents, and inserted words.

## Install

You need:

- Rust 1.85 or newer
- [uv](https://docs.astral.sh/uv/)
- ffmpeg and ffprobe on your `PATH`. Without them you can only process WAV files.

```sh
git clone https://github.com/abishz17/mic-enhance
cd mic-enhance
cargo install --path crates/cli   # puts `mic-enhance` on your PATH
uv sync --project ml              # Python environment for the noise remover (about 1 GB)
```

The installed binary records where the source tree's `ml/` folder is, so keep the checkout where it is, or point
`MIC_ENHANCE_ML` (or `--ml-project`) at it. The first run downloads the DeepFilterNet3 weights.

If the Python side is missing or fails, `mic-enhance` still runs. It falls back to a built-in RNNoise noise
remover and prints a warning saying why.

## Usage

```sh
mic-enhance talk.wav                    # writes talk.enhanced.wav next to the input
mic-enhance memo.m4a                    # any audio ffmpeg reads; the output is a 24-bit WAV
mic-enhance clip.mov                    # writes clip.enhanced.mov: same picture, new sound
mic-enhance *.mov -o enhanced/          # several files into a folder
mic-enhance talk.wav --preset podcast   # -16 LUFS instead of -14
```

Existing files are never overwritten unless you pass `--force`, and the input itself is never overwritten.

**Video** (mp4, mov, m4v, mkv, webm): the video stream is copied bit for bit, and only the audio is replaced. The
new audio is AAC at 320 kbps, 48 kHz (Opus for webm), with the voice on both stereo channels. The peak ceiling for
video is −1.5 dBTP instead of −1 dBTP, because AAC encoding overshoots slightly.

Useful options (see `mic-enhance --help` for all of them):

| Option | Effect |
|---|---|
| `--preset podcast` | −16 LUFS instead of −14 |
| `--restore none` | Filters and dynamics only, no noise remover |
| `--restore rnnoise` | Built-in lightweight noise remover; no Python needed |
| `--tone-strength 0..1` | Turn on automatic tone correction toward a neutral speech curve. Worth trying on thin Bluetooth takes |
| `--set post.additive_eq.presence.enabled=true` | Add a +2 dB "broadcast" presence lift at 4 kHz (`.air` adds a shelf above 11 kHz) |
| `--set KEY=VALUE` | Override any preset value. `--print-config` lists them all |
| `--dump-stages DIR` | Save the audio after every stage, for A/B listening |

## How it works

| # | Stage | What it does |
|---|---|---|
| 1 | Load | Decodes any format through ffmpeg to 48 kHz mono |
| 2 | Clean-up filters | Removes DC offset and applies an 80 Hz 4th-order high-pass. Notches mains hum only if hum is detected |
| 3 | Noise removal | DeepFilterNet3, a masking model: it can only turn sound down, never generate it. A timbre-match EQ restores the top end the model dulls. Attenuation is capped at `clamp(68 − input SNR, 20, 45)` dB, so a faint natural room tone remains |
| 4 | Tone | −2.5 dB at 300 Hz against laptop boxiness, plus an all-pass phase rotator (no change in tone) that makes the waveform more symmetric so the limiter works less. Optional adaptive tone correction (`--tone-strength`) is off by default: in a blind listening comparison the natural version was preferred |
| 5 | Dynamics | Gentle expander between phrases, 3:1 compressor, de-esser |
| 6 | Loudness | −14 LUFS (podcast: −16), then a true-peak limiter at −1 dBTP (−1.5 for video) |

The DSP stages are a Rust library (`crates/dsp`). The noise remover is a Python program (`ml/`) that the CLI runs
as a subprocess.

## Results and how they are measured

Measured on real laptop-mic and AirPods recordings with the default settings, on an Apple M2:

- Loudness lands at −14.0 LUFS.
- Room tone drops from about −50…−58 dBFS to −66…−72 dBFS.
- DNSMOS overall quality goes from 2.83–3.10 to 3.24–3.27 on laptop-mic takes. AirPods takes, which are already
  processed on the device, stay at about 2.95–3.00.
- Speaker similarity to the original is 0.93–0.94, and pitch moves by at most 0.1 semitone.
- Processing runs about 10–30× faster than real time.

The measurement tools are in [`eval/`](eval/). They cover DNSMOS, loudness and true peak, bandwidth, a Whisper-based
check for invented words, ECAPA voice-identity checks, and blind A/B/ABX listening tests:

```sh
uv run --project eval mic-eval score out.wav                         # loudness, peaks, noise floor, bandwidth, DNSMOS
uv run --project eval mic-eval compare --ref raw.wav --cand out.wav  # voice identity and invented-word check
uv run --project eval mic-eval abx raw.wav out.wav                   # blind listening test, loudness-matched (macOS: plays via afplay)
```

The comparison of noise-removal models (DeepFilterNet3, DPDFNet, RNNoise, GTCRN, LavaSR, Sidon and others) is in
[`ml/BAKEOFF.md`](ml/BAKEOFF.md). Generative models were rejected for the default mode because they change the voice.

## Privacy

Audio never leaves your computer. Network access is limited to downloading the Python packages and model weights
during setup and on first use.

## Development

```sh
cargo test --release                                  # Rust: DSP, CLI, ffmpeg handling
uv run --project ml --group dev pytest ml/tests       # ML stage
uv run --project eval pytest eval/tests               # measurement tools
```

The test recordings in `testdata/` are not part of the repository. Tests that need them skip themselves when the
files are absent. [`AGENTS.md`](AGENTS.md) describes the architecture and the rules every change must keep.
