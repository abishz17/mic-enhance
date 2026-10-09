# AGENTS.md

`mic-enhance` makes laptop-mic and AirPods speech clean and loudness-normalised while keeping the speaker's voice.
The README covers usage; this file covers how the code fits together and the rules every change keeps.

## Commands

Run from the repo root. A change is done when all of these pass:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --release                              # debug builds of the DSP are too slow to test
uv run --project ml --group dev pytest ml/tests
uv run --project eval pytest eval/tests
```

`testdata/` (smoke clips, VoiceBank sets) and `corpus/` (private real recordings) are git-ignored and usually
absent. Tests that read them skip with a `skipping: ... not present` message; Rust tests report this as a pass, so
read the `--nocapture` output before claiming those paths were exercised. New tests that need audio should
generate it, as `voice_wav` in `crates/cli/tests/e2e.rs` does.

## Map

| Path | What lives there |
|---|---|
| `crates/dsp` | Rust DSP library. `chain.rs` holds the post-restore stage order and the reason for each position; `config.rs` holds every setting and its `Default` |
| `crates/cli/src/main.rs` | Argument parsing, job planning (output paths, overwrite checks), per-file pipeline, summary |
| `crates/cli/src/media.rs` | Every ffmpeg/ffprobe call: decode, probe, mux audio back into video |
| `crates/cli/src/restore.rs` | `Restorer` trait: built-in RNNoise, the `mlstage` subprocess, fallback wrapper |
| `crates/cli/presets/` | Built-in TOML presets, compiled into the binary |
| `ml/` | uv project `mlstage`: one Python CLI in front of many models (`backends/`). `bakeoff/` scripts and `BAKEOFF.md` record the model comparison |
| `eval/` | uv project `mic-eval`: measurement harness (DNSMOS, loudness, bandwidth, voice identity, invented-word guard, ABX) |

Pipeline: load → `dsp::pre` (DC, high-pass, hum) → restore (mlstage, RNNoise or none) → `dsp::chain` (EQ,
dynamics, loudness, true-peak limiter) → write WAV, or re-mux into the original video.

## Invariants

**Clean mode never generates audio.** Clean mode may only filter, mask (attenuate per band) and apply gain. A model
that synthesises a waveform (vocoder, diffusion, codec resynthesis) belongs to studio mode: register it in
`ml/src/mlstage/backends/` with `mode="studio"`. Clean-mode fallbacks are also masking models (RNNoise). Studio mode
is an explicit opt-in, so a studio failure is an error, never a silent fallback.

**The speaker keeps their voice.** Any change to restore, EQ, tone correction or dynamics is checked on real speech
with `uv run --project eval mic-eval compare --ref raw.wav --cand out.wav`; it must PASS both the voice-identity
check (speaker similarity, F0 shift and range, spectral tilt; limits in `THRESHOLDS` in
`eval/src/mic_eval/voice.py`) and the invented-word guard. Defaults favour the natural voice: a blind listening
comparison preferred it, which is why tone correction and the presence/air boosts are off by default.

**Inputs are never overwritten.** `plan_jobs` in `main.rs` resolves every output path before any processing,
refuses an output that is the input itself (even with `--force`), refuses an existing output without `--force`,
and refuses two inputs writing the same output. Videos are written to a temp file in the destination folder and
moved into place with `persist_noclobber`.

**ffmpeg only opens local files of known container formats.** Every input goes through `input_args` in `media.rs`:
`-protocol_whitelist file` plus `FORMAT_WHITELIST`, which leaves out playlist and script demuxers (hls, concat,
...) because they can make ffmpeg fetch URLs or read other local files. Paths go through `path_arg` (a `./` prefix
stops `-x` or `http:...` file names being parsed as options or URLs), commands run without a shell, and ffmpeg gets
`-nostdin`. `playlist_and_concat_disguised_as_audio_are_rejected` in `crates/cli/tests/media.rs` guards this.

**The mlstage contract is fixed on both sides.** The Rust CLI runs `uv run --project ml mlstage --model M --in
a.wav --out b.wav`. The Python side (docstring of `ml/src/mlstage/cli.py`) writes a 48 kHz mono float32 WAV,
time-aligned with the input, and prints exactly one JSON line on stdout with at least `model`, `seconds` and
`rtf`; all logs go to stderr; exit code 0 on success, 2 on bad input, 1 on model failure. The Rust side
(`MlStage` in `restore.rs`) reads the last non-empty stdout line, rejects output whose length differs by more than
100 ms, validates model names against `[A-Za-z0-9._-]` with no leading `-`, and kills the whole process group on
timeout. A change to either side changes both, plus the fake-mlstage tests in `restore.rs` and `e2e.rs`.

**Presets equal `Default`.** `crates/cli/presets/youtube.toml` spells out every default, and a test asserts it
equals `Preset::default()`. Change a default in `crates/dsp/src/config.rs` (or `RestoreSettings` in
`preset.rs`) and in `youtube.toml` together. `podcast.toml` lists only what differs.

## Models

The default clean model is `deepfilternet3`. Model weights are downloaded on first use from pinned revisions and,
where the backend controls the download, checked against a sha256 (`ml/src/mlstage/weights.py`). The
`mossformer2-48k`, `resemble-*` and `voicefixer` backends need isolated environments under `ml/envs/`, which is
git-ignored; they exist only for the bake-off.
