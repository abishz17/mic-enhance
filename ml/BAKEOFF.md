# ML bake-off: speech enhancement for mic-enhance (Apple M2)

Status: **iteration 1, clean mode decided; studio mode only partly measured.** Every number below was measured on
one machine (M2, 16 GB, macOS 27, Python 3.11, torch 2.8.0). MossFormer2-48K, Resemble Enhance and VoiceFixer run
in their own environments under `ml/envs/` (not included in the repository) and are not scored yet.

"Corpus" below means three private recordings of one speaker: a laptop mic at normal distance (`laptop-normal`),
the same laptop mic close up (`laptop-close`) and AirPods (`airpods`). They are not distributed.

## Recommendation

| mode | default | alternative | why |
|---|---|---|---|
| clean | **`deepfilternet3`** (mlstage defaults: `--timbre-match auto` = on, `--atten-lim-db auto` = 30 dB) | `dpdfnet8-48k` | Same quality and identity scores as the alternative, 10–20× faster (cold RTF 0.05 vs 0.45), MIT/Apache license. The timbre-match EQ undoes DF3's measured loss of 4–14 dB of high frequencies in bands where the input was clearly speech, not noise. |
| studio (experimental) | `deepfilternet3` → `lavasr` (bandwidth extension only, above the codec cutoff) | `sidon` (not recommended) | LavaSR only adds audio **above** the measured codec cutoff (about 13 kHz on AirPods), so everything below it is the real recording, sample for sample. Its output is fully coherent with the input (coherence 1.00 below the cutoff). Sidon regenerates the whole waveform: speaker similarity drops to 0.89–0.92, pitch error p90 is 40–80 cents, and coherence with the input is 0.02. |

**Studio `--strength`:** a dry/wet mix only works when the wet signal has the same phase as the dry one. With
LavaSR it does (below the cutoff wet = dry, so strength just sets the gain of the generated top band). With Sidon it
does not (coherence 0.01–0.11 in every band; a 50/50 mix loses 2.5 dB, which means comb filtering). So strength is
defined as the gain of the generated band above the cutoff, applied by `mlstage --strength`, which is the dry/wet
mix with the stage's own input.

## Defaults decided after iteration 1

- **`--timbre-match auto` is the default.** It is on for clean (masking) models and off for generative ones. The
  restore stage owns its own side effects, so the Rust contract stays "restore preserves timbre". This EQ is
  referenced to the input; the Rust chain's tone-correction stage is referenced to a target, so the two don't
  overlap. On the reference sets it costs nothing: DF3 vs DF3+tm on vbdemand gives OVRL 3.31 / 3.31, PESQ
  2.49 / 2.49, Δspk −0.019 / −0.019.
- **`--atten-lim-db auto` = 30 dB** for clean models, `off` for generative ones (rejected with exit code 2, because
  mixing a generative output with its phase-incoherent input comb-filters).
  - It is applied independently of the model: `y = g·x + (1−g)·y` with `g = 10^(−lim/20)`. For a sample-aligned
    masking model this is exactly DeepFilterNet's own spectral `atten_lim_db` formula, because the STFT is linear.
    So it works the same way for DF3, DPDFNet and RNNoise.
  - Order of operations: model → attenuation limit → timbre match → strength.

Room tone (first 4 s, RMS) after the **full Rust chain** (`mic-enhance --restore deepfilternet3`, output −14 LUFS),
by attenuation limit:

| file (input room tone, raw) | 20 dB | 30 dB | 40 dB | off |
|---|---|---|---|---|
| laptop-normal (−47.3 dBFS, speech-to-noise about 18 dB) | −42.3 | **−53.0** | −63.2 | −89.2 |
| laptop-close (−56.0, about 32 dB) | −57.3 | **−68.2** | −82.0 | −95.2 |
| airpods (−51.2) | −58.1 | **−68.1** | −84.4 | −102.6 |

At 30 dB, the close and AirPods recordings land in the −60 to −70 dBFS target. laptop-normal lands at −53 and needs
about 40 dB. A fixed limit cannot meet the target for every input, because the floor that remains depends on the
input's own SNR. **Now implemented as the default** (`--atten-lim-db auto`): `lim = clamp(68 − SNR, 20, 45)`. The table directly below is the first try, which used 55. SNR is measured as
the mean power of the top 30 % of 20 ms frames over the bottom 20 %, with exact-zero frames excluded
(`dsp.speech_snr_db`, tested in `ml/tests/test_atten_lim.py`). The value is logged to stderr and returned as
`atten_lim_db` in the JSON line.

| file | measured SNR | chosen limit | room tone after full chain (−14 LUFS) |
|---|---|---|---|
| laptop-normal | 24.3 dB | 30.7 dB | −50.9 dBFS |
| laptop-close | 33.8 dB | 21.2 dB | −57.3 dBFS |
| airpods | 36.0 dB | 20.0 dB | −58.1 dBFS |

All three files sit **about 3–9 dB above the −60 dBFS edge of the target**: −50.9, −57.3 and −58.1. The Rust chain's
dynamics close the speech-to-floor gap by roughly 15 dB after mlstage: a 30 dB limit gave −53 on normal, where
the arithmetic predicts about −69. Raising the constant to about 68 (`clamp(68 − SNR, 20, 45)`) would land the
files around −60 to −68 dBFS. **Applied (68):**

| file | limit | room tone after full chain |
|---|---|---|
| laptop-normal | 43.7 dB | −64.3 dBFS |
| laptop-close | 34.2 dB | −69.8 dBFS |
| airpods | 32.0 dB | −69.1 dBFS |

The constant compensates for the Rust compressor, which is a known coupling between the two stages. Iteration 2
should target the final output floor directly instead.

## Identity pass criteria (proposed)

ECAPA speaker embeddings are built to ignore the recording channel, so they barely notice EQ. The measured
anchors from `calibrate_identity.py` show this:

| manipulation | ECAPA cos (laptop-close) |
|---|---|
| lowpass 8 kHz, ±1 dB/oct tilt, +6 dB presence peak | 0.997–1.000 (**it cannot see these**) |
| tilt −3 dB/oct | 0.979 |
| white noise at 30 / 20 / 10 dB SNR | 0.966 / 0.906 / 0.784 |
| pitch +25 / −25 / +100 cents (phase vocoder) | 0.891 / 0.844 / 0.673 |
| same VB speaker, different sentence (mean) | 0.785 |
| different speaker (mean) | 0.101 |

That is why the criteria combine three kinds of check. They are measured on `laptop-close`, whose 32 dB
speech-to-noise ratio means any change there comes from the model, not from noise removal:
1. **Speaker cosine(input, output) ≥ 0.95.** Removing noise from a 32 dB-SNR file should move the embedding no more
   than adding 30 dB-SNR noise does (0.966). On the reference sets, the output must not move further from the clean
   voice: Δspk ≥ −0.02.
2. **Median F0 shift ≤ 10 cents**, and the F0 range change (10th–90th percentile) ≤ 0.5 semitone. pYIN works on a
   10-cent grid, so 10 cents is the smallest shift it can resolve.
3. **Octave-band levels (125 Hz–8 kHz) within ±1.5 dB** of the input on speech frames, wherever the input's
   speech-to-noise margin in that band is > 20 dB. The 16 kHz octave must be within −3 dB.

Results: `deepfilternet3 + timbre-match` passes all three on laptop-close (0.959; 0 cents; 125 Hz–8 kHz within
−0.9…0.0 dB; 16 kHz octave −2.7 dB). Plain DF3, DPDFNet and RNNoise fail check 3 at 8 kHz (−4 to −5 dB) and at
16 kHz (−13 to −15 dB). Sidon fails checks 1 and 3.

## High-frequency loss compared with the input

Every clean model makes the top octaves duller, **including where the input band is clearly speech** (25–29 dB
above the noise floor on laptop-close):

| laptop-close, speech frames | 4k octave | 8k octave (5.7–11 kHz) | 16k octave (11–22 kHz) |
|---|---|---|---|
| rnnoise | −2.0 | −5.4 | −14.9 |
| deepfilternet3 | −1.4 | −4.4 | −13.8 |
| dpdfnet8-48k | −1.2 | −4.2 | −12.8 |
| dpdfnet2-48k | −1.0 | −4.4 | −13.0 |
| gtcrn (16 kHz model) | −2.8 | −7.6 | −104 (no band) |
| **deepfilternet3 + timbre-match** | **0.0** | **0.0** | **−2.7** (boost capped at 12 dB) |

On AirPods the loss is small (DF3: −0.9 dB at 8k, −3.3 dB at 16k). On laptop-normal the 8k and 16k bands sit only
11–12 dB above the noise floor, so timbre-match deliberately leaves them alone: there, that energy counts as noise.

**Timbre-match** (`mlstage/dsp.py:timbre_match`) works like this. In each 1/3-octave band above 2 kHz it compares
input and output on the same speech frames. It restores the lost level in bands where the input was ≥ 20 dB above its
noise floor, does nothing below 12 dB, and ramps linearly in between. It never cuts and boosts at most 12 dB. It is
applied as one static linear-phase FIR of 2047 taps, delay-compensated. Because it is a fixed EQ it cannot invent
audio and cannot pump. Room tone after DF3 is about −110 dBFS, so a +12 dB high-frequency boost there stays inaudible.

## Results: corpus

Δ columns are output minus raw input. `spk` = ECAPA cosine(input, output). `room` = level of the first 4 s (room tone) in dBFS. `gaps` = level where the input is exact digital zero. WER = Whisper small.en vs the transcript of the raw input (ins/del/sub word counts).

**laptop-close**

| model | SIG | BAK | OVRL | spk | F0 Δ cents (p90 abs) | F0 range Δ st | tilt Δ dB/oct | presence Δ dB | room dBFS | bw kHz | >12k dB | WER (i/d/s) | lag ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input | 3.46 | 4.01 | 3.14 | 1.000 | +0.0 (0) | – | -7.86 | -15.0 | -56 | 24.0 | -93 | 0.00 (0/0/0) | +0.00 |
| deepfilternet3 | 3.47 | 4.17 | 3.24 | 0.959 | +0.0 (11) | +0.00 | -0.28 | -0.3 | -110 | 10.4 | -107 | 0.06 (0/3/2) | +0.00 |
| dpdfnet8-48k | 3.52 | 4.19 | 3.29 | 0.957 | +0.0 (10) | +0.00 | -0.26 | -0.5 | -132 | 10.4 | -106 | 0.04 (2/1/1) | -0.00 |
| dpdfnet8-48k@0.9 | 3.57 | 4.19 | 3.35 | 0.975 | +0.0 (10) | +0.00 | -0.26 | -0.5 | -76 | 10.4 | -105 | 0.04 (2/1/1) | -0.00 |
| dpdfnet2-48k | 3.49 | 4.17 | 3.27 | 0.961 | +0.0 (10) | +0.00 | -0.26 | -0.5 | -98 | 10.4 | -106 | 0.09 (3/2/3) | -0.00 |
| rnnoise | 3.38 | 4.17 | 3.16 | 0.927 | +0.0 (10) | +0.00 | -0.59 | -0.7 | -96 | 10.1 | -108 | 0.16 (3/5/6) | -0.02 |
| lavasr | 3.46 | 4.01 | 3.14 | 1.000 | +0.0 (0) | +0.00 | +0.00 | +0.0 | -56 | 24.0 | -93 | 0.00 (0/0/0) | -0.00 |
| lavasr-dn | 3.47 | 4.15 | 3.22 | 0.976 | +0.0 (10) | +0.00 | -0.64 | -0.9 | -73 | 12.3 | -101 | 0.06 (0/2/3) | +0.00 |
| sidon | 3.61 | 4.20 | 3.39 | 0.891 | +0.0 (70) | +0.10 | +0.26 | +0.7 | -77 | 15.0 | -102 | 0.07 (0/2/4) | -1.30 |
| deepfilternet3+lavasr | 3.47 | 4.17 | 3.24 | 0.959 | +0.0 (11) | +0.00 | -0.28 | -0.3 | -110 | 10.4 | -107 | 0.06 (0/3/2) | +0.00 |
| deepfilternet3~tm | 3.48 | 4.17 | 3.24 | 0.959 | +0.0 (20) | -0.07 | +0.11 | +0.7 | -110 | 16.9 | -96 | 0.03 (0/2/1) | +0.00 |
| dpdfnet8-48k~tm | 3.52 | 4.19 | 3.29 | 0.956 | +0.0 (10) | +0.08 | +0.11 | +0.3 | -123 | 17.6 | -95 | 0.04 (2/1/1) | -0.00 |

**laptop-normal**

| model | SIG | BAK | OVRL | spk | F0 Δ cents (p90 abs) | F0 range Δ st | tilt Δ dB/oct | presence Δ dB | room dBFS | bw kHz | >12k dB | WER (i/d/s) | lag ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input | 3.45 | 3.52 | 2.87 | 1.000 | +0.0 (0) | – | -7.60 | -16.3 | -47 | 24.0 | -101 | 0.00 (0/0/0) | +0.00 |
| deepfilternet3 | 3.52 | 4.20 | 3.28 | 0.943 | +0.0 (20) | +0.64 | -0.28 | -0.2 | -119 | 10.8 | -115 | 0.21 (7/0/6) | +0.00 |
| dpdfnet8-48k | 3.58 | 4.19 | 3.33 | 0.947 | +0.0 (10) | -0.10 | -0.23 | -0.3 | -126 | 10.9 | -113 | 0.23 (9/0/5) | -0.00 |
| dpdfnet8-48k@0.9 | 3.58 | 4.18 | 3.33 | 0.980 | +0.0 (10) | -0.30 | -0.23 | -0.3 | -67 | 10.9 | -113 | 0.25 (8/2/5) | -0.00 |
| dpdfnet2-48k | 3.53 | 4.15 | 3.26 | 0.913 | +0.0 (10) | -0.10 | -0.20 | -0.3 | -94 | 10.9 | -114 | 0.18 (7/0/4) | -0.00 |
| rnnoise | 3.45 | 4.19 | 3.21 | 0.880 | +0.0 (10) | +0.00 | -0.29 | -0.3 | -101 | 10.7 | -115 | 0.18 (6/0/5) | -0.01 |
| lavasr | 3.45 | 3.52 | 2.87 | 1.000 | +0.0 (0) | +0.00 | +0.00 | +0.0 | -47 | 24.0 | -101 | 0.08 (0/1/4) | +0.00 |
| lavasr-dn | 3.44 | 4.16 | 3.19 | 0.974 | +0.0 (10) | +0.50 | -0.36 | -0.6 | -74 | 15.3 | -104 | 0.08 (0/0/5) | -0.00 |
| sidon | 3.65 | 4.20 | 3.43 | 0.915 | +0.0 (80) | +3.66 | +0.34 | +2.3 | -75 | 13.9 | -105 | 0.31 (11/2/6) | +10.71 |
| deepfilternet3+lavasr | 3.52 | 4.20 | 3.28 | 0.943 | +0.0 (20) | +0.64 | -0.28 | -0.2 | -119 | 10.8 | -115 | 0.21 (7/0/6) | +0.00 |
| deepfilternet3~tm | 3.53 | 4.20 | 3.29 | 0.943 | +0.0 (30) | +0.60 | -0.19 | +0.6 | -119 | 10.9 | -114 | 0.23 (8/0/6) | +0.00 |
| dpdfnet8-48k~tm | 3.58 | 4.19 | 3.34 | 0.947 | +0.0 (10) | -0.10 | -0.19 | +0.0 | -121 | 11.0 | -112 | 0.13 (1/2/5) | -0.00 |

**airpods**

| model | SIG | BAK | OVRL | spk | F0 Δ cents (p90 abs) | F0 range Δ st | tilt Δ dB/oct | presence Δ dB | room dBFS | gaps dBFS | bw kHz | >12k dB | WER (i/d/s) | lag ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input | 3.56 | 3.45 | 2.97 | 1.000 | +0.0 (0) | – | -7.63 | -18.0 | -51 | -200 | 11.5 | -109 | 0.00 (0/0/0) | +0.00 |
| deepfilternet3 | 3.56 | 4.10 | 3.30 | 0.976 | +0.0 (0) | +0.00 | -0.06 | -0.1 | -106 | -134 | 11.4 | -114 | 0.00 (0/0/0) | -0.00 |
| dpdfnet8-48k | 3.51 | 4.16 | 3.28 | 0.971 | +0.0 (0) | +0.00 | -0.05 | -0.0 | -142 | -187 | 11.5 | -112 | 0.00 (0/0/0) | -0.00 |
| dpdfnet8-48k@0.9 | 3.56 | 4.11 | 3.31 | 0.984 | +0.0 (0) | +0.00 | -0.05 | -0.0 | -71 | -188 | 11.5 | -112 | 0.00 (0/0/0) | -0.00 |
| dpdfnet2-48k | 3.56 | 4.17 | 3.33 | 0.968 | +0.0 (0) | +0.00 | -0.04 | -0.0 | -138 | -171 | 11.4 | -114 | 0.00 (0/0/0) | -0.00 |
| rnnoise | 3.42 | 4.14 | 3.19 | 0.956 | +0.0 (10) | +0.00 | -0.09 | -0.1 | -91 | -143 | 11.4 | -113 | 0.03 (1/0/1) | -0.01 |
| lavasr | 3.56 | 3.45 | 2.97 | 1.000 | +0.0 (0) | +0.00 | +0.00 | +0.0 | -51 | -75 | 15.5 | -94 | 0.01 (1/0/0) | +0.00 |
| lavasr-dn | 3.43 | 4.04 | 3.14 | 0.971 | +0.0 (0) | +0.00 | -0.15 | -0.1 | -63 | -75 | 15.5 | -94 | 0.00 (0/0/0) | -0.00 |
| sidon | 3.70 | 4.19 | 3.48 | 0.904 | +0.0 (40) | -0.10 | +0.14 | +2.3 | -66 | -68 | 12.1 | -99 | 0.04 (0/2/1) | -1.26 |
| deepfilternet3+lavasr | 3.56 | 4.10 | 3.30 | 0.976 | +0.0 (0) | +0.00 | -0.06 | -0.1 | -75 | -75 | 15.5 | -95 | 0.00 (0/0/0) | -0.00 |
| deepfilternet3~tm | 3.56 | 4.10 | 3.30 | 0.975 | +0.0 (0) | +0.00 | +0.01 | +0.3 | -106 | -134 | 11.4 | -114 | 0.00 (0/0/0) | -0.00 |
| dpdfnet8-48k~tm | 3.51 | 4.16 | 3.28 | 0.971 | +0.0 (0) | +0.00 | -0.00 | +0.1 | -142 | -187 | 11.5 | -111 | 0.00 (0/0/0) | -0.00 |

## Results: real speech with references

`spk→ref` = ECAPA cosine(clean reference, output); `Δspk` = that minus the same for the raw input (>0: output is closer to the true voice than the input was). WER vs the official transcript.

**vbdemand** — noisy, SNR 2.5 dB (8 clips)

| model | SIG | BAK | OVRL | PESQ-WB | SI-SDR dB | spk→ref | Δspk | F0 Δ vs ref cents | tilt Δ vs ref | bw kHz | WER | ins | del |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input | 2.73 | 2.34 | 2.14 | 1.39 | 1.7 | 0.856 | +0.000 | +0.0 | +0.23 | 5.7 | 0.020 | 0 | 0 |
| deepfilternet3 | 3.54 | 4.14 | 3.31 | 2.49 | 15.6 | 0.836 | -0.019 | +0.0 | -0.15 | 14.3 | 0.051 | 0 | 1 |
| dpdfnet8-48k | 3.53 | 4.10 | 3.29 | 2.41 | 16.4 | 0.855 | -0.001 | +0.0 | +0.03 | 13.9 | 0.065 | 2 | 0 |
| dpdfnet2-48k | 3.48 | 4.03 | 3.21 | 2.49 | 16.2 | 0.853 | -0.002 | +0.0 | +0.04 | 14.4 | 0.054 | 1 | 0 |
| rnnoise | 3.26 | 3.75 | 2.90 | 1.75 | 9.7 | 0.802 | -0.054 | +0.0 | -0.19 | 13.7 | 0.034 | 0 | 1 |
| lavasr | 2.73 | 2.34 | 2.14 | 1.39 | 1.7 | 0.856 | +0.000 | +0.0 | +0.23 | 5.7 | 0.020 | 0 | 0 |
| lavasr-dn | 3.16 | 3.72 | 2.79 | 1.96 | 10.8 | 0.811 | -0.045 | +0.0 | -0.10 | 15.1 | 0.063 | 0 | 1 |
| sidon | 3.71 | 4.19 | 3.50 | 2.38 | -10.4 | 0.827 | -0.028 | +0.0 | +0.07 | 15.7 | 0.039 | 0 | 1 |
| deepfilternet3+lavasr | 3.54 | 4.14 | 3.31 | 2.49 | 15.6 | 0.836 | -0.019 | +0.0 | -0.15 | 14.3 | 0.051 | 0 | 1 |
| deepfilternet3~tm | 3.54 | 4.14 | 3.31 | 2.49 | 15.6 | 0.836 | -0.019 | +0.0 | -0.14 | 14.3 | 0.051 | 0 | 1 |

**vbreverb** — noisy + real-RIR reverb (8 clips)

| model | SIG | BAK | OVRL | PESQ-WB | SI-SDR dB | spk→ref | Δspk | F0 Δ vs ref cents | tilt Δ vs ref | bw kHz | WER | ins | del |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input | 3.16 | 2.47 | 2.25 | 1.30 | -3.8 | 0.824 | +0.000 | +12.5 | +1.66 | 8.2 | 0.026 | 0 | 0 |
| deepfilternet3 | 3.37 | 4.02 | 3.09 | 1.62 | -2.8 | 0.806 | -0.018 | +16.2 | +2.25 | 12.3 | 0.049 | 0 | 1 |
| dpdfnet8-48k | 3.38 | 3.99 | 3.08 | 1.52 | -3.3 | 0.845 | +0.021 | +15.0 | +1.65 | 12.0 | 0.042 | 0 | 1 |
| dpdfnet2-48k | 3.37 | 4.01 | 3.09 | 1.59 | -3.3 | 0.827 | +0.004 | +15.0 | +1.70 | 12.1 | 0.042 | 0 | 1 |
| rnnoise | 3.07 | 3.61 | 2.70 | 1.37 | -3.7 | 0.736 | -0.088 | +17.5 | +1.61 | 11.6 | 0.291 | 5 | 6 |
| lavasr | 3.16 | 2.47 | 2.25 | 1.30 | -3.8 | 0.824 | +0.000 | +12.5 | +1.66 | 8.2 | 0.026 | 0 | 0 |
| lavasr-dn | 3.14 | 3.61 | 2.71 | 1.40 | -3.5 | 0.778 | -0.046 | +16.2 | +1.29 | 15.0 | 0.064 | 0 | 1 |
| sidon | 3.69 | 4.19 | 3.48 | 2.23 | -12.3 | 0.827 | +0.003 | +5.0 | -0.03 | 15.2 | 0.042 | 0 | 2 |
| deepfilternet3+lavasr | 3.37 | 4.02 | 3.09 | 1.62 | -2.8 | 0.806 | -0.018 | +16.2 | +2.25 | 12.3 | 0.049 | 0 | 1 |
| deepfilternet3~tm | 3.39 | 4.02 | 3.11 | 1.62 | -2.9 | 0.806 | -0.018 | +15.6 | +2.45 | 12.3 | 0.057 | 0 | 2 |

**vbbt_msbc** — Bluetooth HFP mSBC round trip, 8 kHz band (6 clips)

| model | SIG | BAK | OVRL | PESQ-WB | SI-SDR dB | spk→ref | Δspk | F0 Δ vs ref cents | tilt Δ vs ref | bw kHz | WER | ins | del |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input | 3.61 | 3.71 | 3.14 | 2.76 | 15.5 | 0.957 | +0.000 | +0.0 | +0.02 | 5.1 | 0.013 | 0 | 1 |
| deepfilternet3 | 3.65 | 4.16 | 3.41 | 3.11 | 19.2 | 0.957 | -0.001 | +0.0 | -0.00 | 6.7 | 0.024 | 0 | 1 |
| dpdfnet8-48k | 3.54 | 4.15 | 3.32 | 2.98 | 19.5 | 0.947 | -0.010 | +0.0 | -0.04 | 6.2 | 0.023 | 0 | 1 |
| dpdfnet2-48k | 3.55 | 4.16 | 3.34 | 2.91 | 19.5 | 0.934 | -0.023 | +0.0 | +0.05 | 6.6 | 0.023 | 0 | 1 |
| rnnoise | 3.52 | 4.06 | 3.26 | 2.46 | 14.3 | 0.907 | -0.050 | +0.0 | -0.02 | 6.5 | 0.013 | 0 | 1 |
| lavasr | 3.61 | 3.71 | 3.14 | 2.76 | 15.2 | 0.957 | +0.000 | +0.0 | +0.02 | 13.0 | 0.013 | 0 | 1 |
| lavasr-dn | 3.48 | 4.04 | 3.20 | 2.44 | 16.2 | 0.948 | -0.010 | +0.0 | -0.09 | 17.5 | 0.024 | 0 | 1 |
| sidon | 3.68 | 4.18 | 3.45 | 2.88 | -9.8 | 0.910 | -0.047 | +0.0 | -0.21 | 14.9 | 0.011 | 0 | 0 |
| deepfilternet3+lavasr | 3.65 | 4.16 | 3.41 | 3.11 | 18.9 | 0.957 | -0.001 | +0.0 | -0.00 | 16.4 | 0.024 | 0 | 1 |
| deepfilternet3~tm | 3.65 | 4.16 | 3.41 | 3.10 | 19.2 | 0.957 | -0.001 | +0.0 | +0.03 | 6.7 | 0.024 | 0 | 1 |

**vbbt_eld** — AAC-ELD 24 kHz round trip, ~11 kHz band (6 clips)

| model | SIG | BAK | OVRL | PESQ-WB | SI-SDR dB | spk→ref | Δspk | F0 Δ vs ref cents | tilt Δ vs ref | bw kHz | WER | ins | del |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input | 3.61 | 3.73 | 3.15 | 2.75 | 15.5 | 0.969 | +0.000 | +0.0 | +0.01 | 6.9 | 0.013 | 0 | 1 |
| deepfilternet3 | 3.64 | 4.14 | 3.40 | 3.21 | 19.3 | 0.966 | -0.003 | +0.0 | -0.08 | 10.5 | 0.013 | 0 | 1 |
| dpdfnet8-48k | 3.56 | 4.14 | 3.33 | 2.97 | 19.6 | 0.962 | -0.007 | +0.0 | +0.04 | 10.5 | 0.013 | 0 | 1 |
| dpdfnet2-48k | 3.54 | 4.17 | 3.33 | 2.99 | 19.6 | 0.959 | -0.010 | +0.0 | +0.03 | 10.5 | 0.013 | 0 | 1 |
| rnnoise | 3.47 | 4.00 | 3.18 | 2.45 | 14.3 | 0.920 | -0.050 | +0.0 | +0.01 | 10.8 | 0.023 | 0 | 1 |
| lavasr | 3.61 | 3.73 | 3.15 | 2.75 | 15.4 | 0.969 | -0.000 | +0.0 | +0.01 | 14.0 | 0.013 | 0 | 1 |
| lavasr-dn | 3.51 | 4.07 | 3.24 | 2.61 | 16.4 | 0.956 | -0.013 | +0.0 | -0.11 | 16.9 | 0.024 | 0 | 1 |
| sidon | 3.68 | 4.18 | 3.45 | 2.88 | -9.7 | 0.918 | -0.052 | +0.0 | -0.05 | 15.0 | 0.013 | 0 | 1 |
| deepfilternet3+lavasr | 3.64 | 4.14 | 3.40 | 3.21 | 19.1 | 0.966 | -0.003 | +0.0 | -0.08 | 15.6 | 0.013 | 0 | 1 |
| deepfilternet3~tm | 3.64 | 4.14 | 3.40 | 3.21 | 19.3 | 0.966 | -0.004 | +0.0 | -0.04 | 10.5 | 0.013 | 0 | 1 |

## Timbre: octave-band level change on speech frames (corpus)


**laptop-close** — octave-band level change on speech frames (dB, output − input)

| model | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | 16k |
|---|---|---|---|---|---|---|---|---|---|
| *input speech-to-noise margin* | *17* | *35* | *38* | *42* | *39* | *38* | *30* | *25* | *29* |
| deepfilternet3 | -6.2 | -0.9 | -0.6 | -0.7 | -0.8 | -0.7 | -1.4 | -4.4 | -13.8 |
| deepfilternet3~tm | -6.2 | -0.9 | -0.6 | -0.7 | -0.8 | -0.3 | +0.0 | -0.0 | -2.7 |
| dpdfnet8-48k | -12.1 | -1.0 | -0.5 | -0.3 | -0.3 | -0.4 | -1.2 | -4.2 | -12.8 |
| dpdfnet8-48k~tm | -12.1 | -1.0 | -0.5 | -0.3 | -0.3 | -0.2 | +0.0 | +0.0 | -2.1 |
| dpdfnet2-48k | -9.6 | -0.7 | -0.4 | -0.2 | -0.3 | -0.4 | -1.0 | -4.4 | -13.0 |
| rnnoise | -2.8 | -0.5 | -0.3 | -0.2 | -0.3 | -0.5 | -2.0 | -5.4 | -14.9 |
| gtcrn | -6.0 | -0.8 | -0.5 | -0.3 | -0.4 | -0.7 | -2.8 | -7.6 | -103.7 |
| lavasr-dn | -5.9 | -0.7 | -0.5 | -0.3 | -0.3 | -0.6 | -2.5 | -4.6 | -6.5 |
| sidon | -15.0 | -3.4 | -3.1 | -0.4 | -0.4 | +1.5 | -1.9 | -2.9 | -9.1 |

**laptop-normal** — octave-band level change on speech frames (dB, output − input)

| model | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | 16k |
|---|---|---|---|---|---|---|---|---|---|
| *input speech-to-noise margin* | *10* | *25* | *31* | *33* | *28* | *23* | *16* | *11* | *12* |
| deepfilternet3 | -8.9 | -1.0 | -0.8 | -0.8 | -0.7 | -0.7 | -1.1 | -2.1 | -11.1 |
| deepfilternet3~tm | -8.9 | -1.0 | -0.8 | -0.8 | -0.7 | -0.4 | -0.5 | -2.0 | -9.6 |
| dpdfnet8-48k | -11.6 | -1.0 | -0.4 | -0.2 | -0.2 | -0.2 | -0.5 | -1.8 | -10.0 |
| dpdfnet8-48k~tm | -11.6 | -1.0 | -0.4 | -0.2 | -0.2 | -0.1 | -0.3 | -1.7 | -8.8 |
| dpdfnet2-48k | -11.0 | -1.0 | -0.3 | -0.1 | -0.1 | -0.2 | -0.5 | -1.8 | -9.7 |
| rnnoise | -5.2 | -0.8 | -0.4 | -0.2 | -0.3 | -0.3 | -0.7 | -3.3 | -11.1 |
| gtcrn | -6.3 | -1.0 | -0.6 | -0.4 | -0.4 | -0.5 | -1.3 | -4.8 | -95.8 |
| lavasr-dn | -5.5 | -1.0 | -0.4 | -0.2 | -0.3 | -0.3 | -0.9 | -2.9 | -0.8 |
| sidon | -6.5 | +1.6 | +1.0 | +3.1 | +4.7 | +6.1 | +3.3 | +1.1 | -3.0 |

**airpods** — octave-band level change on speech frames (dB, output − input)

| model | 63 | 125 | 250 | 500 | 1k | 2k | 4k | 8k | 16k |
|---|---|---|---|---|---|---|---|---|---|
| *input speech-to-noise margin* | *21* | *44* | *48* | *50* | *37* | *32* | *23* | *15* | *13* |
| deepfilternet3 | -1.8 | -0.2 | -0.2 | -0.2 | -0.3 | -0.3 | -0.4 | -0.9 | -3.3 |
| deepfilternet3~tm | -1.8 | -0.2 | -0.2 | -0.2 | -0.3 | -0.1 | -0.0 | -0.6 | -2.9 |
| dpdfnet8-48k | -8.2 | -0.4 | -0.0 | -0.1 | -0.1 | -0.1 | -0.2 | -1.1 | -2.8 |
| dpdfnet8-48k~tm | -8.2 | -0.4 | -0.0 | -0.1 | -0.1 | -0.0 | -0.0 | -0.8 | -2.4 |
| dpdfnet2-48k | -8.3 | -0.4 | -0.0 | +0.0 | +0.0 | +0.0 | -0.2 | -1.1 | -4.4 |
| rnnoise | -1.9 | -0.1 | -0.2 | -0.1 | -0.1 | -0.1 | -0.3 | -1.5 | -3.9 |
| gtcrn | -0.3 | -0.0 | -0.0 | -0.0 | -0.0 | -0.1 | -0.3 | -5.6 | -92.8 |
| lavasr-dn | -0.9 | -0.1 | -0.1 | -0.0 | -0.1 | -0.1 | -0.3 | -4.7 | +7.8 |
| sidon | -6.8 | -2.2 | -1.8 | -3.6 | -3.6 | -0.2 | -0.2 | -4.4 | +3.9 |

## Candidates, licenses, weights

| model | mode | rate | license (code / weights) | weights source, format | device verified |
|---|---|---|---|---|---|
| deepfilternet3 | clean | 48k | MIT or Apache-2.0 | GitHub release `DeepFilterNet3.zip`, **torch pickle** `.ckpt` | CPU (MPS output matches to 133 dB SNR but runs 3.7× slower) |
| dpdfnet8-48k / dpdfnet2-48k (Ceva, Dec 2025) | clean | 48k | Apache-2.0 | HF `Ceva-IP/DPDFNet@dd6818d0`, ONNX | CPU (onnxruntime) |
| rnnoise (baseline) | clean | 48k | BSD-3 (xiph), wrapper `pyrnnoise` Apache-2.0 | compiled into the dylib | CPU |
| gtcrn | clean | **16k only** | MIT | GitHub `Xiaobin-Rong/gtcrn@9202557c` ONNX, sha256 pinned | CPU |
| mossformer2-48k | clean | 48k | Apache-2.0 | HF `alibabasglab/MossFormer2_SE_48K`, torch pickle | own env (`ml/envs/clearvoice`), not scored yet |
| resemble-denoise / resemble-enhance | clean / studio | 44.1k | MIT | HF `ResembleAI/resemble-enhance`, DeepSpeed **torch pickle** | own env, not scored yet |
| voicefixer | studio | 44.1k | MIT | Zenodo, **torch pickle** | own env, not scored yet |
| lavasr / lavasr-dn (v2, Interspeech 2026) | studio | 16k→48k | Apache-2.0 | HF `YatharthS/LavaSR@b98dc8be`, torch pickle loaded with `weights_only=True`; model code vendored in `ml/src/mlstage/vendor/` (Vocos MIT, UL-UNAS MIT) | lavasr: CPU and MPS (MPS vs CPU 113–123 dB SNR); **lavasr-dn: CPU only (MPS differs, only 18–23 dB SNR)** |
| sidon v0.1 (2025) | studio | 16k→48k | MIT (HF card) | HF `sarulab-speech/sidon-v0.1@b3b02d8b`, **TorchScript** archives, sha256 pinned | **CPU only**: the official `*_cpu.pt` graphs create CPU tensors internally ("Passed CPU tensor to MPS op") |
| VoiceBridge (2025) | – | – | – | no released weights found → skipped | – |

Pickle checkpoints run code when loaded. DF3, MossFormer2, Resemble and VoiceFixer all use them, so only load
them from the pinned official sources. LavaSR is loaded with `weights_only=True`, which is the safe loader.

## Timing, length and memory (cold CLI runs through `uv run --project ml mlstage`)

Every model returns exactly `round(n_in·48000/sr_in)` samples. The lag of the output against the input is measured by
band-limited cross-correlation (`bakeoff/align_check.py`):
- 0.000 ms for DF3, DPDFNet, GTCRN and LavaSR.
- RNNoise has a fixed 960-sample (20 ms) delay. It is now compensated, leaving a residual of −0.02 ms.
- Sidon has about 7.8 ms of latency, now compensated. Because Sidon generates its own phase, a ±2 ms jitter
  remains from clip to clip.

| model | clip | dur s | wall s | RTF | load s | peak footprint MB |
|---|---|---|---|---|---|---|
| deepfilternet3 | airpods 37 s | 37.1 | 1.7 | 0.047 | 0.9 | 1040 |
| deepfilternet3 (20 s chunks) | 10-min file | 600 | 35.6* | 0.059* | 1.6 | 1179 |
| dpdfnet8-48k | 10-min file | 600 | 275 | 0.459 | 0.3 | 757 |
| dpdfnet2-48k | 10-min file | 600 | 88 | 0.146 | 0.3 | 819 |
| rnnoise | 10-min file | 600 | 36 | 0.060 | 0.4 | 505 |
| gtcrn | 10-min file | 600 | 19 | 0.031 | 0.2 | 589 |
| lavasr | 10-min file | 600 | 9.5 | 0.016 | 1.0 | 2134† |
| sidon | 3.5–37 s clips (warm, CPU) | – | – | median 1.24, max 2.29 | 8 | 4357 |

\* measured while scoring jobs were loading the CPU. Uncontended DF3 runs at RTF 0.024 on the same file
(with 60 s chunks, but those peaked at 2.8 GB). † measured before the bandwidth estimator was rewritten to use
constant memory, so it is lower now; re-measure.

"Peak footprint" is the macOS lifetime-max physical footprint of the Python process, which includes Metal
allocations. Long files are processed in chunks with 1 s of context on each side and a 0.5 s sin² crossfade.
Whole-file and chunked DF3 outputs differ by 31–34 dB SNR. The difference is spread across the file, not peaked
at the seams: the recurrent state simply diverges slightly.

## End to end with the Rust CLI

`./target/release/mic-enhance --restore deepfilternet3 -o out.wav laptop-close.wav` exits 0. It reports
input −29.4 LUFS → output −14.00 LUFS, −1.04 dBTP; the mlstage stage took 6.0 s and the whole run 11 s (both
with the CPU loaded). `--restore dpdfnet8-48k` also works (38.6 s for the stage). The Rust side passes no new flags: timbre-match and the
30 dB attenuation limit are mlstage defaults for clean models.

## Observations that matter

- **Room tone and gaps.** DF3 takes the 4 s room tone from −56 to −110 dBFS, and DPDFNet takes it to −132 to
  −142 dBFS, which is effectively digital silence between phrases. That is why the 30 dB attenuation limit is now the default
  (see the table above). LavaSR **adds** HF hiss at about −75 dBFS into the
  AirPods' exact-zero gaps (DF3 output alone: −134).
- **Babble / hallucination.** No clean model inserted words. Most transcript changes are single-word ASR flips
  (a→the, would→will). On laptop-normal both DF3 and DPDFNet produce the *same* changed transcript ("Avi", "mic
  now", an extra "I want it to be", "Yeah"). Two independent masking models agreeing suggests the denoised audio
  shows words the raw-input transcript missed; nobody has listened yet to confirm. Sidon on laptop-normal: WER
  0.31 against the raw transcript (11 insertions, 2 deletions, 6 substitutions), F0 range +3.7 semitones, presence
  +2.3 dB, timing +10.7 ms: it changes the voice.
- **AirPods 9–11 kHz SBR-like bump.** The masking models keep it: they are linear per band, and the input-vs-output
  level at 8k changes by only −0.9 dB. LavaSR keeps the original below the cutoff, so it keeps the bump too, and then
  generates 13–24 kHz (+6 to +8 dB in the 16k octave against an empty input). Sidon resynthesises everything, the bump
  included (8k octave −4.4 dB).
- **AirPods bandwidth extension (LavaSR) on the AAC-ELD simulation** (`vbbt_eld`, ~11 kHz band): no change in
  speaker similarity (0.969 → 0.969, and 0.966 → 0.966 after DF3) and no change in WER; measured bandwidth goes
  from 10.5 to 15.6 kHz. Whether the generated 13–20 kHz sounds natural has **not** been checked against the true
  high band yet: the `lsd_12_20k` metric is implemented but these runs have not been re-scored with it. Nobody has
  listened yet.
- **GTCRN** has only 16 kHz weights, so it drops everything above 8 kHz. Rejected for full-band laptop-mic input.

## Reproduce (exact commands, run from the repo root)

```sh
# 0. env (Python 3.11 pinned in ml/.python-version; torch 2.8.0 pinned: DeepFilterNet 0.5.6 needs torchaudio.backend)
uv sync --project ml --group bakeoff
# if Hugging Face xet downloads stall, plain HTTP works:
export HF_HUB_DISABLE_XET=1

# 1. contract
uv run --project ml mlstage --model deepfilternet3 --in testdata/smoke/degraded_noise_hum_room.wav --out o.wav   # defaults: --timbre-match auto, --atten-lim-db auto (30 dB)

# 2. real test sets (VoiceBank-DEMAND + noisy-reverberant, CC BY 4.0, ~500 MB of zips, delete afterwards)
B=https://datashare.ed.ac.uk/server/api/core/bitstreams; D=$(mktemp -d)
curl -L -o $D/clean_testset_wav.zip       $B/dec213d3-bf57-4777-9663-c24bdce92d5e/content
curl -L -o $D/noisy_testset_wav.zip       $B/13c1bfbf-14a6-41db-9b41-8f7310f01ad5/content
curl -L -o $D/noisyreverb_testset_wav.zip $B/56b70bb6-a6c7-4302-8875-6be66314fd58/content
curl -L -o $D/testset_txt.zip             $B/09e12592-879e-4749-bf6d-e34cf0d1ddf6/content
curl -L -o $D/logfiles.zip                $B/11185dc8-9cf1-405b-b858-35bd6a04aedd/content
curl -L -o $D/logfiles_reverb.zip         $B/9beb6e20-1f00-4215-a51e-9b2036ff2791/content
uv run --project ml --group bakeoff python -I ml/bakeoff/prepare_testdata.py --dl $D   # -> testdata/vb*/ (+ mSBC / AAC-ELD sims via ffmpeg)

# 3. eval models (local, sha256-verified): ECAPA (speechbrain/spkrec-ecapa-voxceleb@0f99f2d0) -> ml/models/ecapa,
#    Whisper (Systran/faster-whisper-small.en@d1d751a5, model.bin sha256 62b2a45b...) -> ml/models/faster-whisper-small.en

# 4. run + score + tables
uv run --project ml --group bakeoff python ml/bakeoff/run_models.py --models deepfilternet3,deepfilternet3~tm,dpdfnet8-48k,dpdfnet2-48k,rnnoise,gtcrn,lavasr,lavasr-dn,sidon,deepfilternet3+lavasr
uv run --project ml --group bakeoff python ml/bakeoff/score.py
uv run --project ml --group bakeoff python ml/bakeoff/aggregate.py > ml/out/tables.md
uv run --project ml --group bakeoff python ml/bakeoff/timbre.py --models deepfilternet3,deepfilternet3~tm,dpdfnet8-48k,rnnoise
uv run --project ml --group bakeoff python ml/bakeoff/calibrate_identity.py
uv run --project ml --group bakeoff python ml/bakeoff/align_check.py deepfilternet3 rnnoise sidon
uv run --project ml --group bakeoff python ml/bakeoff/mps_check.py deepfilternet3 lavasr lavasr-dn sidon
uv run --project ml --group bakeoff python ml/bakeoff/mix_check.py --dry deepfilternet3 --wet sidon,deepfilternet3+lavasr
uv run --project ml --group bakeoff python ml/bakeoff/perf.py deepfilternet3,dpdfnet8-48k --clips <airpods take>.wav,<10-min wav>
```
Model-name syntax in the bake-off scripts: `a+b` means a chain (a's output feeds b), `m@0.9` means `--strength 0.9`,
and `m~tm` means timbre-match on. Outputs go to `ml/out/<name>/<set>/<clip>.wav` and per-file metrics to
`ml/out/_scores/`.

Metric definitions (`ml/bakeoff/metrics.py`):
- DNSMOS P.835 (`speechmos` ONNX) is computed at 16 kHz, so it cannot see bandwidth extension.
- PESQ-WB is computed at 16 kHz.
- SI-SDR is computed after aligning to the reference (reverb sets lag by the RIR delay). It is meaningless for
  generative outputs.
- Effective bandwidth is the highest band where speech frames are > 10 dB above quiet frames and within 60 dB of
  the peak band.
- ECAPA cosine is computed at 16 kHz.
- F0 comes from pYIN, 60–400 Hz, 10 ms hop.
- WER uses faster-whisper small.en with beam 5, no VAD, against the official text, the TTS clean transcript, or (on
  the corpus) the raw input's transcript.

## Unverified / open

- **No listening test yet.** In particular: whether timbre-match sounds natural, whether the generated LavaSR top
  band is natural, and whether DF3's near-silent room tone sounds gated.
- MossFormer2-48K, Resemble (denoise and enhance) and VoiceFixer are not scored. Their environments
  live in `ml/envs/`, which is not included in the repository.
- Corpus WER is measured against the raw-input transcript from Whisper small.en, which itself contains errors
  ("Micronaut", "Arvius"). Treat it as "transcript changed", not "words invented". The test set is small: 3 corpus
  files and 28 VoiceBank clips from two speakers.
- Bake-off tables were scored **without** the attenuation limit; the `m~tm` rows are DF3 + timbre-match at full
  attenuation. GTCRN, `dpdfnet8-48k~tm` and the studio rows not listed here were never scored, because studio work
  is deferred.
- LSD in the 12–20 kHz band against the true high band (`lsd_12_20k`) is implemented but not yet in the tables.
