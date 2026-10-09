//! `mic-enhance`: make laptop / AirPods speech sound production-ready.
//!
//! Orchestrates, per input file: decode (WAV directly, anything else via
//! ffmpeg) -> pre (DC, HPF, hum) -> restore (ML / rnnoise / none) -> post
//! chain (tone correction, EQ, dynamics, loudness, true-peak limiter) ->
//! write (WAV, or the original video with the new audio).

mod media;
mod preset;
mod restore;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use dsp::io::{AudioError, LoadLimits, OutputFormat};
use dsp::tone::ToneOutcome;
use dsp::{SAMPLE_RATE, chain, loudness, pre};
use media::Probe;
use preset::{Fallback, Preset};
use restore::{MlStage, Restorer, Rnnoise, WithFallback};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// Never invents audio: filtering, noise masking and dynamics only.
    Clean,
    /// Opt-in generative AI voice restoration (can re-synthesise speech).
    Studio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// 24-bit integer PCM (what editors and YouTube expect).
    Pcm24,
    /// 32-bit float.
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum HumArg {
    Auto,
    Off,
    #[value(name = "50")]
    Hz50,
    #[value(name = "60")]
    Hz60,
}

const LONG_ABOUT: &str = "\
Makes a voice recording from a laptop mic, AirPods or any other mic sound
clean, clear and evenly loud, ready for YouTube (-14 LUFS) or podcasts.

It removes background noise and mains hum, takes the boxiness out of
laptop mics, evens out loud and quiet words, and sets the final loudness.
Tone correction for boomy, dull or harsh mics is optional (--tone-strength). Nothing is uploaded anywhere; it all runs on this computer.

Inputs can be WAV, or (with ffmpeg installed) m4a, mp3, aac, flac, ogg,
and videos (mp4, mov, m4v, mkv, webm). For a video, the picture is copied
untouched and only the sound is replaced.";

const EXAMPLES: &str = "\
Examples:
  mic-enhance talk.wav                 writes talk.enhanced.wav next to it
  mic-enhance clip.mov                 writes clip.enhanced.mov (same picture, new sound)
  mic-enhance *.m4a -o enhanced/       processes many files into a folder
  mic-enhance talk.wav --preset podcast   -16 LUFS instead of -14
  mic-enhance talk.wav --restore none     skip noise removal

Existing files are never overwritten unless you add --force.";

#[derive(Debug, Parser)]
#[command(version, about = "Make spoken recordings sound production-ready.", long_about = LONG_ABOUT, after_help = EXAMPLES)]
struct Args {
    /// One or more recordings (WAV, or any audio/video format ffmpeg reads).
    #[arg(required = true, value_name = "INPUT")]
    inputs: Vec<PathBuf>,
    /// Where to write. A file name (one input only) or a folder.
    /// Default: `<name>.enhanced.<ext>` next to each input.
    #[arg(short, long, value_name = "FILE_OR_DIR")]
    output: Option<PathBuf>,
    /// Allow replacing files that already exist.
    #[arg(long)]
    force: bool,
    /// `clean` only removes and shapes; `studio` uses generative AI.
    #[arg(long, value_enum, default_value_t = Mode::Clean)]
    mode: Mode,
    /// Sound preset: `youtube` (-14 LUFS), `podcast` (-16 LUFS), or a
    /// path to your own TOML preset.
    #[arg(long, default_value = "youtube")]
    preset: String,
    /// Noise remover for clean mode: an AI model name run via mlstage
    /// (e.g. deepfilternet3), `rnnoise` (built in, no Python needed) or
    /// `none`. Default: the preset's model, falling back to rnnoise.
    #[arg(long, value_name = "MODEL")]
    restore: Option<String>,
    /// Generative model for studio mode (default: the preset's).
    #[arg(long, value_name = "MODEL")]
    studio_model: Option<String>,
    /// Save the sound after every processing step (for A/B listening).
    #[arg(long, value_name = "DIR")]
    dump_stages: Option<PathBuf>,
    /// Sample format of WAV outputs.
    #[arg(long, value_enum, default_value_t = Format::Pcm24)]
    format: Format,
    /// Change any preset value, e.g. `--set post.compressor.ratio=4`
    /// (see --print-config for the names).
    #[arg(long = "set", value_name = "KEY=VALUE")]
    overrides: Vec<String>,
    /// Target loudness in LUFS (shortcut for --set post.loudness.target_lufs=..).
    #[arg(long, allow_hyphen_values = true, value_name = "LUFS")]
    target_lufs: Option<f64>,
    /// Peak ceiling in dBTP, for every output type (default -1.0, or -1.5
    /// for videos because AAC encoding overshoots slightly).
    #[arg(long, allow_hyphen_values = true, value_name = "DBTP")]
    ceiling_dbtp: Option<f64>,
    /// How strongly to correct the mic's tone, 0 (off) to 1 (full).
    #[arg(long, value_name = "0..1")]
    tone_strength: Option<f64>,
    /// Mains hum removal: auto-detect, off, or force 50/60 Hz.
    #[arg(long, value_enum)]
    hum: Option<HumArg>,
    /// Refuse recordings longer than this many minutes (protects memory).
    #[arg(long, default_value_t = 60.0, value_name = "MINUTES")]
    max_minutes: f64,
    /// Folder of the AI models project (used via `uv run`). Default:
    /// $MIC_ENHANCE_ML, else the `ml/` folder of the source tree this
    /// binary was built from (so it works from any current folder).
    #[arg(long, value_name = "DIR")]
    ml_project: Option<PathBuf>,
    /// Run this program instead of `uv run --project <ml> mlstage`.
    #[arg(long, value_name = "PATH", hide = true)]
    mlstage_cmd: Option<PathBuf>,
    /// Print the full effective settings as TOML and exit.
    #[arg(long)]
    print_config: bool,
}

fn main() {
    if let Err(err) = run(Args::parse()) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

/// Preset + `--set` + dedicated flags, validated.
fn effective_preset(args: &Args) -> Result<Preset> {
    let mut overrides = args.overrides.clone();
    if let Some(v) = args.target_lufs {
        overrides.push(format!("post.loudness.target_lufs={v:?}"));
    }
    if let Some(v) = args.ceiling_dbtp {
        overrides.push(format!("post.limiter.ceiling_dbtp={v:?}"));
        overrides.push(format!("post.limiter.video_ceiling_dbtp={v:?}"));
    }
    if let Some(v) = args.tone_strength {
        overrides.push(format!("post.tone_correction.strength={v:?}"));
    }
    if let Some(h) = args.hum {
        let mode = match h {
            HumArg::Auto => "auto",
            HumArg::Off => "off",
            HumArg::Hz50 => "\"50\"",
            HumArg::Hz60 => "\"60\"",
        };
        overrides.push(format!("pre.hum.mode={mode}"));
    }
    let preset = Preset::load(&args.preset, &overrides)?;
    let fs = f64::from(SAMPLE_RATE);
    preset.pre.validate(fs)?;
    preset.post.validate(fs)?;
    Ok(preset)
}

/// Build the restore stage from mode, `--restore` and the preset.
/// `None` means "skip the stage".
fn build_restorer(args: &Args, preset: &Preset) -> Result<Option<Box<dyn Restorer>>> {
    let settings = &preset.restore;
    let timeout = Duration::from_secs(settings.timeout_s);
    let project = resolve_ml_project(args.ml_project.as_deref());
    let ml = |model: &str| -> Result<Box<dyn Restorer>> {
        Ok(match (&args.mlstage_cmd, &project) {
            (Some(cmd), _) => Box::new(MlStage::with_command(cmd.into(), vec![], model, timeout)?),
            (None, Ok(dir)) => Box::new(MlStage::via_uv(dir.clone(), model, timeout)?),
            // Not found: a stage that explains why when it is used, so the
            // fallback (or the studio-mode error) names the reason.
            (None, Err(reason)) => Box::new(Unavailable {
                model: model.to_owned(),
                reason: reason.clone(),
            }),
        })
    };

    match args.mode {
        Mode::Studio => {
            if args.restore.is_some() {
                bail!(
                    "--restore selects the clean-mode denoiser; use --studio-model with --mode studio"
                );
            }
            // Explicit opt-in: if the generative model fails we stop rather
            // than silently delivering a different kind of result.
            let model = args
                .studio_model
                .as_deref()
                .unwrap_or(&settings.studio_model);
            Ok(Some(ml(model)?))
        }
        Mode::Clean => {
            let choice = args.restore.as_deref().unwrap_or(&settings.clean_model);
            Ok(match choice {
                "none" => None,
                "rnnoise" => Some(Box::new(Rnnoise)),
                model => {
                    let primary = ml(model)?;
                    Some(match settings.fallback {
                        Fallback::Rnnoise => Box::new(WithFallback {
                            primary,
                            fallback: Box::new(Rnnoise),
                        }),
                        Fallback::None => primary,
                    })
                }
            })
        }
    }
}

/// Where the ML project lives: `--ml-project`, else `$MIC_ENHANCE_ML`,
/// else the source tree's `ml/` (baked in at compile time). A relative
/// `ml` default would only work when run from the repo root.
fn resolve_ml_project(arg: Option<&Path>) -> Result<PathBuf, String> {
    let (candidate, source) = match (arg, std::env::var_os("MIC_ENHANCE_ML")) {
        (Some(p), _) => (p.to_path_buf(), "--ml-project"),
        (None, Some(env)) => (PathBuf::from(env), "$MIC_ENHANCE_ML"),
        (None, None) => (
            PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../ml")),
            "the build's source tree",
        ),
    };
    match candidate.canonicalize() {
        Ok(dir) if dir.join("pyproject.toml").is_file() => Ok(dir),
        _ => Err(format!(
            "the ML project was not found at {} (from {source}); set MIC_ENHANCE_ML or --ml-project",
            candidate.display()
        )),
    }
}

/// Stand-in for an ML stage that can't run; fails with the reason.
struct Unavailable {
    model: String,
    reason: String,
}

impl Restorer for Unavailable {
    fn name(&self) -> String {
        self.model.clone()
    }

    fn restore(&self, _input: &[f32]) -> Result<restore::Restored> {
        bail!("{}", self.reason)
    }
}

/// Writes numbered stage files into the dump directory (if any).
struct StageDumper {
    dir: Option<PathBuf>,
    next: usize,
}

impl StageDumper {
    fn new(dir: Option<&Path>) -> Result<Self> {
        if let Some(d) = dir {
            std::fs::create_dir_all(d).with_context(|| format!("cannot create {}", d.display()))?;
        }
        Ok(Self {
            dir: dir.map(Path::to_path_buf),
            next: 0,
        })
    }

    /// File names are fixed (`NN_stage.wav`), never derived from input,
    /// so nothing can escape the dump directory.
    fn dump(&mut self, name: &str, audio: &[f32]) -> Result<(), dsp::io::AudioError> {
        let index = self.next;
        self.next += 1;
        match &self.dir {
            Some(dir) => dsp::io::write_wav(
                &dir.join(format!("{index:02}_{name}.wav")),
                audio,
                OutputFormat::Float32,
            )
            .map(|_| ()),
            None => Ok(()),
        }
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// What an input becomes.
enum Kind {
    /// Read with our own WAV reader (ffmpeg only as a fallback).
    Wav,
    /// Any other audio file: decoded by ffmpeg, written as WAV.
    Audio,
    /// Video: decoded by ffmpeg, re-muxed into the same container.
    Video { ext: String, probe: Probe },
}

/// One input -> one output, decided before any processing starts.
struct Job {
    input: PathBuf,
    output: PathBuf,
    kind: Kind,
}

fn classify(input: &Path, have_ffmpeg: bool) -> Result<Kind> {
    let ext = media::extension(input);
    if matches!(ext.as_str(), "wav" | "wave") {
        return Ok(Kind::Wav);
    }
    if !input.exists() {
        bail!("{} does not exist", input.display());
    }
    if !have_ffmpeg {
        bail!(
            "{}: reading .{ext} files needs ffmpeg, which was not found on PATH \
             (install ffmpeg, or convert the file to WAV first)",
            input.display()
        );
    }
    let probe = media::probe(input)?;
    if probe.audio_streams == 0 {
        bail!("{} has no audio stream", input.display());
    }
    if probe.has_video {
        if !media::VIDEO_EXTENSIONS.contains(&ext.as_str()) {
            bail!(
                "{}: video in .{ext} is not supported; use one of {}",
                input.display(),
                media::VIDEO_EXTENSIONS.join(", ")
            );
        }
        Ok(Kind::Video { ext, probe })
    } else {
        Ok(Kind::Audio)
    }
}

/// Resolve every output path up front, so conflicts are reported before
/// any time is spent processing.
fn plan_jobs(args: &Args, have_ffmpeg: bool) -> Result<Vec<Result<Job>>> {
    let out_is_dir = args.output.as_ref().is_some_and(|o| {
        o.is_dir()
            || o.as_os_str()
                .to_string_lossy()
                .ends_with(std::path::MAIN_SEPARATOR)
    });
    if args.inputs.len() > 1 && args.output.is_some() && !out_is_dir {
        bail!("with several inputs, -o must be a folder (end it with '/')");
    }
    if out_is_dir {
        let dir = args.output.as_ref().expect("checked");
        std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
    }
    let mut taken: Vec<PathBuf> = Vec::new();
    Ok(args
        .inputs
        .iter()
        .map(|input| {
            let kind = classify(input, have_ffmpeg)?;
            let ext = match &kind {
                Kind::Video { ext, .. } => ext.as_str(),
                Kind::Wav | Kind::Audio => "wav",
            };
            let output = match &args.output {
                Some(dir) if out_is_dir => media::enhanced_name(input, dir, ext),
                Some(file) => file.clone(),
                None => media::enhanced_name(input, input.parent().unwrap_or(Path::new("")), ext),
            };
            if same_file(input, &output) {
                bail!(
                    "{}: output would overwrite the input itself",
                    input.display()
                );
            }
            if output.exists() && !args.force {
                bail!(
                    "{} already exists (use --force to replace it)",
                    output.display()
                );
            }
            if taken.contains(&output) {
                bail!(
                    "{}: two inputs would both write {}",
                    input.display(),
                    output.display()
                );
            }
            taken.push(output.clone());
            Ok(Job {
                input: input.clone(),
                output,
                kind,
            })
        })
        .collect())
}

fn run(args: Args) -> Result<()> {
    let preset = effective_preset(&args)?;
    if args.print_config {
        print!("{}", toml::to_string_pretty(&preset)?);
        return Ok(());
    }
    if !(0.0..=24.0 * 60.0).contains(&args.max_minutes) {
        bail!("--max-minutes must be within 0..=1440");
    }
    let restorer = build_restorer(&args, &preset)?;
    let have_ffmpeg = media::available();
    let jobs = plan_jobs(&args, have_ffmpeg)?;

    let mut failed = 0;
    for (index, job) in jobs.into_iter().enumerate() {
        let result = job.and_then(|job| {
            process_job(&job, &args, &preset, restorer.as_deref(), have_ffmpeg)
                .with_context(|| format!("{}", job.input.display()))
        });
        if let Err(err) = result {
            eprintln!("error: {err:#}");
            failed += 1;
        }
        if index + 1 < args.inputs.len() {
            println!();
        }
    }
    if failed > 0 {
        bail!("{failed} of {} file(s) failed", args.inputs.len());
    }
    Ok(())
}

/// Load the input as 48 kHz mono, via ffmpeg when needed. The temp dir
/// (if any) is returned so it lives until processing is done, then
/// deletes itself.
fn load(
    job: &Job,
    limits: LoadLimits,
    have_ffmpeg: bool,
) -> Result<(dsp::io::LoadedAudio, Option<tempfile::TempDir>)> {
    let via_ffmpeg = |input: &Path| -> Result<(dsp::io::LoadedAudio, Option<tempfile::TempDir>)> {
        let dir = tempfile::tempdir().context("cannot create a temp folder")?;
        let wav = dir.path().join("decoded.wav");
        media::decode_to_wav(input, &wav)?;
        let loaded = dsp::io::load_wav(&wav, limits)?;
        Ok((loaded, Some(dir)))
    };
    match job.kind {
        Kind::Wav => match dsp::io::load_wav(&job.input, limits) {
            Ok(loaded) => Ok((loaded, None)),
            // e.g. ADPCM or 64-bit float WAV: hound can't, ffmpeg can.
            Err(AudioError::Unsupported { .. }) if have_ffmpeg => via_ffmpeg(&job.input),
            Err(e) => Err(e.into()),
        },
        Kind::Audio | Kind::Video { .. } => via_ffmpeg(&job.input),
    }
}

fn process_job(
    job: &Job,
    args: &Args,
    preset: &Preset,
    restorer: Option<&dyn Restorer>,
    have_ffmpeg: bool,
) -> Result<()> {
    let started = Instant::now();
    let fs = f64::from(SAMPLE_RATE);
    let mut preset = preset.clone();
    // A video's mono voice goes on both stereo channels, and YouTube
    // measures the stereo mix: +3 dB versus mono. Measure it that way so
    // the uploaded file lands on the target.
    if matches!(job.kind, Kind::Video { .. }) {
        preset.post.loudness.dual_mono = true;
        preset.post.limiter.ceiling_dbtp = preset.post.limiter.video_ceiling_dbtp;
    }
    let dump_dir = args.dump_stages.as_ref().map(|d| {
        if args.inputs.len() > 1 {
            d.join(job.input.file_stem().unwrap_or_default())
        } else {
            d.clone()
        }
    });
    let mut dumper = StageDumper::new(dump_dir.as_deref())?;

    // 1. load
    let limits = LoadLimits {
        max_seconds: args.max_minutes * 60.0,
    };
    let (loaded, _decoded_tmp) = load(job, limits, have_ffmpeg)?;
    let src = &loaded.source;
    if src.truncated {
        eprintln!(
            "warning: {} ends early (truncated data); using the readable part",
            job.input.display()
        );
    }
    if src.repaired_samples > 0 {
        eprintln!(
            "warning: replaced {} NaN/Inf/absurd samples with silence",
            src.repaired_samples
        );
    }
    let mut audio = loaded.samples;
    let input_measure = loudness::measure(&audio, SAMPLE_RATE, preset.post.loudness.dual_mono)?;
    dumper.dump("input", &audio)?;

    // 2. pre
    let pre_report = pre::process(&mut audio, fs, &preset.pre);
    dumper.dump("pre", &audio)?;

    // 3. restore (at the working level: denoisers are level-sensitive)
    let restore_note = match restorer {
        None => "none".to_owned(),
        Some(r) => {
            loudness::align(
                &mut audio,
                SAMPLE_RATE,
                preset.post.working_lufs,
                preset.post.loudness.max_gain_db,
            )?;
            let restored = r.restore(&audio)?;
            audio = restored.audio;
            restored.note
        }
    };
    dumper.dump("restore", &audio)?;

    // 4. post chain
    let post = chain::process(&mut audio, SAMPLE_RATE, &preset.post, |name, a| {
        dumper.dump(name, a)
    })?;

    // 5. write
    let mut notes = Vec::new();
    match &job.kind {
        Kind::Wav | Kind::Audio => {
            let format = match args.format {
                Format::Pcm24 => OutputFormat::Pcm24,
                Format::F32 => OutputFormat::Float32,
            };
            let clipped = dsp::io::write_wav(&job.output, &audio, format)?;
            if clipped > 0 {
                notes.push(format!(
                    "WARNING: {clipped} samples clipped when writing 24-bit PCM"
                ));
            }
        }
        Kind::Video { ext, probe } => {
            notes.push(write_video(job, ext, probe, &audio, args.force)?);
        }
    }

    print_summary(
        job,
        src,
        &input_measure,
        &pre_report,
        &restore_note,
        &post,
        &notes,
        started.elapsed(),
    );
    Ok(())
}

/// Encode the processed audio into a copy of the original video.
/// Written to a temp file in the destination folder, checked, then
/// renamed into place (never clobbering unless `force`).
fn write_video(job: &Job, ext: &str, probe: &Probe, audio: &[f32], force: bool) -> Result<String> {
    let tmp_dir = tempfile::tempdir().context("cannot create a temp folder")?;
    let wav = tmp_dir.path().join("processed.wav");
    dsp::io::write_wav(&wav, audio, OutputFormat::Float32)?;

    let parent = job
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let partial = tempfile::Builder::new()
        .prefix(".mic-enhance-")
        .suffix(&format!(".{ext}"))
        .tempfile_in(parent)
        .with_context(|| format!("cannot write in {}", parent.display()))?;
    media::mux_video(&job.input, &wav, partial.path(), ext, probe)?;
    let durations = media::verify_durations(probe, partial.path())?;
    // Temp files are created owner-only (0600); give the finished video
    // normal file permissions, like the WAV outputs.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(partial.path(), std::fs::Permissions::from_mode(0o644))
            .with_context(|| format!("cannot set permissions on {}", partial.path().display()))?;
    }
    if force {
        partial.persist(&job.output).map_err(|e| e.error)?;
    } else {
        partial
            .persist_noclobber(&job.output)
            .map_err(|e| e.error)
            .with_context(|| {
                format!(
                    "{} appeared meanwhile; not overwriting",
                    job.output.display()
                )
            })?;
    }
    let codec = if ext == "webm" { "Opus" } else { "AAC" };
    Ok(format!(
        "video copied untouched; audio {codec} 320 kbps 48 kHz stereo; {durations}"
    ))
}

#[allow(clippy::too_many_arguments)]
fn print_summary(
    job: &Job,
    src: &dsp::io::SourceInfo,
    input: &loudness::Measurement,
    pre: &pre::PreReport,
    restore: &str,
    post: &chain::PostReport,
    notes: &[String],
    elapsed: Duration,
) {
    let seconds = src.frames as f64 / f64::from(SAMPLE_RATE);
    println!(
        "mic-enhance  {} -> {}",
        job.input.display(),
        job.output.display()
    );
    println!(
        "  source     {} Hz, {} ch, {}-bit {}, {:.1} s",
        src.sample_rate,
        src.channels,
        src.bits_per_sample,
        if src.float { "float" } else { "PCM" },
        seconds
    );
    println!("  input      {input}");
    match pre.hum.fundamental_hz {
        Some(f0) => {
            let k: Vec<String> = pre
                .hum
                .lines
                .iter()
                .map(|l| format!("{}x", l.harmonic))
                .collect();
            println!("  hum        {f0:.2} Hz, notched harmonics {}", k.join(" "));
        }
        None => println!("  hum        none detected"),
    }
    println!("  restore    {restore}");
    match &post.tone {
        Some(ToneOutcome::Applied {
            correction_db,
            analysis,
            ..
        }) => println!(
            "  tone       {} (bandwidth {:.1} kHz, {:.0} s of speech measured)",
            dsp::tone::describe(correction_db),
            analysis.bandwidth_hz / 1000.0,
            analysis.speech_seconds
        ),
        Some(ToneOutcome::Skipped(why)) => println!("  tone       not corrected: {why}"),
        None => println!(
            "  tone       natural (no tone shaping; --tone-strength 0..1 corrects mic colouring)"
        ),
    }
    if let Some(e) = &post.expander {
        println!(
            "  expander   threshold {:.1} dBFS, {} open/close changes; {}",
            e.threshold_db, e.transitions, e.gain_reduction
        );
    }
    if let Some(c) = &post.compressor {
        println!(
            "  compressor {}; make-up {:+.1} dB",
            c.gain_reduction, c.makeup_db
        );
    }
    if let Some(d) = &post.deesser {
        println!("  de-esser   {}", d.gain_reduction);
    }
    match post.loudness.gain_db {
        Some(g) => println!(
            "  loudness   {g:+.1} dB in {} pass(es){}",
            post.loudness.iterations,
            if post.loudness.gain_capped {
                " (capped by max_gain_db: target not reached)"
            } else {
                ""
            }
        ),
        None => println!("  loudness   input is silent; no gain applied"),
    }
    if let Some(l) = &post.limiter {
        println!("  limiter    {}", l.gain_reduction);
    }
    if let Some(out) = &post.output {
        println!("  output     {out}");
    }
    for note in notes {
        println!("  note       {note}");
    }
    println!(
        "  time       {:.2} s ({:.0}x real time)",
        elapsed.as_secs_f64(),
        seconds / elapsed.as_secs_f64().max(1e-9)
    );
}
