//! The pluggable restore stage (denoise / generative restoration).
//!
//! Why a trait and not an enum: the CLI needs three interchangeable
//! implementations *plus* a combinator (`WithFallback`) that wraps any two
//! of them, and tests need to plug in a fake ML stage. With a trait each
//! piece is a small independent type; an enum would need a match arm in
//! every method for every variant and could not wrap itself as cleanly.
//! The cost is dynamic dispatch (`Box<dyn Restorer>`), which is one call
//! per file here, i.e. free.

use anyhow::{Context, Result, bail, ensure};
use dsp::io::{LoadLimits, OutputFormat, load_wav, write_wav};
use serde::Deserialize;
use std::ffi::OsString;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Output of a restore stage.
pub struct Restored {
    pub audio: Vec<f32>,
    /// Human-readable description for the summary (model, speed, fallback).
    pub note: String,
}

pub trait Restorer {
    /// Short name for messages, e.g. `rnnoise` or `deepfilternet3 (mlstage)`.
    fn name(&self) -> String;
    /// Process 48 kHz mono audio; output must have the same length.
    fn restore(&self, input: &[f32]) -> Result<Restored>;
}

/// Built-in pure-Rust RNNoise. Always available, no Python needed.
pub struct Rnnoise;

impl Restorer for Rnnoise {
    fn name(&self) -> String {
        "built-in RNNoise".into()
    }

    fn restore(&self, input: &[f32]) -> Result<Restored> {
        let (audio, report) = dsp::rnnoise::denoise(input);
        Ok(Restored {
            audio,
            note: format!(
                "built-in RNNoise, mean voice probability {:.2}",
                report.mean_vad
            ),
        })
    }
}

/// Runs the Python `mlstage` tool (in `ml/`) as a subprocess:
/// `<program> <prefix args> --model M --in a.wav --out b.wav`.
pub struct MlStage {
    program: OsString,
    prefix: Vec<OsString>,
    model: String,
    timeout: Duration,
}

/// The one JSON line mlstage prints on stdout.
#[derive(Debug, Deserialize)]
struct MlReport {
    model: String,
    seconds: f64,
    rtf: f64,
}

/// Captured output is capped so a misbehaving child cannot fill memory.
const MAX_CAPTURE_BYTES: u64 = 1 << 20;
/// ML output may differ in length by this much (model padding); it is
/// trimmed/padded back. Anything larger means misalignment: error.
const MAX_LENGTH_DRIFT: usize = 4800; // 100 ms

impl MlStage {
    /// The default command: `uv run --project <ml_project> mlstage`.
    pub fn via_uv(ml_project: PathBuf, model: &str, timeout: Duration) -> Result<Self> {
        Self::with_command(
            "uv".into(),
            vec![
                "run".into(),
                "--project".into(),
                ml_project.into(),
                "mlstage".into(),
            ],
            model,
            timeout,
        )
    }

    /// Any executable that implements the mlstage contract (used by tests
    /// and by `--mlstage-cmd`).
    pub fn with_command(
        program: OsString,
        prefix: Vec<OsString>,
        model: &str,
        timeout: Duration,
    ) -> Result<Self> {
        validate_model_name(model)?;
        Ok(Self {
            program,
            prefix,
            model: model.to_owned(),
            timeout,
        })
    }
}

/// Model names go on a command line, so restrict them to a safe alphabet
/// and forbid a leading `-` (which the child would parse as an option).
/// No shell is involved (`Command` passes argv directly), so this is
/// about option injection, not shell injection.
fn validate_model_name(model: &str) -> Result<()> {
    let ok = !model.is_empty()
        && model.len() <= 64
        && !model.starts_with('-')
        && model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c));
    ensure!(
        ok,
        "invalid model name `{model}`: use letters, digits, '.', '_' or '-' (max 64)"
    );
    Ok(())
}

/// Read a child pipe on its own thread (so a chatty child cannot block on
/// a full pipe while we wait for it), keeping at most `MAX_CAPTURE_BYTES`.
fn capture(pipe: impl Read + Send + 'static) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let mut limited = pipe.take(MAX_CAPTURE_BYTES);
        let _ = limited.read_to_end(&mut buf);
        // Drain (and drop) anything beyond the cap so the child never blocks.
        let _ = std::io::copy(&mut limited.into_inner(), &mut std::io::sink());
        String::from_utf8_lossy(&buf).into_owned()
    })
}

/// Kill the child and everything it started (its process group). Uses the
/// `kill` command because std has no killpg and this crate forbids
/// `unsafe` (so no direct libc call).
fn kill_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    let _ = Command::new("kill")
        .args(["-KILL", &format!("-{}", child.id())])
        .stderr(Stdio::null())
        .status();
    let _ = child.kill();
    let _ = child.wait();
}

fn tail(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

impl Restorer for MlStage {
    fn name(&self) -> String {
        self.model.clone()
    }

    fn restore(&self, input: &[f32]) -> Result<Restored> {
        let dir = tempfile::tempdir().context("cannot create temp dir for mlstage")?;
        let in_path = dir.path().join("in.wav");
        let out_path = dir.path().join("out.wav");
        write_wav(&in_path, input, OutputFormat::Float32)?;

        let started = Instant::now();
        let mut command = Command::new(&self.program);
        // Own process group, so a timeout can kill the whole tree:
        // `uv run` starts Python as a grandchild, which a plain
        // `child.kill()` would leave running as an orphan.
        #[cfg(unix)]
        std::os::unix::process::CommandExt::process_group(&mut command, 0);
        let mut child = command
            .args(&self.prefix)
            .arg("--model")
            .arg(&self.model)
            .arg("--in")
            .arg(&in_path)
            .arg("--out")
            .arg(&out_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("cannot start `{}`", self.program.to_string_lossy()))?;
        let stdout = capture(child.stdout.take().expect("stdout is piped"));
        let stderr = capture(child.stderr.take().expect("stderr is piped"));

        // std has no wait-with-timeout, so poll.
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if started.elapsed() > self.timeout {
                kill_tree(&mut child);
                bail!("mlstage timed out after {:?}", self.timeout);
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        let stdout = stdout.join().unwrap_or_default();
        let stderr = stderr.join().unwrap_or_default();
        if !status.success() {
            bail!("mlstage failed ({status}):\n{}", tail(&stderr, 15));
        }

        let json_line = stdout
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        let report: MlReport = serde_json::from_str(json_line).with_context(|| {
            format!("mlstage printed no valid JSON report line (got `{json_line}`)")
        })?;

        // Same hostile-input guards as user files: format, NaN, length.
        let limits = LoadLimits {
            max_seconds: input.len() as f64 / f64::from(dsp::SAMPLE_RATE) + 10.0,
        };
        let mut audio = load_wav(&out_path, limits)
            .context("mlstage output is unusable")?
            .samples;
        let drift = audio.len().abs_diff(input.len());
        ensure!(
            drift <= MAX_LENGTH_DRIFT,
            "mlstage output is {} samples long, input was {} (misaligned)",
            audio.len(),
            input.len()
        );
        audio.resize(input.len(), 0.0);

        Ok(Restored {
            audio,
            note: format!(
                "{} via mlstage, {:.1} s (real-time factor {:.2})",
                report.model, report.seconds, report.rtf
            ),
        })
    }
}

/// Try `primary`; if it fails for any reason, warn and use `fallback`.
pub struct WithFallback {
    pub primary: Box<dyn Restorer>,
    pub fallback: Box<dyn Restorer>,
}

impl Restorer for WithFallback {
    fn name(&self) -> String {
        format!(
            "{} (fallback: {})",
            self.primary.name(),
            self.fallback.name()
        )
    }

    fn restore(&self, input: &[f32]) -> Result<Restored> {
        match self.primary.restore(input) {
            Ok(restored) => Ok(restored),
            Err(err) => {
                eprintln!(
                    "warning: using {}; {} unavailable because {err:#}",
                    self.fallback.name(),
                    self.primary.name()
                );
                let mut restored = self.fallback.restore(input)?;
                restored.note = format!(
                    "{} [fallback after: {}]",
                    restored.note,
                    first_line(&format!("{err:#}"))
                );
                Ok(restored)
            }
        }
    }
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// Write an executable shell script that fakes mlstage.
    fn fake_mlstage(dir: &std::path::Path, body: &str) -> OsString {
        // One file per script body: never rewrite a script that may still run.
        let id = body
            .bytes()
            .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
        let path = dir.join(format!("fake-mlstage-{id:x}.sh"));
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.into_os_string()
    }

    /// Copies --in to --out (args: --model M --in A --out B) and reports.
    const COPY_OK: &str =
        r#"cp "$4" "$6"; echo 'some log line'; echo '{"model":"fake","seconds":0.1,"rtf":0.01}'"#;

    fn stage(script: OsString) -> MlStage {
        MlStage::with_command(script, vec![], "fake", Duration::from_secs(10)).unwrap()
    }

    fn signal() -> Vec<f32> {
        (0..48_000).map(|n| (n as f32 * 0.01).sin() * 0.1).collect()
    }

    #[test]
    fn successful_run_round_trips_audio() {
        let dir = tempfile::tempdir().unwrap();
        let input = signal();
        let out = stage(fake_mlstage(dir.path(), COPY_OK))
            .restore(&input)
            .unwrap();
        assert_eq!(out.audio, input);
        assert!(out.note.contains("fake"), "{}", out.note);
    }

    #[test]
    fn failures_are_errors_not_panics() {
        let dir = tempfile::tempdir().unwrap();
        let cases = [
            ("echo 'boom' >&2; exit 1", "boom"),
            (r#"cp "$4" "$6"; echo 'not json'"#, "JSON"),
            (
                r#"echo '{"model":"fake","seconds":0.1,"rtf":0.01}'"#,
                "unusable",
            ),
            ("sleep 30 & wait", "timed out"),
        ];
        for (body, expected) in cases {
            let mut s = stage(fake_mlstage(dir.path(), body));
            if body.starts_with("sleep") {
                s.timeout = Duration::from_millis(500);
            }
            let err = format!("{:#}", s.restore(&signal()).err().expect(body));
            assert!(err.contains(expected), "`{body}` gave: {err}");
        }
        let missing = stage("/nonexistent/mlstage".into());
        assert!(missing.restore(&signal()).is_err());
    }

    /// `uv run` puts Python one level down; a timeout must kill that
    /// grandchild too, not leave it running as an orphan.
    #[test]
    fn timeout_kills_grandchildren() {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("grandchild.pid");
        let body = format!("sleep 30 & echo $! > '{}'; wait", pidfile.display());
        let mut s = stage(fake_mlstage(dir.path(), &body));
        s.timeout = Duration::from_secs(2); // let the shell start and write its pid file
        assert!(s.restore(&signal()).is_err());
        let pid = std::fs::read_to_string(&pidfile).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        let alive = Command::new("kill")
            .args(["-0", pid.trim()])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success();
        assert!(!alive, "grandchild {pid} survived the timeout");
    }

    #[test]
    fn fallback_kicks_in_on_failure() {
        let dir = tempfile::tempdir().unwrap();
        let restorer = WithFallback {
            primary: Box::new(stage(fake_mlstage(dir.path(), "exit 3"))),
            fallback: Box::new(Rnnoise),
        };
        let out = restorer.restore(&signal()).unwrap();
        assert_eq!(out.audio.len(), 48_000);
        assert!(out.note.contains("fallback"), "{}", out.note);
    }

    #[test]
    fn model_names_cannot_inject_options() {
        for bad in ["", "--rm", "-x", "a b", "a;b", "../x/y", &"x".repeat(65)] {
            assert!(
                MlStage::via_uv("ml".into(), bad, Duration::from_secs(1)).is_err(),
                "{bad}"
            );
        }
        assert!(MlStage::via_uv("ml".into(), "deepfilternet3", Duration::from_secs(1)).is_ok());
    }
}
