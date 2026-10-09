//! End-to-end runs of the real `mic-enhance` binary.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_mic-enhance");
const SMOKE: [&str; 3] = [
    "clean.wav",
    "degraded_noise_hum_room.wav",
    "degraded_bluetooth.wav",
];

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A clip from `testdata/smoke/`, or `None` (with a skip message) when it
/// is absent: test recordings are kept out of the repository.
fn smoke(name: &str) -> Option<PathBuf> {
    let path = repo().join("testdata/smoke").join(name);
    if path.exists() {
        Some(path)
    } else {
        eprintln!(
            "skipping: testdata/smoke/{name} not present (test recordings are not in the repo)"
        );
        None
    }
}

/// A 6 s voice-like tone (220 Hz with a 3 Hz syllable rhythm, plus 1.1 kHz)
/// for tests that need some audio but not real speech.
fn voice_wav(path: &Path) {
    use std::f32::consts::PI;
    let samples: Vec<f32> = (0..6 * 48_000)
        .map(|n| {
            let t = n as f32 / 48_000.0;
            0.05 * (2.0 * PI * 220.0 * t).sin() * (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin())
                + 0.02 * (2.0 * PI * 1100.0 * t).sin()
        })
        .collect();
    write_wav(path, float_spec(48_000, 1), &samples);
}

fn run(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(BIN)
        .args(args)
        .current_dir(repo())
        .output()
        .unwrap()
}

fn run_str(args: &[&str]) -> Output {
    let os: Vec<&std::ffi::OsStr> = args.iter().map(std::ffi::OsStr::new).collect();
    run(&os)
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

/// Load the output, check format, return (integrated LUFS, true peak dBTP).
fn check_output(path: &Path) -> (f64, f64) {
    let mut reader = hound::WavReader::open(path).unwrap();
    let spec = reader.spec();
    assert_eq!(
        (spec.channels, spec.sample_rate, spec.bits_per_sample),
        (1, 48_000, 24)
    );
    let samples: Vec<f32> = reader
        .samples::<i32>()
        .map(|s| s.unwrap() as f32 / 8_388_608.0)
        .collect();
    let mut meter =
        ebur128::EbuR128::new(1, 48_000, ebur128::Mode::I | ebur128::Mode::TRUE_PEAK).unwrap();
    meter.add_frames_f32(&samples).unwrap();
    let lufs = meter.loudness_global().unwrap();
    let tp = 20.0 * meter.true_peak(0).unwrap().log10();
    (lufs, tp)
}

fn write_wav(path: &Path, spec: hound::WavSpec, samples: &[f32]) {
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for &s in samples {
        match spec.sample_format {
            hound::SampleFormat::Float => w.write_sample(s).unwrap(),
            hound::SampleFormat::Int => w.write_sample((s * 32767.0) as i16).unwrap(),
        }
    }
    w.finalize().unwrap();
}

fn float_spec(rate: u32, channels: u16) -> hound::WavSpec {
    hound::WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    }
}

#[test]
fn smoke_files_with_rnnoise_and_none() {
    let dir = tempfile::tempdir().unwrap();
    for name in SMOKE {
        let Some(input) = smoke(name) else {
            continue;
        };
        for restore in ["rnnoise", "none"] {
            let out = dir.path().join(format!("{restore}-{name}"));
            let o = run(&[
                input.as_os_str(),
                "-o".as_ref(),
                out.as_os_str(),
                "--restore".as_ref(),
                restore.as_ref(),
            ]);
            assert!(o.status.success(), "{name} {restore}: {}", text(&o));
            let (lufs, tp) = check_output(&out);
            eprintln!("{name:30} --restore {restore:8} -> {lufs:.2} LUFS, {tp:.2} dBTP");
            assert!((lufs + 14.0).abs() <= 0.3, "{name}: {lufs}");
            assert!(tp <= -1.0, "{name}: {tp}");
        }
    }
}

#[test]
fn podcast_preset_and_dump_stages() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("out.wav");
    let stages = dir.path().join("stages");
    let Some(input) = smoke("clean.wav") else {
        return;
    };
    let o = run(&[
        input.as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
        "--preset".as_ref(),
        "podcast".as_ref(),
        "--restore".as_ref(),
        "none".as_ref(),
        "--dump-stages".as_ref(),
        stages.as_os_str(),
    ]);
    assert!(o.status.success(), "{}", text(&o));
    let (lufs, _) = check_output(&out);
    assert!((lufs + 16.0).abs() <= 0.3, "{lufs}");
    let mut names: Vec<String> = std::fs::read_dir(&stages)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names.len(), 13, "{names:?}");
    assert_eq!(names.first().unwrap(), "00_input.wav");
    assert_eq!(names.last().unwrap(), "12_limiter.wav");
}

/// ML stage through `--mlstage-cmd` with a fake that copies input to output.
#[cfg(unix)]
#[test]
fn mlstage_contract_with_fake_and_fallback() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let make = |name: &str, body: &str| {
        let p = dir.path().join(name);
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    };
    let ok = make(
        "ok.sh",
        r#"cp "$4" "$6"; echo '{"model":"deepfilternet3","seconds":0.5,"rtf":0.03}'"#,
    );
    let broken = make("broken.sh", "echo 'CUDA exploded' >&2; exit 1");
    let input = dir.path().join("voice.wav");
    voice_wav(&input);
    let out = dir.path().join("out.wav");

    let o = run(&[
        input.as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
        "--mlstage-cmd".as_ref(),
        ok.as_os_str(),
    ]);
    assert!(
        o.status.success() && text(&o).contains("deepfilternet3 via mlstage"),
        "{}",
        text(&o)
    );

    // Clean mode falls back to the built-in rnnoise.
    let o = run(&[
        input.as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
        "--force".as_ref(),
        "--mlstage-cmd".as_ref(),
        broken.as_os_str(),
    ]);
    assert!(o.status.success(), "{}", text(&o));
    assert!(
        text(&o).contains("using built-in RNNoise; deepfilternet3 unavailable because")
            && text(&o).contains("CUDA exploded"),
        "{}",
        text(&o)
    );

    // Studio mode is an explicit opt-in: failure is an error, not a fallback.
    let o = run(&[
        input.as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
        "--force".as_ref(),
        "--mode".as_ref(),
        "studio".as_ref(),
        "--mlstage-cmd".as_ref(),
        broken.as_os_str(),
    ]);
    assert!(!o.status.success(), "{}", text(&o));
}

#[test]
fn converts_stereo_44k1_16bit_input() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("stereo.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let samples: Vec<f32> = (0..44_100 * 4 * 2)
        .map(|i| 0.2 * ((i / 2) as f32 * 2.0 * std::f32::consts::PI * 220.0 / 44_100.0).sin())
        .collect();
    write_wav(&input, spec, &samples);
    let out = dir.path().join("out.wav");
    let o = run(&[
        input.as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
        "--restore".as_ref(),
        "none".as_ref(),
    ]);
    assert!(o.status.success(), "{}", text(&o));
    let reader = hound::WavReader::open(&out).unwrap();
    assert_eq!(reader.duration(), 48_000 * 4);
}

/// Hostile / broken inputs: every one must fail with a clear message and
/// exit code 1, never a panic (exit 101) or a hang.
#[test]
fn hostile_inputs_fail_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let out = d.join("out.wav");

    let empty = d.join("empty.wav");
    write_wav(&empty, float_spec(48_000, 1), &[]);

    let nan = d.join("nan.wav");
    write_wav(&nan, float_spec(48_000, 1), &vec![f32::NAN; 48_000]);

    let huge = d.join("huge.wav");
    write_wav(&huge, float_spec(48_000, 1), &vec![1e30; 48_000]);

    let bad_rate = d.join("rate.wav");
    write_wav(&bad_rate, float_spec(1_000, 1), &vec![0.1; 1000]);

    let not_wav = d.join("text.wav");
    std::fs::write(&not_wav, "this is not audio").unwrap();

    let long = d.join("long.wav");
    write_wav(&long, float_spec(48_000, 1), &vec![0.0; 48_000 * 7]);

    // Header claims far more data than the file holds (truncated): the
    // readable part is kept with a warning.
    let truncated = d.join("truncated.wav");
    write_wav(&truncated, float_spec(48_000, 1), &vec![0.1; 48_000]);
    let bytes = std::fs::read(&truncated).unwrap();
    std::fs::write(&truncated, &bytes[..bytes.len() / 2]).unwrap();

    let cases: [(&Path, &[&str], &str); 7] = [
        (&empty, &[], "no audio samples"),
        (&nan, &[], "corrupt"),
        (&huge, &[], "corrupt"),
        (&bad_rate, &[], "sample rate 1000 Hz"),
        (&not_wav, &[], "cannot open"),
        (&long, &["--max-minutes", "0.1"], "exceeds"),
        (Path::new("/definitely/missing.wav"), &[], "cannot open"),
    ];
    for (input, extra, expected) in cases {
        let mut args = vec![
            input.as_os_str(),
            "-o".as_ref(),
            out.as_os_str(),
            "--restore".as_ref(),
            "none".as_ref(),
        ];
        args.extend(extra.iter().map(std::ffi::OsStr::new));
        let o = run(&args);
        assert_eq!(
            o.status.code(),
            Some(1),
            "{}: {}",
            input.display(),
            text(&o)
        );
        assert!(
            text(&o).contains(expected),
            "{}: expected `{expected}` in: {}",
            input.display(),
            text(&o)
        );
    }

    let o = run(&[
        truncated.as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
        "--restore".as_ref(),
        "none".as_ref(),
    ]);
    assert!(
        o.status.success() && text(&o).contains("truncated"),
        "{}",
        text(&o)
    );

    // Refuses to overwrite its own input.
    let o = run(&[long.as_os_str(), "-o".as_ref(), long.as_os_str()]);
    assert!(!o.status.success() && text(&o).contains("overwrite the input"));
}

#[test]
fn bad_config_is_rejected_with_field_name() {
    let o = run_str(&["x.wav", "-o", "y.wav", "--set", "post.compressor.ratio=0.5"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(text(&o).contains("post.compressor.ratio"), "{}", text(&o));

    let o = run_str(&["x.wav", "-o", "y.wav", "--restore", "--evil"]);
    assert!(!o.status.success());

    let o = run_str(&[
        "x.wav",
        "-o",
        "y.wav",
        "--print-config",
        "--target-lufs",
        "-16",
    ]);
    assert!(o.status.success());
    assert!(text(&o).contains("target_lufs = -16.0"), "{}", text(&o));
}

/// Installed-binary situation: run from an unrelated folder. The ML
/// project must still be found (compile-time path), and when it can't be
/// (bad $MIC_ENHANCE_ML) the fallback must be announced, never silent.
#[test]
fn ml_project_is_found_from_any_folder() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("voice.wav");
    voice_wav(&input);

    // 1. Broken location: clear warning naming the fallback and the reason.
    let out = dir.path().join("a.wav");
    let o = Command::new(BIN)
        .arg(&input)
        .arg("-o")
        .arg(&out)
        .env("MIC_ENHANCE_ML", "/nonexistent/ml")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", text(&o));
    assert!(
        text(&o).contains("using built-in RNNoise; deepfilternet3 unavailable because the ML project was not found at /nonexistent/ml"),
        "{}",
        text(&o)
    );

    // 2. Default location from a foreign cwd: either mlstage really ran,
    //    or (no uv / models on this machine) the fallback is announced.
    let out = dir.path().join("b.wav");
    let o = Command::new(BIN)
        .arg(&input)
        .arg("-o")
        .arg(&out)
        .env_remove("MIC_ENHANCE_ML")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", text(&o));
    let ran = text(&o).contains("deepfilternet3 via mlstage");
    let announced = text(&o).contains("using built-in RNNoise; deepfilternet3 unavailable because");
    eprintln!("from a temp cwd: mlstage ran = {ran}, fallback announced = {announced}");
    assert!(ran || announced, "{}", text(&o));
    assert!(
        !text(&o).contains("was not found at"),
        "compile-time ml/ path not used: {}",
        text(&o)
    );
}
