//! ffmpeg-backed inputs: video in -> same video, new sound; m4a in -> WAV.
//! Skipped (with a message) when ffmpeg isn't installed.

use std::path::Path;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_mic-enhance");

fn have_ffmpeg() -> bool {
    let ok = |t: &str| {
        Command::new(t)
            .arg("-version")
            .output()
            .is_ok_and(|o| o.status.success())
    };
    let found = ok("ffmpeg") && ok("ffprobe");
    if !found {
        eprintln!("skipping: ffmpeg/ffprobe not on PATH");
    }
    found
}

fn ffmpeg(args: &[&str]) -> Output {
    let o = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error"])
        .args(args)
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "ffmpeg {args:?}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    o
}

fn enhance(args: &[&str], cwd: &Path) -> Output {
    Command::new(BIN)
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap()
}

fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

/// MD5 of the video stream's packets (no decoding: proves bit-identity).
fn video_md5(path: &Path) -> String {
    let o = ffmpeg(&[
        "-i",
        path.to_str().unwrap(),
        "-map",
        "0:v",
        "-c",
        "copy",
        "-f",
        "md5",
        "-",
    ]);
    String::from_utf8_lossy(&o.stdout).trim().to_owned()
}

/// Integrated loudness of the audio as delivered (all channels), via a
/// decode to float WAV and the ebur128 crate.
fn delivered_lufs(path: &Path, dir: &Path) -> (f64, u16) {
    let wav = dir.join(format!(
        "measure-{}.wav",
        path.file_name().unwrap().to_string_lossy()
    ));
    ffmpeg(&[
        "-y",
        "-i",
        path.to_str().unwrap(),
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_f32le",
        wav.to_str().unwrap(),
    ]);
    let mut r = hound::WavReader::open(&wav).unwrap();
    let ch = r.spec().channels;
    let samples: Vec<f32> = r.samples::<f32>().map(Result::unwrap).collect();
    let mut m = ebur128::EbuR128::new(
        u32::from(ch),
        r.spec().sample_rate,
        ebur128::Mode::I | ebur128::Mode::TRUE_PEAK,
    )
    .unwrap();
    m.add_frames_f32(&samples).unwrap();
    let tp = (0..u32::from(ch))
        .map(|c| m.true_peak(c).unwrap())
        .fold(0.0, f64::max);
    eprintln!("delivered true peak {:+.2} dBTP", 20.0 * tp.log10());
    (m.loudness_global().unwrap(), ch)
}

fn duration(path: &Path) -> f64 {
    let o = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()
        .unwrap();
    String::from_utf8_lossy(&o.stdout).trim().parse().unwrap()
}

/// A short "talk": 1 kHz-ish voice-like tone bursts, at a quiet level.
const VOICE: &str =
    "aevalsrc=0.05*sin(2*PI*220*t)*(0.6+0.4*sin(2*PI*3*t))+0.02*sin(2*PI*1100*t):s=48000:d=6";

/// The same signal as `VOICE`, written as a 48 kHz float WAV without ffmpeg.
fn voice_wav(path: &Path) {
    use std::f32::consts::PI;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for n in 0..6 * 48_000 {
        let t = n as f32 / 48_000.0;
        let s = 0.05 * (2.0 * PI * 220.0 * t).sin() * (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin())
            + 0.02 * (2.0 * PI * 1100.0 * t).sin();
        w.write_sample(s).unwrap();
    }
    w.finalize().unwrap();
}

#[test]
fn video_keeps_picture_and_gets_new_sound() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    // Name starts with '-' on purpose: must not be read as an option.
    let input = d.join("-clip.mp4");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x120:rate=30:duration=6",
        "-f",
        "lavfi",
        "-i",
        VOICE,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        input.to_str().unwrap(),
    ]);

    let o = enhance(&["--restore", "none", "./-clip.mp4"], d);
    assert!(o.status.success(), "{}", text(&o));
    let output = d.join("-clip.enhanced.mp4");
    assert!(output.exists(), "{}", text(&o));
    eprintln!("{}", text(&o));

    assert_eq!(
        video_md5(&input),
        video_md5(&output),
        "video stream changed"
    );
    let (lufs, channels) = delivered_lufs(&output, d);
    eprintln!("video output: {lufs:.2} LUFS over {channels} channels");
    assert_eq!(channels, 2);
    assert!((lufs + 14.0).abs() <= 0.5, "{lufs}");
    let (a, b) = (duration(&input), duration(&output));
    assert!((a - b).abs() <= 1.0 / 30.0, "duration {a} -> {b}");

    // Second run must refuse to overwrite; --force allows it.
    let again = enhance(&["--restore", "none", "./-clip.mp4"], d);
    assert!(
        !again.status.success() && text(&again).contains("already exists"),
        "{}",
        text(&again)
    );
    let forced = enhance(&["--restore", "none", "--force", "./-clip.mp4"], d);
    assert!(forced.status.success(), "{}", text(&forced));
    // No temp files left behind next to the output.
    let leftovers: Vec<_> = std::fs::read_dir(d)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with(".mic-enhance-"))
        .collect();
    assert!(leftovers.is_empty());
}

#[test]
fn m4a_becomes_24_bit_wav() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let input = d.join("memo.m4a");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        VOICE,
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        input.to_str().unwrap(),
    ]);
    let out_dir = d.join("out");
    let o = enhance(&["--restore", "none", "memo.m4a", "-o", "out/"], d);
    assert!(o.status.success(), "{}", text(&o));
    let output = out_dir.join("memo.enhanced.wav");
    let spec = hound::WavReader::open(&output).unwrap().spec();
    assert_eq!(
        (spec.channels, spec.sample_rate, spec.bits_per_sample),
        (1, 48_000, 24)
    );
    let (lufs, _) = delivered_lufs(&output, d);
    eprintln!("m4a output: {lufs:.2} LUFS");
    assert!((lufs + 14.0).abs() <= 0.5, "{lufs}");
}

#[test]
fn several_inputs_and_bad_output_combinations() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    for name in ["clean.wav", "degraded_bluetooth.wav"] {
        voice_wav(&d.join(name));
    }
    // Two inputs, -o naming a file: refused before any work.
    let o = enhance(&["clean.wav", "degraded_bluetooth.wav", "-o", "x.wav"], d);
    assert!(
        !o.status.success() && text(&o).contains("folder"),
        "{}",
        text(&o)
    );
    // Default outputs next to the inputs.
    let o = enhance(
        &["--restore", "none", "clean.wav", "degraded_bluetooth.wav"],
        d,
    );
    assert!(o.status.success(), "{}", text(&o));
    assert!(
        d.join("clean.enhanced.wav").exists() && d.join("degraded_bluetooth.enhanced.wav").exists()
    );
    // Unknown format without being a real media file: clear error, exit 1.
    std::fs::write(d.join("notes.txt"), "hello").unwrap();
    let o = enhance(&["notes.txt"], d);
    assert_eq!(o.status.code(), Some(1), "{}", text(&o));
}

/// Hostile "audio" files whose *content* is a playlist/script that points
/// at another local file. ffmpeg chooses the demuxer by content, so only
/// the format whitelist stops these. Both must be rejected cleanly, with
/// no output written.
#[test]
fn playlist_and_concat_disguised_as_audio_are_rejected() {
    if !have_ffmpeg() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    voice_wav(&d.join("secret.wav"));
    let secret_abs = d.join("secret.wav");

    std::fs::write(
        d.join("evil.m4a"),
        format!(
            "#EXTM3U\n#EXT-X-TARGETDURATION:20\n#EXTINF:16.0,\n{}\n#EXT-X-ENDLIST\n",
            secret_abs.display()
        ),
    )
    .unwrap();
    // Relative name: allowed by concat's own "safe" mode, so without our
    // whitelist even ffmpeg 8 would read secret.wav through this file.
    std::fs::write(
        d.join("evil.mp3"),
        "ffconcat version 1.0\nfile 'secret.wav'\n",
    )
    .unwrap();
    let unprotected = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=format_name",
            "-of",
            "csv=p=0",
            "evil.mp3",
        ])
        .current_dir(d)
        .output()
        .unwrap();
    eprintln!(
        "plain ffprobe on evil.mp3 (no whitelist): {:?}",
        String::from_utf8_lossy(&unprotected.stdout).trim()
    );

    for name in ["evil.m4a", "evil.mp3"] {
        let o = enhance(&["--restore", "none", name], d);
        eprintln!("{name}: {}", text(&o).trim());
        assert_eq!(o.status.code(), Some(1), "{name}: {}", text(&o));
        assert!(
            text(&o).contains("allowed container formats"),
            "{name}: {}",
            text(&o)
        );
        let stem = name.trim_end_matches(".m4a").trim_end_matches(".mp3");
        assert!(!d.join(format!("{stem}.enhanced.wav")).exists());
    }
}
