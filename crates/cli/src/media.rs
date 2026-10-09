//! Everything that talks to ffmpeg / ffprobe: decoding any audio or video
//! file to a 48 kHz WAV, and putting processed audio back into a video.
//!
//! Safety rules for the command lines built here:
//! * `std::process::Command`, never a shell: arguments are passed as-is,
//!   so spaces, quotes or `$(...)` in file names mean nothing.
//! * Every path goes through [`path_arg`]: relative paths get a `./`
//!   prefix, so a file called `-y` or `-i` can't be read as an option and
//!   a file called `http:evil` or `concat:a|b` can't be read as an ffmpeg
//!   protocol URL. Absolute paths start with `/` and are safe already.
//! * Every *input* is opened with `-protocol_whitelist file` and a
//!   `-format_whitelist` of the real container demuxers we support.
//!   ffmpeg picks the demuxer from the file's *content*, not its name, so
//!   without this a hostile `clip.m4a` that is really an HLS playlist or an
//!   ffconcat script could make ffmpeg fetch URLs (SSRF) or read other
//!   local files into our output (the known playlist file-disclosure
//!   attack). Playlist/script demuxers (hls, concat, ...) are not listed.
//! * `-nostdin`: ffmpeg must never wait for keyboard input (it otherwise
//!   reads `q` from stdin and can hang a batch run).
//! * Temporary files live in `tempfile` directories/files that delete
//!   themselves when dropped, on success and on every error path.

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// Containers we treat as video (output keeps the same container).
pub const VIDEO_EXTENSIONS: [&str; 5] = ["mp4", "mov", "m4v", "mkv", "webm"];

/// `./` prefix for relative paths (see module docs).
pub fn path_arg(path: &Path) -> OsString {
    if path.is_absolute() {
        path.as_os_str().to_owned()
    } else {
        Path::new(".").join(path).into_os_string()
    }
}

/// Demuxers allowed to open user files (names from `ffmpeg -formats`).
/// Deliberately absent: hls, concat, dash, image2, lavfi, any network or
/// "virtual" demuxer that follows references to other files.
pub const FORMAT_WHITELIST: &str =
    "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,mp3,aac,flac,ogg,wav,w64,aiff,caf";

/// `-protocol_whitelist file -format_whitelist ... -i <path>`: the only way
/// this module opens an input.
fn input_args(path: &Path) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        FORMAT_WHITELIST,
        "-i",
    ]
    .map(OsString::from)
    .to_vec();
    args.push(path_arg(path));
    args
}

fn refs(args: &[OsString]) -> Vec<&OsStr> {
    args.iter().map(OsString::as_os_str).collect()
}

/// True if `ffmpeg` and `ffprobe` can be started.
pub fn available() -> bool {
    ["ffmpeg", "ffprobe"].iter().all(|tool| {
        Command::new(tool)
            .arg("-version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

/// Run a tool, turning failure into an error with the end of its stderr.
fn run(tool: &str, args: &[&OsStr]) -> Result<Output> {
    let output = Command::new(tool)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .with_context(|| format!("cannot start `{tool}` (is ffmpeg installed and on PATH?)"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(6).collect();
        bail!(
            "{tool} failed ({}): {}",
            output.status,
            tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
        );
    }
    Ok(output)
}

#[derive(Debug, Deserialize)]
struct ProbeJson {
    #[serde(default)]
    streams: Vec<ProbeStream>,
    format: Option<ProbeFormat>,
}

#[derive(Debug, Deserialize)]
struct ProbeStream {
    codec_type: Option<String>,
    start_time: Option<String>,
    duration: Option<String>,
    avg_frame_rate: Option<String>,
    #[serde(default)]
    disposition: Option<ProbeDisposition>,
}

#[derive(Debug, Deserialize)]
struct ProbeDisposition {
    #[serde(default)]
    attached_pic: u8,
}

#[derive(Debug, Deserialize)]
struct ProbeFormat {
    duration: Option<String>,
}

/// What ffprobe says about a media file.
#[derive(Debug, Clone, PartialEq)]
pub struct Probe {
    /// A real video stream (cover art in an m4a/mp3 does not count).
    pub has_video: bool,
    pub audio_streams: usize,
    /// Start time of the first audio stream relative to the first video
    /// stream (seconds), so the new audio keeps the original sync.
    pub audio_offset_s: f64,
    pub duration_s: Option<f64>,
    pub audio_duration_s: Option<f64>,
    /// One video frame, seconds (from the average frame rate).
    pub frame_s: Option<f64>,
}

fn parse_f64(s: &Option<String>) -> Option<f64> {
    s.as_deref()
        .and_then(|v| v.parse().ok())
        .filter(|v: &f64| v.is_finite())
}

fn parse_rate(s: &Option<String>) -> Option<f64> {
    let (n, d) = s.as_deref()?.split_once('/')?;
    let (n, d): (f64, f64) = (n.parse().ok()?, d.parse().ok()?);
    (n > 0.0 && d > 0.0).then(|| d / n)
}

pub fn probe(path: &Path) -> Result<Probe> {
    let mut args: Vec<OsString> = [
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "stream=codec_type,start_time,duration,avg_frame_rate:stream_disposition=attached_pic:format=duration",
    ]
    .map(OsString::from)
    .to_vec();
    args.extend(input_args(path));
    let out = run("ffprobe", &refs(&args)).with_context(|| {
        format!(
            "cannot read {} as audio/video (allowed container formats: {FORMAT_WHITELIST})",
            path.display()
        )
    })?;
    let json: ProbeJson =
        serde_json::from_slice(&out.stdout).context("unexpected ffprobe output")?;
    let video = json.streams.iter().find(|s| {
        s.codec_type.as_deref() == Some("video")
            && s.disposition.as_ref().is_none_or(|d| d.attached_pic == 0)
    });
    let audio: Vec<&ProbeStream> = json
        .streams
        .iter()
        .filter(|s| s.codec_type.as_deref() == Some("audio"))
        .collect();
    let audio_start = audio
        .first()
        .and_then(|a| parse_f64(&a.start_time))
        .unwrap_or(0.0);
    let video_start = video
        .and_then(|v| parse_f64(&v.start_time))
        .unwrap_or(audio_start);
    Ok(Probe {
        has_video: video.is_some(),
        audio_streams: audio.len(),
        audio_offset_s: audio_start - video_start,
        duration_s: json.format.as_ref().and_then(|f| parse_f64(&f.duration)),
        audio_duration_s: audio.first().and_then(|a| parse_f64(&a.duration)),
        frame_s: video.and_then(|v| parse_rate(&v.avg_frame_rate)),
    })
}

/// Decode the first audio stream to a 48 kHz mono 32-bit float WAV.
pub fn decode_to_wav(input: &Path, wav: &Path) -> Result<()> {
    let mut args: Vec<OsString> = ["-nostdin", "-hide_banner", "-loglevel", "error"]
        .map(OsString::from)
        .to_vec();
    args.extend(input_args(input));
    args.extend(
        [
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
        ]
        .map(OsString::from),
    );
    args.push(path_arg(wav));
    run("ffmpeg", &refs(&args))
        .with_context(|| format!("ffmpeg could not decode the audio of {}", input.display()))?;
    Ok(())
}

/// ffmpeg muxer name and audio codec for a video container extension.
/// WebM only allows Opus/Vorbis, so it gets Opus instead of AAC.
fn container(ext: &str) -> Result<(&'static str, &'static [&'static str])> {
    const AAC: &[&str] = &["-c:a", "aac", "-b:a", "320k"];
    const OPUS: &[&str] = &["-c:a", "libopus", "-b:a", "320k"];
    Ok(match ext {
        "mp4" | "m4v" => ("mp4", AAC),
        "mov" => ("mov", AAC),
        "mkv" => ("matroska", AAC),
        "webm" => ("webm", OPUS),
        other => bail!("unsupported video container .{other}"),
    })
}

/// Write `output`: the video (and subtitle) streams of `original`, copied
/// bit-for-bit (`-c copy`), plus `processed_wav` encoded as 48 kHz stereo
/// (the mono signal on both channels), delayed by `probe.audio_offset_s`
/// to keep the original A/V sync. Metadata and chapters are carried over.
pub fn mux_video(
    original: &Path,
    processed_wav: &Path,
    output: &Path,
    ext: &str,
    probe: &Probe,
) -> Result<()> {
    let (muxer, audio_codec) = container(ext)?;
    let offset = format!("{:.6}", probe.audio_offset_s);
    let mut args: Vec<OsString> = ["-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        .map(OsString::from)
        .to_vec();
    args.extend(input_args(original));
    args.extend(["-itsoffset", &offset].map(OsString::from));
    args.extend(input_args(processed_wav));
    args.extend(
        [
            // Video and subtitles from the original, new audio from us.
            // Data tracks (e.g. iPhone "mebx" metadata) are left out: many
            // muxers can't write them and they carry nothing audible/visible.
            "-map",
            "0:v",
            "-map",
            "0:s?",
            "-map",
            "1:a:0",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
            "-c",
            "copy",
        ]
        .map(OsString::from),
    );
    args.extend(audio_codec.iter().map(OsString::from));
    // Duplicate mono to both channels at full level. (`-ac 2` would pan
    // it at -3 dB per channel, and the file would measure 3 LU quiet.)
    args.extend(["-af", "pan=stereo|c0=c0|c1=c0", "-ar", "48000", "-f", muxer].map(OsString::from));
    if matches!(muxer, "mp4" | "mov") {
        args.extend(["-movflags", "+faststart"].map(OsString::from));
    }
    args.push(path_arg(output));
    run("ffmpeg", &refs(&args)).context("ffmpeg could not write the video")?;
    Ok(())
}

/// The new file must be as long as the original, and its audio as long as
/// the original audio, each within one video frame.
pub fn verify_durations(original: &Probe, output_path: &Path) -> Result<String> {
    let out = probe(output_path)?;
    let frame = original.frame_s.unwrap_or(1.0 / 30.0);
    let check = |what: &str, a: Option<f64>, b: Option<f64>| -> Result<f64> {
        match (a, b) {
            (Some(a), Some(b)) => {
                let diff = (a - b).abs();
                ensure!(
                    diff <= frame + 1e-6,
                    "{what} changed by {:.1} ms (more than one {:.1} ms frame)",
                    diff * 1e3,
                    frame * 1e3
                );
                Ok(diff)
            }
            _ => Ok(0.0), // container doesn't report it (e.g. some mkv)
        }
    };
    let total = check("file duration", original.duration_s, out.duration_s)?;
    let audio = check(
        "audio duration",
        original.audio_duration_s,
        out.audio_duration_s,
    )?;
    Ok(format!(
        "duration matches within {:.1} ms, audio within {:.1} ms (1 frame = {:.1} ms)",
        total * 1e3,
        audio * 1e3,
        frame * 1e3
    ))
}

/// Lower-case extension, or "".
pub fn extension(path: &Path) -> String {
    path.extension()
        .and_then(OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .unwrap_or_default()
}

/// `dir/stem.enhanced.ext`.
pub fn enhanced_name(input: &Path, dir: &Path, ext: &str) -> PathBuf {
    let stem = input.file_stem().unwrap_or_else(|| OsStr::new("output"));
    let mut name = stem.to_os_string();
    name.push(format!(".enhanced.{ext}"));
    dir.join(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_cannot_become_options_or_urls() {
        assert_eq!(path_arg(Path::new("-y.mp4")), OsString::from("./-y.mp4"));
        assert_eq!(
            path_arg(Path::new("http:evil.mp4")),
            OsString::from("./http:evil.mp4")
        );
        assert_eq!(
            path_arg(Path::new("dir/a.wav")),
            OsString::from("./dir/a.wav")
        );
        assert_eq!(
            path_arg(Path::new("/abs/-x.wav")),
            OsString::from("/abs/-x.wav")
        );
    }

    #[test]
    fn enhanced_names() {
        assert_eq!(
            enhanced_name(Path::new("/a/talk.final.MOV"), Path::new("/a"), "mov"),
            PathBuf::from("/a/talk.final.enhanced.mov")
        );
    }

    #[test]
    fn frame_rate_parsing() {
        assert!((parse_rate(&Some("30000/1001".into())).unwrap() - 1001.0 / 30000.0).abs() < 1e-12);
        assert_eq!(parse_rate(&Some("0/0".into())), None);
    }
}
