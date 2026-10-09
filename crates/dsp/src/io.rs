//! WAV in/out. Everything that touches untrusted bytes lives here, so the
//! rest of the crate can assume clean, finite, 48 kHz mono `f32` audio.
//!
//! Guards against hostile or broken input (each returns a clear error):
//! * unreadable / non-WAV / unsupported encodings  -> `Open`, `Unsupported`
//! * absurd sample rates                           -> `SampleRate`
//! * zero samples                                  -> `Empty`
//! * longer than the configured limit              -> `TooLong`
//!   (checked against the header *and* while reading, because a header
//!   can lie; memory is never reserved from the header alone)
//! * NaN / Inf / absurd float values               -> repaired to 0.0 if
//!   rare, `Corrupt` if more than 1 % of samples
//! * truncated data chunk                          -> keep what was read,
//!   flagged in `SourceInfo::truncated`

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// The one internal sample rate.
pub const SAMPLE_RATE: u32 = 48_000;
pub const MIN_INPUT_RATE: u32 = 8_000;
pub const MAX_INPUT_RATE: u32 = 384_000;
/// Float samples beyond +36 dBFS are treated as corrupt, not loud.
const MAX_ABS_SAMPLE: f32 = 64.0;
/// Above this fraction of bad samples the file is rejected, not repaired.
const MAX_BAD_FRACTION: f64 = 0.01;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("cannot open {path}: {source}")]
    Open { path: PathBuf, source: hound::Error },
    #[error("{path}: unsupported WAV encoding ({detail}); use PCM 8/16/24/32-bit or 32-bit float")]
    Unsupported { path: PathBuf, detail: String },
    #[error("{path}: sample rate {rate} Hz is outside {MIN_INPUT_RATE}..={MAX_INPUT_RATE} Hz")]
    SampleRate { path: PathBuf, rate: u32 },
    #[error("{path}: contains no audio samples")]
    Empty { path: PathBuf },
    #[error("{path}: {seconds:.0} s of audio exceeds the {max_seconds:.0} s limit")]
    TooLong {
        path: PathBuf,
        seconds: f64,
        max_seconds: f64,
    },
    #[error(
        "{path}: {bad} of {total} samples are NaN, infinite or absurdly large; the file looks corrupt"
    )]
    Corrupt {
        path: PathBuf,
        bad: usize,
        total: usize,
    },
    #[error("resampling {from} Hz -> {to} Hz failed: {detail}")]
    Resample { from: u32, to: u32, detail: String },
    #[error("cannot write {path}: {detail}")]
    Write { path: PathBuf, detail: String },
}

/// Limits applied while loading.
#[derive(Debug, Clone, Copy)]
pub struct LoadLimits {
    pub max_seconds: f64,
}

impl Default for LoadLimits {
    fn default() -> Self {
        Self {
            max_seconds: 60.0 * 60.0,
        }
    }
}

/// What the input file was before conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub float: bool,
    pub frames: usize,
    /// NaN / Inf / absurd samples replaced by 0.0.
    pub repaired_samples: usize,
    /// The data chunk ended early; only the readable part was kept.
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct LoadedAudio {
    /// 48 kHz mono.
    pub samples: Vec<f32>,
    pub source: SourceInfo,
}

/// Load any supported WAV as 48 kHz mono `f32`.
pub fn load_wav(path: &Path, limits: LoadLimits) -> Result<LoadedAudio, AudioError> {
    let reader = WavReader::open(path).map_err(|source| match source {
        hound::Error::Unsupported => AudioError::Unsupported {
            path: path.into(),
            detail: "format tag not supported".into(),
        },
        source => AudioError::Open {
            path: path.into(),
            source,
        },
    })?;
    let spec = reader.spec();
    if !(MIN_INPUT_RATE..=MAX_INPUT_RATE).contains(&spec.sample_rate) {
        return Err(AudioError::SampleRate {
            path: path.into(),
            rate: spec.sample_rate,
        });
    }
    if spec.channels == 0 {
        return Err(AudioError::Unsupported {
            path: path.into(),
            detail: "zero channels".into(),
        });
    }
    let max_frames = (limits.max_seconds * f64::from(spec.sample_rate)) as usize;
    let header_frames = reader.duration() as usize;
    if header_frames > max_frames {
        return Err(AudioError::TooLong {
            path: path.into(),
            seconds: header_frames as f64 / f64::from(spec.sample_rate),
            max_seconds: limits.max_seconds,
        });
    }

    let (mono, truncated) = read_mono(reader, spec, path, max_frames)?;
    if mono.is_empty() {
        return Err(AudioError::Empty { path: path.into() });
    }
    let (mono, repaired) = repair_samples(mono, path)?;

    let samples = if spec.sample_rate == SAMPLE_RATE {
        mono
    } else {
        resample(&mono, spec.sample_rate, SAMPLE_RATE)?
    };

    Ok(LoadedAudio {
        source: SourceInfo {
            sample_rate: spec.sample_rate,
            channels: spec.channels,
            bits_per_sample: spec.bits_per_sample,
            float: spec.sample_format == SampleFormat::Float,
            frames: samples.len(),
            repaired_samples: repaired,
            truncated,
        },
        samples,
    })
}

/// Decode, convert to `f32` in -1..1 and average the channels.
fn read_mono<R: std::io::Read>(
    mut reader: WavReader<R>,
    spec: WavSpec,
    path: &Path,
    max_frames: usize,
) -> Result<(Vec<f32>, bool), AudioError> {
    let channels = usize::from(spec.channels);
    let unsupported = |detail: String| AudioError::Unsupported {
        path: path.into(),
        detail,
    };
    // Box the per-format decoders behind one iterator type.
    let samples: Box<dyn Iterator<Item = Result<f32, hound::Error>> + '_> =
        match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Float, 32) => Box::new(reader.samples::<f32>()),
            (SampleFormat::Int, bits @ 8..=32) => {
                let scale = 1.0 / f64::from(1u32 << (bits - 1));
                Box::new(
                    reader
                        .samples::<i32>()
                        .map(move |s| s.map(|v| (f64::from(v) * scale) as f32)),
                )
            }
            (format, bits) => return Err(unsupported(format!("{format:?} {bits}-bit"))),
        };

    // Reserve one second at most: a lying header must not make us allocate
    // gigabytes up front. `Vec` growth handles real lengths.
    let mut mono = Vec::with_capacity((spec.sample_rate as usize).min(max_frames));
    let mut frame_sum = 0.0f64;
    let mut in_frame = 0;
    let mut truncated = false;
    for sample in samples {
        match sample {
            Ok(s) => {
                frame_sum += f64::from(s);
                in_frame += 1;
                if in_frame == channels {
                    if mono.len() == max_frames {
                        return Err(AudioError::TooLong {
                            path: path.into(),
                            seconds: max_frames as f64 / f64::from(spec.sample_rate),
                            max_seconds: max_frames as f64 / f64::from(spec.sample_rate),
                        });
                    }
                    // NaN/Inf propagate through the average and are caught later.
                    mono.push((frame_sum / channels as f64) as f32);
                    frame_sum = 0.0;
                    in_frame = 0;
                }
            }
            // hound reports a short data chunk as a generic I/O error
            // ("Failed to read enough bytes"), so any I/O error mid-stream
            // counts as truncation. An empty result is still rejected later.
            Err(hound::Error::IoError(_)) => {
                truncated = true;
                break;
            }
            Err(e) => return Err(unsupported(e.to_string())),
        }
    }
    Ok((mono, truncated))
}

/// Replace NaN / Inf / absurd values with silence; reject if too many.
fn repair_samples(mut samples: Vec<f32>, path: &Path) -> Result<(Vec<f32>, usize), AudioError> {
    let mut bad = 0;
    for s in &mut samples {
        if !s.is_finite() || s.abs() > MAX_ABS_SAMPLE {
            *s = 0.0;
            bad += 1;
        }
    }
    if bad as f64 > MAX_BAD_FRACTION * samples.len() as f64 {
        return Err(AudioError::Corrupt {
            path: path.into(),
            bad,
            total: samples.len(),
        });
    }
    Ok((samples, bad))
}

/// High-quality sample-rate conversion (rubato's FFT resampler: fixed
/// ratio, linear phase, delay compensated by `process_all`).
pub fn resample(x: &[f32], from: u32, to: u32) -> Result<Vec<f32>, AudioError> {
    use rubato::audioadapter_buffers::direct::InterleavedSlice;
    use rubato::{Fft, FixedSync, Resampler};

    let err = |detail: String| AudioError::Resample { from, to, detail };
    let input: Vec<f64> = x.iter().map(|&s| f64::from(s)).collect();
    let mut resampler = Fft::<f64>::new(from as usize, to as usize, 1024, 1, FixedSync::Both)
        .map_err(|e| err(e.to_string()))?;
    let adapter = InterleavedSlice::new(&input, 1, input.len()).map_err(|e| err(e.to_string()))?;
    let output = resampler
        .process_all(&adapter, input.len(), None)
        .map_err(|e| err(e.to_string()))?;
    // rubato rounds the length up via float maths (44.1k s -> 48001); trim
    // to the exact rounded length so durations line up across stages.
    let exact = ((x.len() as u64 * u64::from(to) + u64::from(from) / 2) / u64::from(from)) as usize;
    let mut out: Vec<f32> = output.take_data().into_iter().map(|s| s as f32).collect();
    out.truncate(exact);
    Ok(out)
}

/// Output sample encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// 24-bit integer PCM (delivery default).
    #[default]
    Pcm24,
    /// 32-bit float (no clipping; used for stage dumps and ML hand-off).
    Float32,
}

/// Write 48 kHz mono. The file is written next to `path` under a temporary
/// name and renamed into place, so a failure never leaves a half-written
/// file that looks finished. Returns the number of samples that had to be
/// clipped (only possible for `Pcm24`).
pub fn write_wav(path: &Path, samples: &[f32], format: OutputFormat) -> Result<usize, AudioError> {
    let werr = |detail: String| AudioError::Write {
        path: path.into(),
        detail,
    };
    let spec = match format {
        OutputFormat::Pcm24 => WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 24,
            sample_format: SampleFormat::Int,
        },
        OutputFormat::Float32 => WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        },
    };
    let mut tmp_name = path.file_name().unwrap_or_default().to_os_string();
    tmp_name.push(".partial");
    let tmp = path.with_file_name(tmp_name);

    let mut clipped = 0;
    let result = (|| -> Result<(), hound::Error> {
        let mut writer = WavWriter::create(&tmp, spec)?;
        match format {
            OutputFormat::Pcm24 => {
                const FULL: f64 = 8_388_608.0; // 2^23
                for &s in samples {
                    let v = (f64::from(s) * FULL).round();
                    let q = v.clamp(-FULL, FULL - 1.0);
                    if q != v {
                        clipped += 1;
                    }
                    writer.write_sample(q as i32)?;
                }
            }
            OutputFormat::Float32 => {
                for &s in samples {
                    writer.write_sample(s)?;
                }
            }
        }
        writer.finalize()
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(werr(e.to_string()));
    }
    std::fs::rename(&tmp, path).map_err(|e| werr(e.to_string()))?;
    Ok(clipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repair_counts_and_rejects() {
        let (fixed, bad) = repair_samples(
            [vec![f32::NAN, f32::INFINITY, 1e30], vec![0.1; 1000]].concat(),
            Path::new("x.wav"),
        )
        .unwrap();
        assert_eq!(bad, 3);
        assert!(fixed.iter().all(|s| s.is_finite()));
        assert!(matches!(
            repair_samples(vec![f32::NAN; 100], Path::new("x.wav")),
            Err(AudioError::Corrupt { bad: 100, .. })
        ));
    }

    #[test]
    fn resample_44k1_keeps_length_and_tone() {
        let x: Vec<f32> = (0..44_100)
            .map(|n| (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / 44_100.0).sin() as f32 * 0.5)
            .collect();
        let y = resample(&x, 44_100, 48_000).unwrap();
        assert_eq!(y.len(), 48_000);
        let rms = crate::util::rms_db(&y[4000..44_000]);
        assert!(
            (rms - crate::util::rms_db(&x[4000..40_000])).abs() < 0.05,
            "{rms}"
        );
    }
}
