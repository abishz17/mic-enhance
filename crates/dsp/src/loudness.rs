//! Loudness (ITU-R BS.1770-4 / EBU R128) measurement via the `ebur128`
//! crate, plus gain alignment to a target.

use crate::util::{apply_gain, db_to_gain, gain_to_db, sample_peak};
use ebur128::{Channel, EbuR128, Mode};
use thiserror::Error;

#[derive(Debug, Error)]
#[error("loudness meter error: {0}")]
pub struct LoudnessError(#[from] ebur128::Error);

/// Loudness and peak figures for one buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measurement {
    /// Gated integrated loudness. `None` when nothing is above the -70 LUFS
    /// absolute gate (silence, or near-silence).
    pub integrated_lufs: Option<f64>,
    /// True peak per BS.1770 (4x oversampled at 48 kHz), dBTP.
    pub true_peak_dbtp: f64,
    pub sample_peak_dbfs: f64,
}

impl std::fmt::Display for Measurement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.integrated_lufs {
            Some(l) => write!(f, "{l:.2} LUFS")?,
            None => write!(f, "-inf LUFS (silent)")?,
        }
        write!(
            f,
            ", true peak {:.2} dBTP, sample peak {:.2} dBFS",
            self.true_peak_dbtp, self.sample_peak_dbfs
        )
    }
}

/// Measure integrated loudness and true peak of a mono buffer.
///
/// `dual_mono` counts the channel twice (+3.01 dB), matching how a mono
/// file reads once it is placed on both channels of a stereo mix.
pub fn measure(x: &[f32], sample_rate: u32, dual_mono: bool) -> Result<Measurement, LoudnessError> {
    let mut meter = EbuR128::new(1, sample_rate, Mode::I | Mode::TRUE_PEAK)?;
    if dual_mono {
        meter.set_channel(0, Channel::DualMono)?;
    }
    meter.add_frames_f32(x)?;
    let integrated = meter.loudness_global()?;
    let true_peak = meter.true_peak(0)?;
    Ok(Measurement {
        integrated_lufs: (integrated.is_finite() && integrated > -70.0).then_some(integrated),
        true_peak_dbtp: gain_to_db(true_peak),
        sample_peak_dbfs: gain_to_db(f64::from(sample_peak(x))),
    })
}

/// Integrated loudness only (no true-peak oversampling, so cheaper).
pub fn integrated_lufs(
    x: &[f32],
    sample_rate: u32,
    dual_mono: bool,
) -> Result<Option<f64>, LoudnessError> {
    let mut meter = EbuR128::new(1, sample_rate, Mode::I)?;
    if dual_mono {
        meter.set_channel(0, Channel::DualMono)?;
    }
    meter.add_frames_f32(x)?;
    let l = meter.loudness_global()?;
    Ok((l.is_finite() && l > -70.0).then_some(l))
}

/// Scale `buf` so its integrated loudness becomes `target_lufs`, boosting
/// by at most `max_gain_db`. Returns the gain applied in dB, or `None` if
/// the buffer is silent (left untouched).
pub fn align(
    buf: &mut [f32],
    sample_rate: u32,
    target_lufs: f64,
    max_gain_db: f64,
) -> Result<Option<f64>, LoudnessError> {
    let Some(current) = integrated_lufs(buf, sample_rate, false)? else {
        return Ok(None);
    };
    let gain_db = (target_lufs - current).min(max_gain_db);
    apply_gain(buf, db_to_gain(gain_db));
    Ok(Some(gain_db))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f64, amp: f64, secs: f64) -> Vec<f32> {
        (0..(secs * 48_000.0) as usize)
            .map(|n| (amp * (2.0 * std::f64::consts::PI * freq * n as f64 / 48_000.0).sin()) as f32)
            .collect()
    }

    #[test]
    fn full_scale_1k_sine_is_minus_3_lufs() {
        // EBU Tech 3341: a 0 dBFS 1 kHz sine in one channel reads -3.01 LUFS.
        let m = measure(&sine(1000.0, 1.0, 5.0), 48_000, false).unwrap();
        assert!((m.integrated_lufs.unwrap() + 3.01).abs() < 0.05, "{m}");
        let dual = measure(&sine(1000.0, 1.0, 5.0), 48_000, true).unwrap();
        assert!((dual.integrated_lufs.unwrap() - 0.0).abs() < 0.05, "{dual}");
    }

    #[test]
    fn silence_has_no_loudness() {
        let m = measure(&vec![0.0; 48_000 * 2], 48_000, false).unwrap();
        assert_eq!(m.integrated_lufs, None);
        assert!(m.true_peak_dbtp.is_finite());
        let mut silent = vec![0.0f32; 48_000];
        assert_eq!(align(&mut silent, 48_000, -14.0, 40.0).unwrap(), None);
    }

    #[test]
    fn align_hits_target_and_respects_max_gain() {
        let mut x = sine(1000.0, 0.01, 5.0); // about -43 LUFS
        align(&mut x, 48_000, -23.0, 60.0).unwrap();
        let l = integrated_lufs(&x, 48_000, false).unwrap().unwrap();
        assert!((l + 23.0).abs() < 0.01, "{l}");

        let mut quiet = sine(1000.0, 0.001, 5.0); // about -63 LUFS
        let gain = align(&mut quiet, 48_000, -14.0, 20.0).unwrap().unwrap();
        assert_eq!(gain, 20.0);
    }
}
