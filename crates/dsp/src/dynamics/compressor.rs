//! Feed-forward RMS compressor with a soft knee, lookahead and
//! loudness-preserving auto make-up gain.

use super::{GainStats, apply_gain_track, rms_envelope_db, smooth_gain_db};
use crate::config::CompressorConfig;
use crate::loudness::{self, LoudnessError};
use crate::util::{apply_gain, db_to_gain, ms_to_samples};

/// Static curve: input level -> output level (both dB), with a quadratic
/// soft knee of total width `knee_db` centred on the threshold
/// (Giannoulis, Massberg & Reiss, JAES 2012, eq. 4).
pub fn static_curve(level_db: f64, threshold_db: f64, ratio: f64, knee_db: f64) -> f64 {
    let over = level_db - threshold_db;
    let slope = 1.0 / ratio - 1.0;
    if 2.0 * over < -knee_db {
        level_db
    } else if knee_db > 0.0 && 2.0 * over.abs() <= knee_db {
        level_db + slope * (over + knee_db / 2.0).powi(2) / (2.0 * knee_db)
    } else {
        threshold_db + over / ratio
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CompressorReport {
    pub gain_reduction: GainStats,
    pub makeup_db: f64,
}

/// Compress `buf` in place.
pub fn process(
    buf: &mut [f32],
    sample_rate: u32,
    cfg: &CompressorConfig,
) -> Result<CompressorReport, LoudnessError> {
    let fs = f64::from(sample_rate);
    let loudness_before = if cfg.auto_makeup {
        loudness::integrated_lufs(buf, sample_rate, false)?
    } else {
        None
    };

    // 1. detect, 2. compute, 3. smooth
    let mut gain_db = rms_envelope_db(buf.iter().map(|&s| f64::from(s)), cfg.rms_ms, fs);
    for g in &mut gain_db {
        let level = f64::from(*g);
        *g = (static_curve(level, cfg.threshold_db, cfg.ratio, cfg.knee_db) - level) as f32;
    }
    smooth_gain_db(&mut gain_db, cfg.attack_ms, cfg.release_ms, fs);

    // 4. apply with lookahead
    apply_gain_track(buf, &gain_db, ms_to_samples(cfg.lookahead_ms, fs));

    // Make-up: offline we can simply measure what the compression cost in
    // loudness and give exactly that back, so the stage is loudness-neutral
    // and every later threshold still sees the working level. (The textbook
    // formula -T(1-1/R)/2 only guesses this from the curve.)
    let makeup_db = match (cfg.auto_makeup, loudness_before) {
        (true, Some(before)) => match loudness::integrated_lufs(buf, sample_rate, false)? {
            Some(after) => before - after,
            None => 0.0,
        },
        (true, None) => 0.0,
        (false, _) => cfg.makeup_db,
    };
    apply_gain(buf, db_to_gain(makeup_db));

    Ok(CompressorReport {
        gain_reduction: GainStats::from_track(&gain_db),
        makeup_db,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curve_regions() {
        // Below the knee: unity. Above: slope 1/ratio. Continuous at edges.
        assert_eq!(static_curve(-50.0, -30.0, 3.0, 6.0), -50.0);
        assert!((static_curve(-12.0, -30.0, 3.0, 6.0) - (-24.0)).abs() < 1e-12);
        for edge in [-33.0, -27.0] {
            let below = static_curve(edge - 1e-9, -30.0, 3.0, 6.0);
            let above = static_curve(edge + 1e-9, -30.0, 3.0, 6.0);
            assert!((below - above).abs() < 1e-6);
        }
        // Hard knee.
        assert_eq!(static_curve(-30.0, -30.0, 3.0, 0.0), -30.0);
        assert!((static_curve(-27.0, -30.0, 3.0, 0.0) - (-29.0)).abs() < 1e-12);
    }
}
