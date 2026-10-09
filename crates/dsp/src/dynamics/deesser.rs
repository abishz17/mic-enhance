//! Split-band de-esser.
//!
//! **Detector**: band-pass 5-8 kHz (where "s" and "sh" energy peaks),
//! fast RMS, compressor-style curve capped at `max_reduction_db`.
//!
//! **Band split**: instead of a crossover (whose phase shifts make the two
//! bands sum with a bump of up to +1.8 dB around the split while
//! reducing), the high band is a *zero-phase* high-pass, made by running
//! a Butterworth high-pass forwards and then backwards (`filtfilt`).
//! Its response `|H|^2` is real and between 0 and 1, so
//!
//! ```text
//! out = in - (1 - g) * high      =>   |response| = 1 - (1 - g) |H|^2
//! ```
//!
//! is exactly `in` when g = 1, never boosts, and bottoms out at `g` far
//! above the split. Only possible offline (the backward pass needs the
//! future), which this tool is.

use super::{
    GainStats, apply_gain_track, compressor::static_curve, rms_envelope_db, smooth_gain_db,
};
use crate::biquad::{Cascade, Coefficients, filtfilt};
use crate::config::DeesserConfig;
use crate::util::{db_to_gain, ms_to_samples};

const BUTTERWORTH_Q: f64 = std::f64::consts::FRAC_1_SQRT_2;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DeesserReport {
    pub gain_reduction: GainStats,
}

/// De-ess `buf` in place.
pub fn process(buf: &mut [f32], sample_rate: u32, cfg: &DeesserConfig) -> DeesserReport {
    let fs = f64::from(sample_rate);

    // Detector: band-pass 5-8 kHz built from a high-pass and a low-pass.
    let mut detector = Cascade::new([
        Coefficients::highpass(fs, cfg.detector_low_hz, BUTTERWORTH_Q),
        Coefficients::lowpass(fs, cfg.detector_high_hz, BUTTERWORTH_Q),
    ]);
    let band = buf.iter().map(|&s| detector.process_sample(f64::from(s)));
    let mut gain_db = rms_envelope_db(band, cfg.rms_ms, fs);
    for g in &mut gain_db {
        let level = f64::from(*g);
        // Knee fixed at 4 dB: sibilants should be caught smoothly, not snapped.
        let reduction = static_curve(level, cfg.threshold_db, cfg.ratio, 4.0) - level;
        *g = reduction.max(-cfg.max_reduction_db) as f32;
    }
    smooth_gain_db(&mut gain_db, cfg.attack_ms, cfg.release_ms, fs);

    // Zero-phase high band, then out = in - (1 - g) * high.
    let mut high = buf.to_vec();
    filtfilt(
        &Cascade::new([Coefficients::highpass(fs, cfg.split_hz, BUTTERWORTH_Q)]),
        &mut high,
    );
    let mut cut = vec![1.0f32; buf.len()];
    apply_gain_track(&mut cut, &gain_db, ms_to_samples(cfg.lookahead_ms, fs));
    for ((s, &h), &g) in buf.iter_mut().zip(&high).zip(&cut) {
        *s -= (1.0 - g) * h;
    }

    DeesserReport {
        gain_reduction: GainStats::from_track(&gain_db),
    }
}

/// Steady-state magnitude (dB) of the de-esser at `freq` when the gain is
/// fixed at `reduction_db`. Used by tests to check "never boosts".
pub fn static_response_db(fs: f64, cfg: &DeesserConfig, reduction_db: f64, freq: f64) -> f64 {
    let h2 = db_to_gain(
        2.0 * Coefficients::highpass(fs, cfg.split_hz, BUTTERWORTH_Q).magnitude_db(fs, freq),
    );
    let g = db_to_gain(-reduction_db);
    crate::util::gain_to_db(1.0 - (1.0 - g) * h2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_never_boosts_and_cut_is_bounded() {
        let cfg = DeesserConfig::default();
        for f in [100.0, 1000.0, 3000.0, 4500.0, 6000.0, 10_000.0, 20_000.0] {
            let r = static_response_db(48_000.0, &cfg, 6.0, f);
            assert!((-6.0 - 1e-9..=1e-9).contains(&r), "{f} Hz: {r} dB");
        }
        assert!(static_response_db(48_000.0, &cfg, 6.0, 500.0) > -0.01);
        assert!(static_response_db(48_000.0, &cfg, 6.0, 15_000.0) < -5.5);
    }
}
