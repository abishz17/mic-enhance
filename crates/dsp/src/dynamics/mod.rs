//! Level-dependent gain stages: compressor, expander, de-esser.
//!
//! All three share the same offline shape, which keeps each one short and
//! easy to test in isolation:
//!
//! 1. **detect**: follow the signal level (RMS, in dB) sample by sample;
//! 2. **compute**: turn level into a target gain with a static curve;
//! 3. **smooth**: apply attack / release time constants to that gain;
//! 4. **apply**: multiply the audio by the gain track, reading the track
//!    `lookahead` samples ahead.
//!
//! Step 4 is where offline processing pays off. A real-time compressor must
//! delay its audio to "look ahead"; here we already have the whole gain
//! track in memory, so reading it shifted gives the same effect with zero
//! output latency and no extra buffer.

pub mod compressor;
pub mod deesser;
pub mod expander;

use crate::util::{one_pole_coeff, power_to_db};

/// Running mean-square level in dB for every sample (one-pole average of
/// `x^2` with time constant `ms`). A sine of amplitude 1 settles at -3 dB.
pub fn rms_envelope_db(x: impl IntoIterator<Item = f64>, ms: f64, sample_rate: f64) -> Vec<f32> {
    let coeff = one_pole_coeff(ms, sample_rate);
    let mut mean_square = 0.0f64;
    x.into_iter()
        .map(|s| {
            mean_square = coeff * mean_square + (1.0 - coeff) * s * s;
            power_to_db(mean_square) as f32
        })
        .collect()
}

/// Attack/release smoothing of a gain-in-dB track, in place.
///
/// "Attack" applies when the gain is moving *down* (more reduction),
/// "release" when it recovers upwards. Done in the dB domain so the
/// recovery sounds equally fast at every depth.
pub fn smooth_gain_db(gain_db: &mut [f32], attack_ms: f64, release_ms: f64, sample_rate: f64) {
    let attack = one_pole_coeff(attack_ms, sample_rate);
    let release = one_pole_coeff(release_ms, sample_rate);
    let Some(&first) = gain_db.first() else {
        return;
    };
    let mut state = f64::from(first);
    for g in gain_db.iter_mut() {
        let target = f64::from(*g);
        let coeff = if target < state { attack } else { release };
        state = coeff * state + (1.0 - coeff) * target;
        *g = state as f32;
    }
}

/// Multiply `buf` by the gain track, reading `lookahead` samples ahead:
/// `out[n] = in[n] * gain[n + lookahead]` (the last gain value is reused
/// at the very end).
pub fn apply_gain_track(buf: &mut [f32], gain_db: &[f32], lookahead: usize) {
    debug_assert_eq!(buf.len(), gain_db.len());
    let last = gain_db.len().saturating_sub(1);
    for (n, s) in buf.iter_mut().enumerate() {
        let g = gain_db[(n + lookahead).min(last)];
        *s = (f64::from(*s) * crate::util::db_to_gain(f64::from(g))) as f32;
    }
}

/// Summary of a gain track for the CLI report.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GainStats {
    /// Deepest reduction, positive dB.
    pub max_reduction_db: f64,
    /// Mean reduction while active (positive dB).
    pub mean_active_reduction_db: f64,
    /// Fraction of samples with more than 0.5 dB of reduction.
    pub active_fraction: f64,
}

impl GainStats {
    const ACTIVE_DB: f32 = 0.5;

    pub fn from_track(gain_db: &[f32]) -> Self {
        let mut max: f32 = 0.0;
        let mut sum = 0.0f64;
        let mut active = 0usize;
        for &g in gain_db {
            let reduction = -g;
            max = max.max(reduction);
            if reduction > Self::ACTIVE_DB {
                sum += f64::from(reduction);
                active += 1;
            }
        }
        Self {
            max_reduction_db: f64::from(max),
            mean_active_reduction_db: if active > 0 { sum / active as f64 } else { 0.0 },
            active_fraction: if gain_db.is_empty() {
                0.0
            } else {
                active as f64 / gain_db.len() as f64
            },
        }
    }
}

impl std::fmt::Display for GainStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max {:.1} dB, mean {:.1} dB while active, active {:.0}% of the time",
            self.max_reduction_db,
            self.mean_active_reduction_db,
            self.active_fraction * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothing_reaches_63_percent_in_one_time_constant() {
        let fs = 48_000.0;
        let mut track = vec![0.0f32; 1];
        track.extend(std::iter::repeat_n(-10.0f32, 480)); // 10 ms step
        smooth_gain_db(&mut track, 10.0, 100.0, fs);
        let reached = -track[480];
        assert!((reached - 6.32).abs() < 0.05, "reached {reached}");
    }

    #[test]
    fn lookahead_reads_future_gain() {
        let mut buf = vec![1.0f32; 4];
        let gains = [0.0, 0.0, -6.0206, -6.0206];
        apply_gain_track(&mut buf, &gains, 2);
        assert!((buf[0] - 0.5).abs() < 1e-4);
        assert!((buf[3] - 0.5).abs() < 1e-4);
    }
}
