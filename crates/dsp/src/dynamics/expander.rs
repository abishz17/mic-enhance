//! Gentle downward expander that lowers room tone between phrases.
//!
//! A plain expander follows the level continuously, so a level hovering
//! near the threshold makes the gain flutter ("chatter"), and a decaying
//! word tail gets clamped down as it crosses. Two mechanisms prevent that:
//!
//! * **Hysteresis**: an open/closed state. It opens when the level reaches
//!   `threshold`, but closes only once the level falls below
//!   `threshold - hysteresis`. Noise wobbling by +-2 dB around one value
//!   cannot toggle it.
//! * **Hold**: after falling below the close level it stays open for
//!   `hold_ms` more, which covers the decay of a word ending.
//!
//! While closed, the gain follows a 1:`ratio` expansion curve under the
//! threshold, capped at `range_db`, so room tone is turned down, not cut.
//! Lookahead lets the gain start rising just before a word begins, so
//! onsets are never clipped.

use super::{GainStats, apply_gain_track, rms_envelope_db, smooth_gain_db};
use crate::config::ExpanderConfig;
use crate::util::{ms_to_samples, percentile, rms_db};

/// Block length for the auto-threshold level histogram.
const HISTOGRAM_BLOCK_MS: f64 = 50.0;
/// Blocks quieter than this are digital silence and say nothing about room tone.
const DIGITAL_SILENCE_DB: f64 = -100.0;
/// Auto threshold stays at least this far below the speech level.
const MIN_SPEECH_DISTANCE_DB: f64 = 18.0;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExpanderReport {
    pub threshold_db: f64,
    pub gain_reduction: GainStats,
    /// Number of open<->closed state changes (a chatter indicator).
    pub transitions: usize,
}

/// Pick a threshold from the file itself: `noise floor + margin`, where the
/// floor is the 10th percentile of 50 ms block levels and speech is the
/// 95th. Capped at `speech - 18 dB` so speech is never expanded even when
/// the noise is high, and kept within -80..-20 dBFS.
pub fn auto_threshold(buf: &[f32], sample_rate: f64, margin_db: f64) -> Option<f64> {
    let block = ms_to_samples(HISTOGRAM_BLOCK_MS, sample_rate).max(1);
    let levels: Vec<f64> = buf
        .chunks_exact(block)
        .map(rms_db)
        .filter(|&l| l > DIGITAL_SILENCE_DB)
        .collect();
    let floor = percentile(&levels, 10.0)?;
    let speech = percentile(&levels, 95.0)?;
    Some(
        (floor + margin_db)
            .min(speech - MIN_SPEECH_DISTANCE_DB)
            .clamp(-80.0, -20.0),
    )
}

/// Open/closed state machine with hysteresis and hold.
struct Gate {
    open_db: f64,
    close_db: f64,
    hold_samples: usize,
    open: bool,
    hold_left: usize,
    transitions: usize,
}

impl Gate {
    fn new(threshold_db: f64, hysteresis_db: f64, hold_samples: usize) -> Self {
        Self {
            open_db: threshold_db,
            close_db: threshold_db - hysteresis_db,
            hold_samples,
            open: false,
            hold_left: 0,
            transitions: 0,
        }
    }

    /// Advance one sample; returns whether the gate is open.
    fn step(&mut self, level_db: f64) -> bool {
        let was_open = self.open;
        if self.open {
            if level_db >= self.close_db {
                self.hold_left = self.hold_samples;
            } else if self.hold_left > 0 {
                self.hold_left -= 1;
            } else {
                self.open = false;
            }
        } else if level_db >= self.open_db {
            self.open = true;
            self.hold_left = self.hold_samples;
        }
        if was_open != self.open {
            self.transitions += 1;
        }
        self.open
    }
}

/// Expand `buf` in place.
pub fn process(buf: &mut [f32], sample_rate: u32, cfg: &ExpanderConfig) -> ExpanderReport {
    let fs = f64::from(sample_rate);
    let threshold_db = if cfg.auto_threshold {
        auto_threshold(buf, fs, cfg.auto_margin_db).unwrap_or(cfg.threshold_db)
    } else {
        cfg.threshold_db
    };

    let levels = rms_envelope_db(buf.iter().map(|&s| f64::from(s)), cfg.detector_ms, fs);
    let mut gate = Gate::new(
        threshold_db,
        cfg.hysteresis_db,
        ms_to_samples(cfg.hold_ms, fs),
    );
    let mut gain_db: Vec<f32> = levels
        .iter()
        .map(|&level| {
            let level = f64::from(level);
            if gate.step(level) {
                0.0
            } else {
                // Below threshold: 1 dB under -> (ratio - 1) dB extra down.
                let under = (level - threshold_db).min(0.0);
                (under * (cfg.ratio - 1.0)).max(-cfg.range_db) as f32
            }
        })
        .collect();
    // Reduction growing = closing (slow, word tails); shrinking = opening (fast).
    smooth_gain_db(&mut gain_db, cfg.close_ms, cfg.open_ms, fs);
    apply_gain_track(buf, &gain_db, ms_to_samples(cfg.lookahead_ms, fs));

    ExpanderReport {
        threshold_db,
        gain_reduction: GainStats::from_track(&gain_db),
        transitions: gate.transitions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_has_hysteresis_and_hold() {
        let mut gate = Gate::new(-40.0, 4.0, 3);
        assert!(!gate.step(-41.0)); // below open level: stays closed
        assert!(gate.step(-40.0)); // opens
        assert!(gate.step(-43.0)); // inside hysteresis band: stays open
        assert!(gate.step(-45.0)); // below close level: hold 3 -> 2
        assert!(gate.step(-45.0)); // 2 -> 1
        assert!(gate.step(-45.0)); // 1 -> 0
        assert!(!gate.step(-45.0)); // hold expired: closes
        assert_eq!(gate.transitions, 2);
    }

    #[test]
    fn auto_threshold_sits_between_floor_and_speech() {
        let fs = 48_000.0;
        // 1 s "room tone" at about -60 dBFS, 1 s "speech" at about -20 dBFS.
        let mut x: Vec<f32> = (0..48_000)
            .map(|n| if n % 2 == 0 { 0.001 } else { -0.001 })
            .collect();
        x.extend((0..48_000).map(|n| if n % 2 == 0 { 0.1f32 } else { -0.1 }));
        let t = auto_threshold(&x, fs, 6.0).unwrap();
        assert!((t - -54.0).abs() < 0.5, "{t}");
    }

    #[test]
    fn auto_threshold_of_digital_silence_is_none() {
        assert_eq!(auto_threshold(&[0.0; 48_000], 48_000.0, 6.0), None);
    }
}
