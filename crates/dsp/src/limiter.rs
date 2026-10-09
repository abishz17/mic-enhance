//! Lookahead true-peak limiter.
//!
//! The gain track is built in four steps, each with one job:
//!
//! 1. `need[n]`  = gain that would put the true peak of `[n, n+1)` exactly
//!    on the ceiling (1.0 where it is already below);
//! 2. `hold[n]`  = minimum of `need` from `M` samples before `n` to `L + M`
//!    after (`L` = lookahead, `M` = interpolator half-length). The gain is
//!    therefore already down when the peak, and every sample the
//!    interpolator uses to build it, arrives;
//! 3. release: `hold` may only rise back towards 1.0 with the release time
//!    constant (it may always fall);
//! 4. a moving average over `L + 1` samples turns the steps into ramps
//!    (an `L`-sample attack fade instead of a click).
//!
//! Why the ceiling is guaranteed at sample points: at a peak sample `p`
//! the average in step 4 only covers samples in `[p - L, p]`, all of which
//! step 2 already pulled down to at most `need[p]`, so the average is too.
//! Between samples the gain changes slightly across the interpolator's
//! span, so the output is re-measured and, if anything still pokes over
//! (it is rare and tiny), the limiter runs again on its own output. A
//! final whole-file trim is the last-resort guarantee.
//!
//! Two details make the guarantee hold against an *independent* meter:
//!
//! * **Band-limit first** (8th-order Butterworth at 20 kHz). True peak is
//!   only well-defined for content clearly below Nyquist: on full-band
//!   random +-1 noise a 16x/128-tap reference, our detector and the
//!   BS.1770 meter disagree by up to 3 dB because every finite
//!   interpolator treats 20-24 kHz differently. Nothing audible lives
//!   there and YouTube's codecs remove it anyway.
//! * **Aim 0.05 dB under the ceiling**. After band-limiting, our detector
//!   reads within 0.03 dB of the 16x reference on the torture signal in
//!   `tests/limiter.rs`; the margin covers that with room to spare.

use crate::biquad::Cascade;
use crate::config::LimiterConfig;
use crate::dynamics::GainStats;
use crate::truepeak::TruePeakDetector;
use crate::util::{db_to_gain, gain_to_db, ms_to_samples, one_pole_coeff};
use std::collections::VecDeque;

/// Interpolator half-length, in input samples (see `truepeak`).
const INTERP_HALF_SPAN: usize = 16;
const MAX_PASSES: usize = 4;
/// Internal target below the ceiling, covering detector inaccuracy.
pub const DETECTOR_MARGIN_DB: f64 = 0.05;
const BANDLIMIT_ORDER: usize = 8;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LimiterReport {
    pub gain_reduction: GainStats,
    /// Limiter passes that were needed (0 = nothing exceeded the ceiling).
    pub passes: usize,
    /// Whole-file trim applied as a last resort (0.0 normally).
    pub fallback_trim_db: f64,
    /// Final true peak, measured with the limiter's own detector.
    pub true_peak_dbtp: f64,
}

/// `out[n] = min(values[n - before ..= n + after])`, clipped to the
/// buffer, in O(n) with a monotonic deque (van Herk / Lemire).
fn sliding_min(values: &[f32], before: usize, after: usize) -> Vec<f32> {
    let len = values.len();
    let mut out = Vec::with_capacity(len);
    let mut window: VecDeque<usize> = VecDeque::new();
    let mut next = 0; // next index to admit
    for n in 0..len {
        while next < len && next <= n + after {
            while window.back().is_some_and(|&b| values[b] >= values[next]) {
                window.pop_back();
            }
            window.push_back(next);
            next += 1;
        }
        while window.front().is_some_and(|&f| f + before < n) {
            window.pop_front();
        }
        out.push(window.front().map_or(1.0, |&f| values[f]));
    }
    out
}

/// One limiting pass. Returns the linear gain track that was applied.
fn limit_pass(
    buf: &mut [f32],
    peaks: &[f32],
    ceiling: f32,
    lookahead: usize,
    release: f64,
) -> Vec<f32> {
    // 1. required gain
    let need: Vec<f32> = peaks
        .iter()
        .map(|&p| if p > ceiling { ceiling / p } else { 1.0 })
        .collect();
    // 2. look ahead (and a little behind) with a running minimum
    let held = sliding_min(&need, INTERP_HALF_SPAN, lookahead + INTERP_HALF_SPAN);
    // 3. release: rise slowly, fall instantly
    let mut released = Vec::with_capacity(held.len());
    let mut state = 1.0f64;
    for &h in &held {
        state = f64::from(h).min(state + (1.0 - state) * (1.0 - release));
        released.push(state);
    }
    // 4. moving average over lookahead + 1 samples (front padded with the
    //    first value, which step 2 has already pulled low enough)
    let width = lookahead + 1;
    let first = released.first().copied().unwrap_or(1.0);
    let mut sum = first * width as f64;
    let mut gains = Vec::with_capacity(released.len());
    for n in 0..released.len() {
        let leaving = if n >= width {
            released[n - width]
        } else {
            first
        };
        sum += released[n] - leaving;
        gains.push((sum / width as f64) as f32);
    }
    for (s, &g) in buf.iter_mut().zip(&gains) {
        *s *= g;
    }
    gains
}

/// Limit `buf` in place so its true peak is at most `cfg.ceiling_dbtp`.
pub fn process(buf: &mut [f32], sample_rate: u32, cfg: &LimiterConfig) -> LimiterReport {
    let fs = f64::from(sample_rate);
    if cfg.bandlimit_hz > 0.0 {
        Cascade::butterworth_lowpass(fs, cfg.bandlimit_hz, BANDLIMIT_ORDER).process(buf);
    }
    let detector = TruePeakDetector::new();
    let ceiling = db_to_gain(cfg.ceiling_dbtp - DETECTOR_MARGIN_DB) as f32;
    let lookahead = ms_to_samples(cfg.lookahead_ms, fs).max(1);
    let release = one_pole_coeff(cfg.release_ms, fs);

    let mut total_gain = vec![1.0f32; buf.len()];
    let mut passes = 0;
    let mut peaks = detector.peak_track(buf);
    while passes < MAX_PASSES && peaks.iter().any(|&p| p > ceiling) {
        let gains = limit_pass(buf, &peaks, ceiling, lookahead, release);
        total_gain.iter_mut().zip(&gains).for_each(|(t, g)| *t *= g);
        passes += 1;
        peaks = detector.peak_track(buf);
    }

    let max_peak = peaks.iter().copied().fold(0.0f32, f32::max);
    let mut fallback_trim_db = 0.0;
    if max_peak > ceiling {
        let trim = ceiling / max_peak;
        buf.iter_mut().for_each(|s| *s *= trim);
        total_gain.iter_mut().for_each(|t| *t *= trim);
        fallback_trim_db = gain_to_db(f64::from(trim));
    }

    let gain_db: Vec<f32> = total_gain
        .iter()
        .map(|&g| gain_to_db(f64::from(g)) as f32)
        .collect();
    LimiterReport {
        gain_reduction: GainStats::from_track(&gain_db),
        passes,
        fallback_trim_db,
        true_peak_dbtp: gain_to_db(f64::from(max_peak.min(ceiling))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_min_matches_brute_force() {
        let v: Vec<f32> = (0..50).map(|i| ((i * 37) % 11) as f32).collect();
        let fast = sliding_min(&v, 2, 5);
        for (n, &got) in fast.iter().enumerate() {
            let lo = n.saturating_sub(2);
            let hi = (n + 5).min(v.len() - 1);
            let slow = v[lo..=hi].iter().copied().fold(f32::INFINITY, f32::min);
            assert_eq!(got, slow, "n = {n}");
        }
    }

    #[test]
    fn quiet_signal_is_untouched() {
        let mut x: Vec<f32> = (0..4800).map(|n| 0.1 * (n as f32 * 0.05).sin()).collect();
        let orig = x.clone();
        let cfg = LimiterConfig {
            bandlimit_hz: 0.0,
            ..LimiterConfig::default()
        };
        let r = process(&mut x, 48_000, &cfg);
        assert_eq!(x, orig);
        assert_eq!(r.passes, 0);
    }
}
