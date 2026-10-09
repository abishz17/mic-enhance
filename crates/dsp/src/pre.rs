//! Pre-processing before the restore stage: DC removal, rumble high-pass,
//! and mains-hum notches.
//!
//! This runs *before* the denoiser on purpose: a 50 Hz line and sub-80 Hz
//! rumble are cheap to remove exactly with linear filters, and a neural
//! denoiser then spends its effort on broadband noise instead.

use crate::biquad::Cascade;
use crate::config::PreConfig;
use crate::hum::{self, HumReport};

/// What the pre stage found and did.
#[derive(Debug, Clone, Default)]
pub struct PreReport {
    pub dc_offset: f64,
    pub hum: HumReport,
}

/// Run DC removal, high-pass and hum notches in place.
pub fn process(buf: &mut [f32], sample_rate: f64, cfg: &PreConfig) -> PreReport {
    let mut report = PreReport::default();
    if buf.is_empty() {
        return report;
    }

    if cfg.remove_dc {
        // Offline we know the whole file, so the true mean is available.
        // Subtracting it avoids the step response the high-pass would
        // otherwise produce at the start of a file with a large offset.
        // Slow drift is left to the high-pass.
        let mean = buf.iter().map(|&s| f64::from(s)).sum::<f64>() / buf.len() as f64;
        for s in buf.iter_mut() {
            *s = (f64::from(*s) - mean) as f32;
        }
        report.dc_offset = mean;
    }

    // Detect hum before the high-pass removes most of the 50/60 Hz fundamental.
    report.hum = hum::detect(buf, sample_rate, &cfg.hum);

    let mut filters = Cascade::default();
    if cfg.highpass.enabled {
        filters =
            Cascade::butterworth_highpass(sample_rate, cfg.highpass.freq_hz, cfg.highpass.order);
    }
    for c in hum::notch_bank(&report.hum, sample_rate, &cfg.hum).coefficients() {
        filters.push(c);
    }
    filters.process(buf);
    report
}
