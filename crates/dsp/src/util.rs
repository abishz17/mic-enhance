//! Small numeric helpers shared by every stage.

/// Smallest linear magnitude we ever convert to dB. Keeps `log10(0)` from
/// producing `-inf` and poisoning later arithmetic with NaN (`-inf * 0`).
const MIN_MAGNITUDE: f64 = 1e-12; // -240 dB

/// Decibels to linear amplitude gain (`+6.02 dB -> 2.0`).
pub fn db_to_gain(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// Linear amplitude to decibels, floored at -240 dB instead of `-inf`.
pub fn gain_to_db(gain: f64) -> f64 {
    20.0 * gain.abs().max(MIN_MAGNITUDE).log10()
}

/// Mean-square power to decibels, floored at -240 dB instead of `-inf`.
pub fn power_to_db(power: f64) -> f64 {
    10.0 * power.max(MIN_MAGNITUDE * MIN_MAGNITUDE).log10()
}

/// Coefficient of a one-pole smoother with time constant `ms`
/// (the output covers 63 % of a step in `ms` milliseconds).
/// `y[n] = coeff * y[n-1] + (1 - coeff) * x[n]`. A zero time constant
/// returns 0.0, i.e. "follow the input instantly".
pub fn one_pole_coeff(ms: f64, sample_rate: f64) -> f64 {
    if ms <= 0.0 {
        0.0
    } else {
        (-1.0 / (ms * 1e-3 * sample_rate)).exp()
    }
}

/// Milliseconds to a whole number of samples (rounded).
pub fn ms_to_samples(ms: f64, sample_rate: f64) -> usize {
    (ms.max(0.0) * 1e-3 * sample_rate).round() as usize
}

/// Root-mean-square of a block, in dBFS (a full-scale sine reads -3.01 dB).
pub fn rms_db(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return power_to_db(0.0);
    }
    let sum: f64 = samples.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    power_to_db(sum / samples.len() as f64)
}

/// Multiply every sample by a linear gain.
pub fn apply_gain(samples: &mut [f32], gain: f64) {
    for s in samples {
        *s = (f64::from(*s) * gain) as f32;
    }
}

/// Largest absolute sample value.
pub fn sample_peak(samples: &[f32]) -> f32 {
    samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()))
}

/// The `p`-th percentile (0..=100) of `values` by nearest rank.
/// Returns `None` for an empty slice. Sorts a copy, so O(n log n).
pub fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = ((p.clamp(0.0, 100.0) / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    Some(sorted[rank])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn db_round_trip() {
        for db in [-60.0, -6.0, 0.0, 3.0, 12.0] {
            assert!((gain_to_db(db_to_gain(db)) - db).abs() < 1e-9);
        }
    }

    #[test]
    fn silence_is_finite_in_db() {
        assert!(gain_to_db(0.0).is_finite());
        assert!(power_to_db(0.0).is_finite());
        assert!(rms_db(&[0.0; 16]).is_finite());
    }

    #[test]
    fn full_scale_sine_rms_is_minus_3_db() {
        let sine: Vec<f32> = (0..48_000)
            .map(|n| (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / 48_000.0).sin() as f32)
            .collect();
        assert!((rms_db(&sine) + 3.0103).abs() < 0.01);
    }

    #[test]
    fn percentile_basics() {
        let v = [5.0, 1.0, 3.0, 2.0, 4.0];
        assert_eq!(percentile(&v, 0.0), Some(1.0));
        assert_eq!(percentile(&v, 50.0), Some(3.0));
        assert_eq!(percentile(&v, 100.0), Some(5.0));
        assert_eq!(percentile(&[], 50.0), None);
    }
}
