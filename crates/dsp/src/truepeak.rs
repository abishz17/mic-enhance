//! True-peak detection: the peak of the continuous waveform a DAC (or a
//! lossy codec's decoder) reconstructs, which can sit up to several dB
//! above the largest sample when samples straddle a peak.
//!
//! We estimate it by 4x oversampling: three in-between points per sample
//! are interpolated with a Kaiser-windowed sinc (32 taps per point), and
//! each local maximum of the 4x sequence is refined with a parabola
//! through its two neighbours. The parabola step matters for content near
//! 20 kHz, where the true crest can fall between two 4x points (plain 4x
//! under-reads such a tone by up to 0.5 dB).

use std::f64::consts::PI;

/// Oversampling factor (points per input sample, including the sample).
pub const FACTOR: usize = 4;
/// Input samples on each side of the interpolation point.
const HALF_TAPS: usize = 16;
const TAPS: usize = 2 * HALF_TAPS;
/// Kaiser window shape: ~-80 dB sidelobes for this length.
const KAISER_BETA: f64 = 8.0;

/// Zeroth-order modified Bessel function of the first kind (series form).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..50 {
        term *= (half / k as f64).powi(2);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}

fn sinc(t: f64) -> f64 {
    if t.abs() < 1e-12 {
        1.0
    } else {
        (PI * t).sin() / (PI * t)
    }
}

/// 4x windowed-sinc interpolator plus parabolic peak refinement.
#[derive(Debug, Clone)]
pub struct TruePeakDetector {
    /// `phases[p][j]` weights `x[n + j - HALF_TAPS + 1]` to give the
    /// value at `n + (p + 1) / 4`.
    phases: [[f64; TAPS]; FACTOR - 1],
}

impl Default for TruePeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl TruePeakDetector {
    pub fn new() -> Self {
        let mut phases = [[0.0; TAPS]; FACTOR - 1];
        let norm = bessel_i0(KAISER_BETA);
        for (p, taps) in phases.iter_mut().enumerate() {
            let mu = (p + 1) as f64 / FACTOR as f64;
            for (j, tap) in taps.iter_mut().enumerate() {
                // distance from interpolation point to input sample n + j - 15
                let t = mu - (j as f64 - (HALF_TAPS as f64 - 1.0));
                let r = t / HALF_TAPS as f64;
                let window = if r.abs() < 1.0 {
                    bessel_i0(KAISER_BETA * (1.0 - r * r).sqrt()) / norm
                } else {
                    0.0
                };
                *tap = sinc(t) * window;
            }
            // Unity gain at DC for every phase, so a constant stays constant.
            let sum: f64 = taps.iter().sum();
            taps.iter_mut().for_each(|t| *t /= sum);
        }
        Self { phases }
    }

    /// For every input sample `n`, the true-peak magnitude over the
    /// interval `[n, n+1)`. Samples outside the buffer count as zero.
    pub fn peak_track(&self, x: &[f32]) -> Vec<f32> {
        let len = x.len();
        let at = |i: isize| -> f64 {
            if i >= 0 && (i as usize) < len {
                f64::from(x[i as usize])
            } else {
                0.0
            }
        };
        let interpolate = |n: usize, phase: &[f64; TAPS]| -> f64 {
            let start = n as isize - (HALF_TAPS as isize - 1);
            if start >= 0 && n + HALF_TAPS < len {
                // fast path: fully inside the buffer
                let window = &x[start as usize..start as usize + TAPS];
                window
                    .iter()
                    .zip(phase)
                    .map(|(&s, &c)| f64::from(s) * c)
                    .sum()
            } else {
                phase
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| at(start + j as isize) * c)
                    .sum()
            }
        };

        let mut out = Vec::with_capacity(len);
        // previous 4x point (n - 1/4), needed as the left neighbour of x[n]
        let mut prev = 0.0f64;
        for n in 0..len {
            // q[0] = n - 1/4, q[1] = n, q[2..=4] = n + 1/4.., q[5] = n + 1
            let mut q = [0.0f64; FACTOR + 2];
            q[0] = prev;
            q[1] = at(n as isize).abs();
            for (p, phase) in self.phases.iter().enumerate() {
                q[p + 2] = interpolate(n, phase).abs();
            }
            q[FACTOR + 1] = at(n as isize + 1).abs();
            prev = q[FACTOR];

            let mut peak = q[1..=FACTOR].iter().copied().fold(0.0, f64::max);
            for i in 1..=FACTOR {
                let (a, b, c) = (q[i - 1], q[i], q[i + 1]);
                let curvature = a - 2.0 * b + c;
                if b >= a && b >= c && curvature < 0.0 {
                    let offset = 0.5 * (a - c) / curvature;
                    peak = peak.max(b - 0.25 * (a - c) * offset);
                }
            }
            out.push(peak as f32);
        }
        out
    }

    /// Highest true peak in the buffer, linear.
    pub fn max_true_peak(&self, x: &[f32]) -> f32 {
        self.peak_track(x).into_iter().fold(0.0, f32::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_preserves_constants_and_samples() {
        let d = TruePeakDetector::new();
        let x = vec![0.5f32; 200];
        let track = d.peak_track(&x);
        assert!((track[100] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn finds_quarter_rate_intersample_peak() {
        // fs/4 sine at 45 degrees: samples are +-0.7071, true peak is 1.0 (+3 dB).
        let x: Vec<f32> = (0..4800)
            .map(|n| (PI / 2.0 * n as f64 + PI / 4.0).sin() as f32)
            .collect();
        let sample_peak = x.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!((sample_peak - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3);
        let tp = TruePeakDetector::new().peak_track(&x)[1000..3800]
            .iter()
            .fold(0.0f32, |m, &s| m.max(s));
        assert!((tp - 1.0).abs() < 0.002, "true peak {tp}");
    }

    #[test]
    fn high_frequency_tone_is_not_under_read() {
        // 19.7 kHz tone, amplitude 1, arbitrary phase: plain 4x under-reads.
        let x: Vec<f32> = (0..48_000)
            .map(|n| (2.0 * PI * 19_700.0 * n as f64 / 48_000.0 + 0.3).sin() as f32)
            .collect();
        let tp = TruePeakDetector::new().peak_track(&x)[1000..47_000]
            .iter()
            .fold(0.0f32, |m, &s| m.max(s));
        assert!((f64::from(tp) - 1.0).abs() < 0.01, "true peak {tp}");
    }
}
