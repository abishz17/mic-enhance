//! Second-order IIR filters ("biquads") designed with the RBJ Audio EQ
//! Cookbook formulas: <https://www.w3.org/TR/audio-eq-cookbook/>.
//!
//! Coefficients and filter state are `f64`; audio buffers are `f32`. The
//! extra precision matters for low-frequency, high-Q sections such as a
//! 50 Hz hum notch at 48 kHz, whose poles sit within 0.0002 of the unit
//! circle — in `f32` that filter would be noisy or even unstable.
//!
//! Every constructor normalises by `a0`, so `a0` is implicitly 1.

use std::f64::consts::PI;

/// Normalised biquad coefficients:
/// `H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coefficients {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

/// Shared intermediate values from the cookbook ("w0", "cos(w0)", "alpha").
struct Prewarp {
    cos_w0: f64,
    sin_w0: f64,
}

impl Prewarp {
    /// # Panics
    /// If `f0` is not strictly between 0 and Nyquist. Callers validate user
    /// input (see `config::ChainConfig::validate`) before getting here, so a
    /// panic means a programming error, not bad input.
    fn new(sample_rate: f64, f0: f64) -> Self {
        assert!(
            f0 > 0.0 && f0 < sample_rate / 2.0,
            "filter frequency {f0} Hz must be inside (0, {}) Hz",
            sample_rate / 2.0
        );
        let w0 = 2.0 * PI * f0 / sample_rate;
        Self {
            cos_w0: w0.cos(),
            sin_w0: w0.sin(),
        }
    }

    fn alpha_q(&self, q: f64) -> f64 {
        assert!(q > 0.0, "Q must be positive, got {q}");
        self.sin_w0 / (2.0 * q)
    }

    /// Cookbook alpha for shelves, parameterised by slope `S` (1.0 = the
    /// steepest slope that stays monotonic).
    fn alpha_shelf(&self, a: f64, slope: f64) -> f64 {
        assert!(slope > 0.0, "shelf slope must be positive, got {slope}");
        self.sin_w0 / 2.0 * ((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0).sqrt()
    }
}

impl Coefficients {
    /// Pass-through (`H(z) = 1`).
    pub const IDENTITY: Self = Self {
        b0: 1.0,
        b1: 0.0,
        b2: 0.0,
        a1: 0.0,
        a2: 0.0,
    };

    fn normalised(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Second-order low-pass. `q = 1/sqrt(2)` is Butterworth (-3 dB at `f0`).
    pub fn lowpass(sample_rate: f64, f0: f64, q: f64) -> Self {
        let p = Prewarp::new(sample_rate, f0);
        let alpha = p.alpha_q(q);
        let b1 = 1.0 - p.cos_w0;
        Self::normalised(
            b1 / 2.0,
            b1,
            b1 / 2.0,
            1.0 + alpha,
            -2.0 * p.cos_w0,
            1.0 - alpha,
        )
    }

    /// Second-order high-pass. `q = 1/sqrt(2)` is Butterworth (-3 dB at `f0`).
    pub fn highpass(sample_rate: f64, f0: f64, q: f64) -> Self {
        let p = Prewarp::new(sample_rate, f0);
        let alpha = p.alpha_q(q);
        let b0 = (1.0 + p.cos_w0) / 2.0;
        Self::normalised(
            b0,
            -(1.0 + p.cos_w0),
            b0,
            1.0 + alpha,
            -2.0 * p.cos_w0,
            1.0 - alpha,
        )
    }

    /// Band-pass with 0 dB gain at the centre frequency `f0`.
    pub fn bandpass(sample_rate: f64, f0: f64, q: f64) -> Self {
        let p = Prewarp::new(sample_rate, f0);
        let alpha = p.alpha_q(q);
        Self::normalised(
            alpha,
            0.0,
            -alpha,
            1.0 + alpha,
            -2.0 * p.cos_w0,
            1.0 - alpha,
        )
    }

    /// All-pass: unity gain everywhere, phase rotates by 360 degrees around
    /// `f0` (sharper for higher `q`).
    pub fn allpass(sample_rate: f64, f0: f64, q: f64) -> Self {
        let p = Prewarp::new(sample_rate, f0);
        let alpha = p.alpha_q(q);
        let b1 = -2.0 * p.cos_w0;
        Self::normalised(1.0 - alpha, b1, 1.0 + alpha, 1.0 + alpha, b1, 1.0 - alpha)
    }

    /// Notch: zero gain at `f0`, -3 dB bandwidth `f0 / q`.
    pub fn notch(sample_rate: f64, f0: f64, q: f64) -> Self {
        let p = Prewarp::new(sample_rate, f0);
        let alpha = p.alpha_q(q);
        let b1 = -2.0 * p.cos_w0;
        Self::normalised(1.0, b1, 1.0, 1.0 + alpha, b1, 1.0 - alpha)
    }

    /// Peaking ("bell") EQ: `gain_db` at `f0`, unity far away.
    pub fn peaking(sample_rate: f64, f0: f64, q: f64, gain_db: f64) -> Self {
        let p = Prewarp::new(sample_rate, f0);
        let alpha = p.alpha_q(q);
        let a = 10f64.powf(gain_db / 40.0);
        let b1 = -2.0 * p.cos_w0;
        Self::normalised(
            1.0 + alpha * a,
            b1,
            1.0 - alpha * a,
            1.0 + alpha / a,
            b1,
            1.0 - alpha / a,
        )
    }

    /// Low shelf: `gain_db` below `f0`, unity above, half the dB gain at `f0`.
    pub fn low_shelf(sample_rate: f64, f0: f64, slope: f64, gain_db: f64) -> Self {
        let p = Prewarp::new(sample_rate, f0);
        let a = 10f64.powf(gain_db / 40.0);
        let k = 2.0 * a.sqrt() * p.alpha_shelf(a, slope);
        let (ap1, am1, c) = (a + 1.0, a - 1.0, p.cos_w0);
        Self::normalised(
            a * (ap1 - am1 * c + k),
            2.0 * a * (am1 - ap1 * c),
            a * (ap1 - am1 * c - k),
            ap1 + am1 * c + k,
            -2.0 * (am1 + ap1 * c),
            ap1 + am1 * c - k,
        )
    }

    /// High shelf: `gain_db` above `f0`, unity below, half the dB gain at `f0`.
    pub fn high_shelf(sample_rate: f64, f0: f64, slope: f64, gain_db: f64) -> Self {
        let p = Prewarp::new(sample_rate, f0);
        let a = 10f64.powf(gain_db / 40.0);
        let k = 2.0 * a.sqrt() * p.alpha_shelf(a, slope);
        let (ap1, am1, c) = (a + 1.0, a - 1.0, p.cos_w0);
        Self::normalised(
            a * (ap1 + am1 * c + k),
            -2.0 * a * (am1 + ap1 * c),
            a * (ap1 + am1 * c - k),
            ap1 - am1 * c + k,
            2.0 * (am1 - ap1 * c),
            ap1 - am1 * c - k,
        )
    }

    /// Theoretical magnitude response at `freq` Hz, in dB, by evaluating
    /// `H(e^{jw})` directly. Tests compare measured output against this.
    pub fn magnitude_db(&self, sample_rate: f64, freq: f64) -> f64 {
        let w = 2.0 * PI * freq / sample_rate;
        let (c1, s1, c2, s2) = (w.cos(), w.sin(), (2.0 * w).cos(), (2.0 * w).sin());
        // e^{-jw} = cos w - j sin w
        let num_re = self.b0 + self.b1 * c1 + self.b2 * c2;
        let num_im = -(self.b1 * s1 + self.b2 * s2);
        let den_re = 1.0 + self.a1 * c1 + self.a2 * c2;
        let den_im = -(self.a1 * s1 + self.a2 * s2);
        let num = num_re.hypot(num_im);
        let den = den_re.hypot(den_im);
        crate::util::gain_to_db(num / den)
    }

    /// True if both poles are strictly inside the unit circle
    /// (the "stability triangle": `|a2| < 1` and `|a1| < 1 + a2`).
    pub fn is_stable(&self) -> bool {
        self.a2.abs() < 1.0 && self.a1.abs() < 1.0 + self.a2
    }
}

/// One biquad section with its running state, in transposed direct form II.
///
/// TDF-II needs only two state variables and, in floating point, has
/// better round-off behaviour than direct form II (no large internal
/// intermediate). Direct form I would be the choice if coefficients were
/// modulated per sample; none of ours are.
#[derive(Clone, Debug)]
pub struct Biquad {
    coeffs: Coefficients,
    s1: f64,
    s2: f64,
}

impl Biquad {
    pub fn new(coeffs: Coefficients) -> Self {
        Self {
            coeffs,
            s1: 0.0,
            s2: 0.0,
        }
    }

    pub fn coefficients(&self) -> Coefficients {
        self.coeffs
    }

    /// Clear the filter memory.
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    #[inline]
    pub fn process_sample(&mut self, x: f64) -> f64 {
        let c = &self.coeffs;
        let y = c.b0 * x + self.s1;
        self.s1 = c.b1 * x - c.a1 * y + self.s2;
        self.s2 = c.b2 * x - c.a2 * y;
        y
    }

    /// Filter a buffer in place.
    pub fn process(&mut self, buf: &mut [f32]) {
        for s in buf {
            *s = self.process_sample(f64::from(*s)) as f32;
        }
    }
}

/// Q of each biquad section of an even-order Butterworth filter.
fn butterworth_qs(order: usize) -> impl Iterator<Item = f64> {
    assert!(
        order >= 2 && order.is_multiple_of(2),
        "Butterworth order must be even and >= 2, got {order}"
    );
    let n = order as f64;
    (1..=order / 2).map(move |k| 1.0 / (2.0 * ((2.0 * k as f64 - 1.0) * PI / (2.0 * n)).cos()))
}

/// A chain of biquads applied one after another (their dB responses add).
#[derive(Clone, Debug, Default)]
pub struct Cascade {
    sections: Vec<Biquad>,
}

impl Cascade {
    pub fn new(coeffs: impl IntoIterator<Item = Coefficients>) -> Self {
        Self {
            sections: coeffs.into_iter().map(Biquad::new).collect(),
        }
    }

    /// Even-order Butterworth high-pass built from `order / 2` biquads.
    /// Order 4 = 24 dB/octave, -3.01 dB at `fc`.
    ///
    /// A Butterworth of order N has its poles evenly spaced on a circle;
    /// section k (1-based) gets `Q = 1 / (2 cos((2k-1) pi / (2N)))`. For
    /// N = 4 that is Q = 0.5412 and Q = 1.3066. (Two sections both at
    /// Q = 0.7071 would be a Linkwitz-Riley filter: -6 dB at `fc`.)
    ///
    /// # Panics
    /// If `order` is zero or odd.
    pub fn butterworth_highpass(sample_rate: f64, fc: f64, order: usize) -> Self {
        Self::new(butterworth_qs(order).map(|q| Coefficients::highpass(sample_rate, fc, q)))
    }

    /// Even-order Butterworth low-pass, same construction as
    /// [`Cascade::butterworth_highpass`].
    ///
    /// # Panics
    /// If `order` is zero or odd.
    pub fn butterworth_lowpass(sample_rate: f64, fc: f64, order: usize) -> Self {
        Self::new(butterworth_qs(order).map(|q| Coefficients::lowpass(sample_rate, fc, q)))
    }

    pub fn push(&mut self, coeffs: Coefficients) {
        self.sections.push(Biquad::new(coeffs));
    }

    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    pub fn len(&self) -> usize {
        self.sections.len()
    }

    pub fn coefficients(&self) -> impl Iterator<Item = Coefficients> + '_ {
        self.sections.iter().map(Biquad::coefficients)
    }

    pub fn reset(&mut self) {
        self.sections.iter_mut().for_each(Biquad::reset);
    }

    #[inline]
    pub fn process_sample(&mut self, x: f64) -> f64 {
        self.sections
            .iter_mut()
            .fold(x, |acc, section| section.process_sample(acc))
    }

    /// Filter a buffer in place (sample by sample through all sections,
    /// so intermediate values stay in `f64`).
    pub fn process(&mut self, buf: &mut [f32]) {
        for s in buf {
            *s = self.process_sample(f64::from(*s)) as f32;
        }
    }

    /// Theoretical magnitude response of the whole chain, in dB.
    pub fn magnitude_db(&self, sample_rate: f64, freq: f64) -> f64 {
        self.coefficients()
            .map(|c| c.magnitude_db(sample_rate, freq))
            .sum()
    }

    pub fn is_stable(&self) -> bool {
        self.coefficients().all(|c| c.is_stable())
    }
}

/// Zero-phase filtering: run the cascade forwards, then backwards over the
/// result. The phase shifts cancel and the magnitude response is squared
/// (dB doubled). Only possible offline, because the backward pass needs
/// the future. Used by the de-esser's band split.
pub fn filtfilt(cascade: &Cascade, buf: &mut [f32]) {
    let mut forward = cascade.clone();
    forward.reset();
    forward.process(buf);
    buf.reverse();
    let mut backward = cascade.clone();
    backward.reset();
    backward.process(buf);
    buf.reverse();
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    #[test]
    fn identity_passes_through() {
        let mut b = Biquad::new(Coefficients::IDENTITY);
        assert_eq!(b.process_sample(0.5), 0.5);
    }

    #[test]
    fn butterworth_q_values() {
        let c = Cascade::butterworth_highpass(FS, 80.0, 4);
        assert_eq!(c.len(), 2);
        assert!((c.magnitude_db(FS, 80.0) + 3.0103).abs() < 0.01);
    }

    #[test]
    #[should_panic(expected = "must be inside")]
    fn rejects_frequency_above_nyquist() {
        let _ = Coefficients::lowpass(FS, 30_000.0, 0.7);
    }
}
