//! Biquad responses measured on real signals and compared with theory.

mod common;

use common::*;
use dsp::biquad::{Biquad, Cascade, Coefficients};
use realfft::RealFftPlanner;

fn measure_coeffs(c: Coefficients, freq: f64) -> f64 {
    measured_gain_db(freq, |buf| Biquad::new(c).process(buf))
}

#[test]
fn butterworth_hpf_is_minus_3_db_at_cutoff_and_24_db_per_octave() {
    let hpf = Cascade::butterworth_highpass(FS, 80.0, 4);
    let at = |f| measured_gain_db(f, |buf| hpf.clone().process(buf));
    let fc = at(80.0);
    assert!((fc + 3.01).abs() < 0.05, "at fc: {fc:.3} dB");
    // 4th-order Butterworth: |H|^2 = 1 / (1 + (fc/f)^8)
    for f in [20.0, 40.0, 160.0, 1000.0] {
        let theory = -10.0 * (1.0 + (80.0f64 / f).powi(8)).log10();
        let got = at(f);
        assert!(
            (got - theory).abs() < 0.1,
            "{f} Hz: got {got:.3}, theory {theory:.3}"
        );
    }
    // One octave below: ~24 dB down (asymptote, exact value -24.1 dB).
    assert!((at(40.0) - at(80.0) + 21.1).abs() < 0.2);
}

#[test]
fn single_section_hpf_and_lpf_minus_3_db() {
    let q = std::f64::consts::FRAC_1_SQRT_2;
    for fc in [100.0, 1000.0, 8000.0] {
        let hp = measure_coeffs(Coefficients::highpass(FS, fc, q), fc);
        let lp = measure_coeffs(Coefficients::lowpass(FS, fc, q), fc);
        assert!((hp + 3.01).abs() < 0.05, "HPF {fc}: {hp:.3}");
        assert!((lp + 3.01).abs() < 0.05, "LPF {fc}: {lp:.3}");
    }
}

#[test]
fn bell_gain_at_centre_within_0_1_db() {
    for (f0, q, gain) in [
        (300.0, 1.0, -2.5),
        (4000.0, 1.0, 2.0),
        (1000.0, 4.0, 12.0),
        (100.0, 0.7, -9.0),
    ] {
        let got = measure_coeffs(Coefficients::peaking(FS, f0, q, gain), f0);
        assert!(
            (got - gain).abs() < 0.1,
            "bell {f0} Hz {gain} dB: measured {got:.3}"
        );
        // and unity far away
        let far = measure_coeffs(
            Coefficients::peaking(FS, f0, q, gain),
            f0 * 16.0_f64.min(20_000.0 / f0),
        );
        assert!(far.abs() < 0.5, "bell {f0} far: {far:.3}");
    }
}

#[test]
fn shelves_reach_full_gain_and_half_gain_at_corner() {
    let hs = Coefficients::high_shelf(FS, 11_000.0, 1.0, 1.5);
    assert!((measure_coeffs(hs, 11_000.0) - 0.75).abs() < 0.05);
    assert!((measure_coeffs(hs, 20_000.0) - 1.5).abs() < 0.15);
    assert!(measure_coeffs(hs, 1000.0).abs() < 0.05);

    let ls = Coefficients::low_shelf(FS, 200.0, 1.0, -6.0);
    assert!((measure_coeffs(ls, 200.0) + 3.0).abs() < 0.05);
    assert!((measure_coeffs(ls, 30.0) + 6.0).abs() < 0.15);
    assert!(measure_coeffs(ls, 5000.0).abs() < 0.05);
}

#[test]
fn notch_and_bandpass() {
    let notch = Coefficients::notch(FS, 1000.0, 10.0);
    assert!(measure_coeffs(notch, 1000.0) < -60.0);
    // About -3 dB at f0 + bandwidth/2 (bandwidth = f0 / Q = 100 Hz).
    let edge = measure_coeffs(notch, 1050.0);
    assert!((edge - notch.magnitude_db(FS, 1050.0)).abs() < 0.05);
    assert!((edge + 3.0).abs() < 0.3, "notch edge {edge:.2}");
    let bp = Coefficients::bandpass(FS, 6325.0, 2.1);
    assert!(measure_coeffs(bp, 6325.0).abs() < 0.05);
}

#[test]
fn allpass_is_flat() {
    let ap = Coefficients::allpass(FS, 300.0, 0.5);
    for f in [30.0, 300.0, 3000.0, 15_000.0] {
        assert!(measure_coeffs(ap, f).abs() < 0.01, "{f} Hz");
    }
}

/// Full-band check: FFT of the impulse response vs. the analytic
/// `magnitude_db` at every bin from 20 Hz to 20 kHz.
#[test]
fn impulse_response_fft_matches_theory() {
    let filters = [
        Coefficients::highpass(FS, 80.0, 0.5412),
        Coefficients::peaking(FS, 300.0, 1.0, -2.5),
        Coefficients::peaking(FS, 4000.0, 1.0, 2.0),
        Coefficients::high_shelf(FS, 11_000.0, 1.0, 1.5),
        Coefficients::bandpass(FS, 6325.0, 2.1),
        Coefficients::allpass(FS, 300.0, 0.5),
    ];
    let n = 1 << 16;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    for c in filters {
        let mut b = Biquad::new(c);
        let mut ir: Vec<f64> = (0..n)
            .map(|i| b.process_sample(if i == 0 { 1.0 } else { 0.0 }))
            .collect();
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut ir, &mut spectrum).unwrap();
        let mut worst = 0.0f64;
        for (k, bin) in spectrum.iter().enumerate() {
            let f = k as f64 * FS / n as f64;
            if !(20.0..=20_000.0).contains(&f) {
                continue;
            }
            let measured = 20.0 * bin.norm().max(1e-12).log10();
            let theory = c.magnitude_db(FS, f);
            if theory > -60.0 {
                worst = worst.max((measured - theory).abs());
            }
        }
        assert!(worst < 0.01, "{c:?}: worst deviation {worst:.4} dB");
    }
}

#[test]
fn filters_are_stable_across_parameter_space() {
    for f in [
        10.0, 20.0, 50.0, 80.0, 300.0, 1000.0, 4000.0, 11_000.0, 20_000.0, 23_500.0,
    ] {
        for q in [
            0.1,
            0.5,
            std::f64::consts::FRAC_1_SQRT_2,
            1.0,
            5.0,
            25.0,
            100.0,
        ] {
            for c in [
                Coefficients::lowpass(FS, f, q),
                Coefficients::highpass(FS, f, q),
                Coefficients::bandpass(FS, f, q),
                Coefficients::notch(FS, f, q),
                Coefficients::peaking(FS, f, q, 12.0),
                Coefficients::peaking(FS, f, q, -12.0),
            ] {
                assert!(c.is_stable(), "unstable: f={f} q={q} {c:?}");
            }
        }
        for gain in [-12.0, -1.5, 1.5, 12.0] {
            assert!(Coefficients::high_shelf(FS, f, 1.0, gain).is_stable());
            assert!(Coefficients::low_shelf(FS, f, 1.0, gain).is_stable());
        }
    }
    // Stability in practice: the impulse response of a sharp low notch decays.
    let mut b = Biquad::new(Coefficients::notch(FS, 50.0, 25.0));
    let tail: f64 = (0..10 * 48_000)
        .map(|i| b.process_sample(if i == 0 { 1.0 } else { 0.0 }))
        .skip(9 * 48_000)
        .map(f64::abs)
        .fold(0.0, f64::max);
    assert!(tail < 1e-9, "tail {tail:e}");
}

#[test]
fn no_nan_on_silence_or_full_scale() {
    let mut chain = Cascade::butterworth_highpass(FS, 80.0, 4);
    chain.push(Coefficients::notch(FS, 50.0, 25.0));
    let mut silence = vec![0.0f32; 48_000];
    chain.process(&mut silence);
    assert!(silence.iter().all(|&s| s == 0.0));
    // Full-scale square wave (worst case for overshoot) stays finite.
    let mut square: Vec<f32> = (0..48_000)
        .map(|n| if (n / 240) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    chain.reset();
    chain.process(&mut square);
    assert!(square.iter().all(|s| s.is_finite()));
}
