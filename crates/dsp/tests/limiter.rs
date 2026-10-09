//! True-peak limiter: the ceiling must hold on signals designed to hide
//! peaks between samples.

mod common;

use common::*;
use dsp::config::LimiterConfig;
use dsp::limiter;
use dsp::truepeak::TruePeakDetector;

/// Concatenation of known intersample-peak torture cases, all far above
/// the ceiling.
fn nasty_signal() -> Vec<f32> {
    let quarter = (FS * 0.5) as usize;
    let mut x = Vec::new();
    // 1. fs/4 sine at 45 deg: samples +-0.99, true peak 1.4 (+2.9 dBTP).
    x.extend(sine_phase(
        12_000.0,
        1.4,
        std::f64::consts::FRAC_PI_4,
        quarter,
    ));
    // 2. Full-scale random +-1 bits: broadband, overs of several dB.
    x.extend(noise(1.0, quarter, 99).iter().map(|s| s.signum()));
    // 3. 19 kHz tone at 1.2, where plain 4x detection under-reads.
    x.extend(sine_phase(19_000.0, 1.2, 0.4, quarter));
    // 4. Repeating +1 +1 -1 -1 ... with sign flips: classic +3 dB overs.
    x.extend((0..quarter).map(|n| if (n / 2) % 2 == 0 { 1.0 } else { -1.0 }));
    // 5. Sudden 200 Hz bursts at +10 dB over full scale with hard edges.
    x.extend((0..quarter).map(|n| {
        if (n / 4800) % 2 == 0 {
            (3.0 * (2.0 * std::f64::consts::PI * 200.0 * n as f64 / FS).sin()) as f32
        } else {
            0.0
        }
    }));
    // 6. Lone full-scale clicks in silence.
    x.extend((0..quarter).map(|n| if n % 4800 == 100 { 1.0 } else { 0.0 }));
    x
}

#[test]
fn true_peak_never_exceeds_ceiling_on_nasty_signal() {
    let cfg = LimiterConfig::default();
    let mut x = nasty_signal();
    let input_tp = db(reference_true_peak(&x));
    let report = limiter::process(&mut x, FS_U32, &cfg);

    let reference = db(reference_true_peak(&x));
    let own = db(f64::from(TruePeakDetector::new().max_true_peak(&x)));
    let bs1770 = dsp::loudness::measure(&x, FS_U32, false)
        .unwrap()
        .true_peak_dbtp;
    eprintln!(
        "input TP {input_tp:+.2} dBTP -> output: reference(16x) {reference:+.3}, own(4x+parabola) {own:+.3}, \
         ebur128(BS.1770) {bs1770:+.3} dBTP; passes {}, fallback trim {:.3} dB, {}",
        report.passes, report.fallback_trim_db, report.gain_reduction
    );
    assert!(x.iter().all(|s| s.is_finite()));
    for (name, tp) in [
        ("reference 16x", reference),
        ("own", own),
        ("ebur128", bs1770),
    ] {
        assert!(
            tp <= cfg.ceiling_dbtp,
            "{name} true peak {tp:.4} dBTP > ceiling"
        );
    }
}

#[test]
fn limiter_is_transparent_below_ceiling_and_handles_silence() {
    let cfg = LimiterConfig::default();
    // Below the ceiling only the 20 kHz band-limit acts: at 1 kHz that is
    // a pure phase shift, the level is unchanged.
    let mut quiet = sine(1000.0, 0.5, 1.0);
    let orig = quiet.clone();
    let r = limiter::process(&mut quiet, FS_U32, &cfg);
    assert_eq!(r.passes, 0);
    assert!(
        (amplitude_at(&quiet[4800..], 1000.0) / amplitude_at(&orig[4800..], 1000.0) - 1.0).abs()
            < 1e-4
    );

    let mut silence = vec![0.0f32; 48_000];
    limiter::process(&mut silence, FS_U32, &cfg);
    assert!(silence.iter().all(|&s| s == 0.0));
}

/// A single loud transient: the gain must already be down when it
/// arrives (lookahead), so the transient comes out at the ceiling rather
/// than being clipped by a late reaction.
#[test]
fn lookahead_catches_a_sudden_transient() {
    let cfg = LimiterConfig::default();
    let mut x = vec![0.0f32; 9600];
    let burst = sine(1000.0, 2.0, 0.05);
    x[4800..4800 + burst.len()].copy_from_slice(&burst);
    limiter::process(&mut x, FS_U32, &cfg);
    let tp = db(reference_true_peak(&x));
    eprintln!("transient output TP {tp:+.3} dBTP");
    assert!(tp <= cfg.ceiling_dbtp);
    // Gain reduction starts before the burst (within the 5 ms lookahead).
    assert!(x[4800 - 300..4800 - 100].iter().all(|&s| s.abs() < 1e-6));
}
