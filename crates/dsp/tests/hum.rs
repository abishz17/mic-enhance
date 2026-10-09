//! Hum notch: deep on the hum, transparent on the speech band in between.

mod common;

use common::*;
use dsp::config::{HumConfig, HumMode};
use dsp::hum;

/// 50 Hz hum (+ 100/150/200 Hz harmonics) over noise, plus probe tones in
/// the 100-200 Hz speech band at least one notch bandwidth away from every
/// notch ("beyond the notches"). Notch -3 dB widths are 2k Hz for harmonic
/// k (constant Q 25), i.e. 4, 6 and 8 Hz at 100, 150 and 200 Hz.
#[test]
fn removes_50_hz_by_30_db_and_keeps_speech_band() {
    let secs = 6.0;
    let len = (secs * FS) as usize;
    let probes = [
        110.0, 115.0, 125.0, 135.0, 140.0, 160.0, 165.0, 175.0, 185.0, 190.0,
    ];
    let mut x = add(&sine(50.0, 0.1, secs), &noise(0.001, len, 11));
    x = add(&x, &sine(100.0, 0.03, secs));
    x = add(&x, &sine(150.0, 0.02, secs));
    x = add(&x, &sine(200.0, 0.01, secs));
    for f in probes {
        x = add(&x, &sine(f, 0.01, secs));
    }

    let cfg = HumConfig::default();
    let report = hum::detect(&x, FS, &cfg);
    let f0 = report.fundamental_hz.expect("hum detected");
    let harmonics: Vec<u32> = report.lines.iter().map(|l| l.harmonic).collect();
    eprintln!("detected f0 = {f0:.4} Hz, harmonics {harmonics:?}");
    assert!((f0 - 50.0).abs() < 0.005);
    assert_eq!(harmonics, vec![1, 2, 3, 4]);

    let bank = hum::notch_bank(&report, FS, &cfg);
    let mut y = x.clone();
    bank.clone().process(&mut y);
    let settled = 2 * 48_000..; // notch at Q 25 rings for ~0.5 s

    let cut = |f: f64| {
        20.0 * (amplitude_at(&y[settled.clone()], f) / amplitude_at(&x[settled.clone()], f)).log10()
    };
    let hum_cut = cut(50.0);
    eprintln!(
        "50 Hz: {hum_cut:.1} dB, 100 Hz: {:.1} dB, 150 Hz: {:.1} dB",
        cut(100.0),
        cut(150.0)
    );
    assert!(hum_cut < -30.0, "50 Hz only reduced by {hum_cut:.1} dB");
    assert!(cut(100.0) < -30.0 && cut(150.0) < -30.0 && cut(200.0) < -30.0);

    for f in probes {
        let change = cut(f);
        let theory = bank.magnitude_db(FS, f);
        eprintln!("probe {f} Hz: {change:+.3} dB (theory {theory:+.3})");
        assert!(change > -1.0, "{f} Hz lost {change:.2} dB");
        assert!((change - theory).abs() < 0.02);
    }

    // Analytic sweep, 0.1 Hz steps over 100-200 Hz, skipping +-1 notch
    // bandwidth around each notch: the cascade never costs more than 1 dB.
    let worst = (1000..=2000)
        .map(|i| f64::from(i) / 10.0)
        .filter(|&f| {
            report
                .lines
                .iter()
                .all(|l| (f - l.freq_hz).abs() >= 2.0 * f64::from(l.harmonic))
        })
        .map(|f| bank.magnitude_db(FS, f))
        .fold(0.0f64, f64::min);
    eprintln!("worst 100-200 Hz response outside the notches: {worst:.3} dB");
    assert!(worst > -1.1, "worst {worst:.3} dB");
}

#[test]
fn nothing_is_notched_without_hum() {
    let x = add(&sine(125.0, 0.1, 5.0), &noise(0.01, 5 * 48_000, 5));
    let report = hum::detect(&x, FS, &HumConfig::default());
    assert_eq!(report.fundamental_hz, None);
    assert!(hum::notch_bank(&report, FS, &HumConfig::default()).is_empty());
}

#[test]
fn off_mode_skips_detection() {
    let x = sine(50.0, 0.1, 5.0);
    let cfg = HumConfig {
        mode: HumMode::Off,
        ..HumConfig::default()
    };
    assert_eq!(hum::detect(&x, FS, &cfg).fundamental_hz, None);
}

#[test]
fn smoke_file_hum_is_detected_as_50_hz() {
    let (Some(hum_path), Some(clean_path)) = (
        smoke_path("degraded_noise_hum_room.wav"),
        smoke_path("clean.wav"),
    ) else {
        return;
    };
    let audio = dsp::io::load_wav(&hum_path, Default::default()).unwrap();
    let report = hum::detect(&audio.samples, FS, &HumConfig::default());
    eprintln!("smoke hum report: {report:?}");
    let f0 = report
        .fundamental_hz
        .expect("smoke file contains 50 Hz hum");
    assert!((f0 - 50.0).abs() < 0.5);
    let clean = dsp::io::load_wav(&clean_path, Default::default()).unwrap();
    assert_eq!(
        hum::detect(&clean.samples, FS, &HumConfig::default()).fundamental_hz,
        None
    );
}
