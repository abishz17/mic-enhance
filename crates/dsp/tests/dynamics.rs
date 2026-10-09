//! Compressor, expander and de-esser behaviour on synthetic signals.

mod common;

use common::*;
use dsp::config::{CompressorConfig, DeesserConfig, ExpanderConfig};
use dsp::dynamics::compressor::{self, static_curve};
use dsp::dynamics::{deesser, expander};
use dsp::util::rms_db;

/// Feed steady 1 kHz sines at many levels; the settled output level must
/// follow the static curve, and above the knee the slope must be 1/ratio.
#[test]
fn compressor_static_curve_matches_ratio() {
    let cfg = CompressorConfig {
        auto_makeup: false,
        makeup_db: 0.0,
        ..CompressorConfig::default()
    };
    let mut worst = 0.0f64;
    let mut out_at = |level_db: f64| -> f64 {
        let amp = 10f64.powf(level_db / 20.0) * std::f64::consts::SQRT_2; // RMS -> peak
        let mut x = sine(1000.0, amp, 1.0);
        compressor::process(&mut x, FS_U32, &cfg).unwrap();
        let out = rms_db(&x[24_000..]);
        let expected = static_curve(level_db, cfg.threshold_db, cfg.ratio, cfg.knee_db);
        worst = worst.max((out - expected).abs());
        out
    };
    let levels: Vec<f64> = (0..=10).map(|i| -50.0 + 4.5 * f64::from(i)).collect();
    let outs: Vec<f64> = levels.iter().map(|&l| out_at(l)).collect();
    eprintln!("compressor static curve (in -> out dB RMS):");
    for (l, o) in levels.iter().zip(&outs) {
        eprintln!("  {l:6.1} -> {o:6.2}");
    }
    let slope = (out_at(-10.0) - out_at(-20.0)) / 10.0;
    eprintln!(
        "slope above knee: {slope:.4} (1/ratio = {:.4}), worst curve error {worst:.3} dB",
        1.0 / cfg.ratio
    );
    assert!((slope - 1.0 / cfg.ratio).abs() < 0.01, "slope {slope}");
    assert!(
        worst < 0.15,
        "worst deviation from static curve {worst:.3} dB"
    );
}

#[test]
fn compressor_auto_makeup_is_loudness_neutral() {
    let mut x = add(&sine(200.0, 0.3, 4.0), &noise(0.05, 4 * 48_000, 7));
    let before = dsp::loudness::integrated_lufs(&x, FS_U32, false)
        .unwrap()
        .unwrap();
    let r = compressor::process(&mut x, FS_U32, &CompressorConfig::default()).unwrap();
    let after = dsp::loudness::integrated_lufs(&x, FS_U32, false)
        .unwrap()
        .unwrap();
    assert!(r.gain_reduction.max_reduction_db > 3.0);
    assert!(
        (before - after).abs() < 0.05,
        "before {before:.2}, after {after:.2}"
    );
}

fn expander_cfg() -> ExpanderConfig {
    ExpanderConfig {
        auto_threshold: false,
        threshold_db: -40.0,
        ..ExpanderConfig::default()
    }
}

/// 1 kHz bursts (300 ms at -20 dBFS) separated by 1 s of -60 dBFS room
/// tone: exactly one open and one close per burst, full level inside the
/// bursts, attenuation by about the range deep in the gaps.
#[test]
fn expander_does_not_chatter_on_tone_bursts() {
    let on = 14_400; // 300 ms
    let off = 48_000; // 1 s
    let period = on + off;
    let mut x = noise(0.001 * 3f64.sqrt(), off + 5 * period, 3); // ~-60 dBFS RMS
    let burst = sine(1000.0, 0.1 * std::f64::consts::SQRT_2, 0.3);
    for b in 0..5 {
        let start = off + b * period;
        for (i, &s) in burst.iter().enumerate() {
            x[start + i] += s;
        }
    }
    let input = x.clone();
    let r = expander::process(&mut x, FS_U32, &expander_cfg());
    eprintln!(
        "expander on 5 bursts: {} transitions, {}",
        r.transitions, r.gain_reduction
    );
    assert_eq!(r.transitions, 10, "one open + one close per burst");

    // Inside a burst (after 20 ms) the gain is unity.
    let burst2 = off + period + 1000..off + period + on - 1000;
    assert!((rms_db(&x[burst2.clone()]) - rms_db(&input[burst2])).abs() < 0.05);
    // Last 300 ms of a gap: room tone is down by about the 8 dB range.
    let gap_end = off + 2 * period;
    let gap = gap_end - 14_400..gap_end - 600;
    let cut = rms_db(&input[gap.clone()]) - rms_db(&x[gap]);
    eprintln!("gap attenuation {cut:.2} dB");
    assert!(cut > 7.0 && cut < 8.5, "gap attenuation {cut:.2} dB");
}

/// A level wobbling +-1.5 dB around the threshold (inside the 4 dB
/// hysteresis band) must not toggle the gate at all once open.
#[test]
fn expander_hysteresis_ignores_wobble_at_threshold() {
    let len = 3 * 48_000;
    let x: Vec<f32> = (0..len)
        .map(|n| {
            let t = n as f64 / FS;
            let level_db = -40.0 + 1.5 * (2.0 * std::f64::consts::PI * 5.0 * t).sin() + 0.5;
            let amp = 10f64.powf(level_db / 20.0) * std::f64::consts::SQRT_2;
            (amp * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()) as f32
        })
        .collect();
    let mut y = x.clone();
    let r = expander::process(&mut y, FS_U32, &expander_cfg());
    eprintln!("expander wobble: {} transitions", r.transitions);
    assert!(r.transitions <= 1, "{} transitions", r.transitions);
}

/// A word with a slowly decaying tail (60 dB over 400 ms) must not be
/// chopped: the gate closes once, after the tail, and only well after
/// the level crosses the threshold (hold).
#[test]
fn expander_keeps_word_tails() {
    let word = 0.3;
    let tail = 0.4;
    let len = ((word + tail + 0.5) * FS) as usize;
    let x: Vec<f32> = (0..len)
        .map(|n| {
            let t = n as f64 / FS;
            let env_db = if t < word {
                -20.0
            } else {
                -20.0 - 150.0 * (t - word)
            };
            let amp = 10f64.powf(env_db.max(-120.0) / 20.0) * std::f64::consts::SQRT_2;
            (amp * (2.0 * std::f64::consts::PI * 300.0 * t).sin()) as f32
        })
        .collect();
    let mut y = x.clone();
    let r = expander::process(&mut y, FS_U32, &expander_cfg());
    assert_eq!(r.transitions, 2);
    // The tail between -20 and -44 dB (first 160 ms) must pass at > -1 dB.
    let from = ((word + 0.02) * FS) as usize;
    let to = ((word + 0.16) * FS) as usize;
    let loss = rms_db(&x[from..to]) - rms_db(&y[from..to]);
    assert!(loss < 1.0, "tail lost {loss:.2} dB");
}

#[test]
fn deesser_cuts_sibilant_band_only() {
    let cfg = DeesserConfig::default();
    // 6.5 kHz "ess" at -20 dBFS, well above the -32 dB threshold, on top of
    // a 300 Hz vowel.
    let mut x = add(&sine(300.0, 0.3, 1.0), &sine(6500.0, 0.14, 1.0));
    let input = x.clone();
    let r = deesser::process(&mut x, FS_U32, &cfg);
    let tail = 24_000..;
    let hf = 20.0
        * (amplitude_at(&x[tail.clone()], 6500.0) / amplitude_at(&input[tail.clone()], 6500.0))
            .log10();
    let lf =
        20.0 * (amplitude_at(&x[tail.clone()], 300.0) / amplitude_at(&input[tail], 300.0)).log10();
    eprintln!(
        "de-esser: 6.5 kHz {hf:.2} dB, 300 Hz {lf:.3} dB, {}",
        r.gain_reduction
    );
    assert!(
        hf < -4.0 && hf >= -cfg.max_reduction_db - 0.05,
        "HF change {hf:.2}"
    );
    assert!(lf.abs() < 0.05, "LF change {lf:.3}");

    // Quiet HF below threshold: untouched.
    let mut quiet = sine(6500.0, 0.002, 1.0);
    let quiet_in = quiet.clone();
    deesser::process(&mut quiet, FS_U32, &cfg);
    assert!((rms_db(&quiet[24_000..]) - rms_db(&quiet_in[24_000..])).abs() < 0.05);
}
