//! Adaptive tone correction: accuracy of the realised EQ, and whether
//! differently coloured versions of the same voice end up sounding alike.

mod common;

use common::*;
use dsp::biquad::Coefficients;
use dsp::config::ToneConfig;
use dsp::io;
use dsp::tone::{self, BANDS, CENTRES_HZ, ToneOutcome};
use std::path::PathBuf;

/// The algorithm at full strength (the product default is 0 = off).
fn full_strength() -> ToneConfig {
    ToneConfig {
        strength: 1.0,
        ..ToneConfig::default()
    }
}

/// clean.wav and the VoiceBank p232 reference clips joined into one ~30 s
/// take (real human speech; each clip alone is < 8 s), whichever exist.
/// Empty when neither is present: the tests then have nothing to check.
fn speech_sources() -> Vec<(String, Vec<f32>)> {
    let mut out = Vec::new();
    if let Some(path) = smoke_path("clean.wav") {
        out.push((
            "smoke/clean.wav".to_string(),
            io::load_wav(&path, Default::default()).unwrap().samples,
        ));
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vbref");
    let mut joined = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "wav"))
            .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("p232"))
            .collect();
        paths.sort();
        for p in paths {
            joined.extend(io::load_wav(&p, Default::default()).unwrap().samples);
            joined.extend(std::iter::repeat_n(0.0, 24_000)); // 0.5 s pause
        }
    }
    if joined.len() > 20 * 48_000 {
        out.push(("vbref/p232_* joined".to_string(), joined));
    }
    out
}

fn colour(x: &[f32], c: Coefficients) -> Vec<f32> {
    let mut y = x.to_vec();
    dsp::biquad::Biquad::new(c).process(&mut y);
    y
}

fn ltas(x: &[f32]) -> [f64; BANDS] {
    tone::analyse(x, FS, &full_strength()).unwrap().measured_db
}

fn fmt_curve(c: &[f64; BANDS]) -> String {
    c.iter()
        .map(|g| format!("{g:+.1}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn colourings() -> [(&'static str, Coefficients); 3] {
    [
        ("dark", Coefficients::high_shelf(FS, 3000.0, 1.0, -6.0)),
        ("bright", Coefficients::high_shelf(FS, 3000.0, 1.0, 6.0)),
        ("boomy", Coefficients::peaking(FS, 200.0, 1.0, 6.0)),
    ]
}

/// The source voice EQ'd exactly onto the target (raw deviation, no
/// smoothing/deadband/limits), twice to absorb frame-selection shifts.
fn neutralise(source: &[f32]) -> Vec<f32> {
    let mut neutral = source.to_vec();
    for _ in 0..2 {
        let a = tone::analyse(&neutral, FS, &full_strength()).unwrap();
        tone::fit_bank(&tone::raw_deviation(&a), FS).process(&mut neutral);
    }
    neutral
}

fn max_abs(c: &[f64]) -> f64 {
    c.iter().fold(0.0f64, |m, g| m.max(g.abs()))
}

/// An input that is already neutral must be left alone (< 0.5 dB).
#[test]
fn neutral_input_gets_under_half_a_db() {
    for (name, source) in speech_sources() {
        let neutral = neutralise(&source);
        let analysis = tone::analyse(&neutral, FS, &full_strength()).unwrap();
        let c = tone::correction(&analysis, &full_strength());
        eprintln!(
            "{name}: neutral input correction {} (worst {:.2} dB)",
            fmt_curve(&c),
            max_abs(&c)
        );
        assert!(max_abs(&c) < 0.5, "{name}");
    }
}

/// A neutral voice, coloured +-6 dB: in the affected bands between 200 Hz
/// and 8 kHz (colour >= 2 dB), the deviation from target must shrink to
/// about the deadband. Full convergence is deliberately *not* the goal:
/// the +-2 dB deadband leaves 2 dB in every band by design, so the best
/// possible reduction is ~55% for the shelves and ~45% for the 200 Hz
/// bell (whose 315 Hz band is only 2.7 dB off). Asserted: mean residual
/// <= deadband + 0.75 dB (smoothing/caps slack), and it always shrinks.
#[test]
fn colourings_shrink_to_the_deadband() {
    let mut results = Vec::new();
    for (name, source) in speech_sources() {
        let neutral = neutralise(&source);
        let target = tone::target_db();
        for (label, c) in colourings() {
            let coloured = colour(&neutral, c);
            let before = ltas(&coloured);
            let mut corrected = coloured.clone();
            tone::process(&mut corrected, FS, &full_strength());
            let after = ltas(&corrected);
            let bands: Vec<usize> = (0..BANDS)
                .filter(|&b| (200.0..=8000.0).contains(&CENTRES_HZ[b]))
                .filter(|&b| c.magnitude_db(FS, CENTRES_HZ[b]).abs() >= 2.0)
                .collect();
            let mean_dev = |l: &[f64; BANDS]| {
                bands.iter().map(|&b| (l[b] - target[b]).abs()).sum::<f64>() / bands.len() as f64
            };
            let (d0, d1) = (mean_dev(&before), mean_dev(&after));
            eprintln!(
                "{name} {label:6}: mean |deviation| in {} affected bands {d0:.2} -> {d1:.2} dB ({:.0}% removed)",
                bands.len(),
                100.0 * (1.0 - d1 / d0)
            );
            results.push((format!("{name} {label}"), d0, d1));
        }
    }
    let limit = ToneConfig::default().deadband_db + 0.75;
    for (what, d0, d1) in results {
        assert!(
            d1 < d0 && d1 <= limit,
            "{what}: {d0:.2} -> {d1:.2} dB (limit {limit})"
        );
    }
}

/// Nothing below 200 Hz is touched, even when the boomy colouring sits
/// right at 200 Hz, and a naturally low voice (+6 dB at 100-160 Hz) keeps
/// its low end.
#[test]
fn nothing_changes_below_200_hz() {
    let Some((_, source)) = speech_sources().into_iter().next() else {
        return;
    };
    let neutral = neutralise(&source);
    let cases = [
        ("boomy", Coefficients::peaking(FS, 200.0, 1.0, 6.0)),
        ("low voice", Coefficients::peaking(FS, 125.0, 1.6, 6.0)),
    ];
    for (label, c) in cases {
        let coloured = colour(&neutral, c);
        let ToneOutcome::Applied { correction_db, .. } =
            tone::process(&mut coloured.clone(), FS, &full_strength())
        else {
            panic!("expected correction");
        };
        let bank = tone::fit_bank(&correction_db, FS);
        // Sweep 20-160 Hz (up to the band 1/3 octave under 200 Hz) in
        // 1 Hz steps on the realised response. Between 160 and 200 Hz the
        // skirt of the 200 Hz bell is unavoidable for any smooth filter.
        for f in [100.0, 140.0, 160.0, 180.0, 190.0] {
            eprintln!(
                "  {label} realised at {f} Hz: {:+.2} dB",
                bank.magnitude_db(FS, f)
            );
        }
        let worst = (20..=160)
            .map(|f| bank.magnitude_db(FS, f64::from(f)).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "{label}: correction {} ; worst change 20-160 Hz {worst:.3} dB",
            fmt_curve(&correction_db)
        );
        assert!(
            correction_db
                .iter()
                .zip(CENTRES_HZ)
                .filter(|(_, f)| *f < 200.0)
                .all(|(g, _)| *g == 0.0)
        );
        assert!(worst < 0.5, "{label}: {worst:.3} dB below 160 Hz");
        if label == "low voice" {
            let low: Vec<f64> = (0..3).map(|b| ltas(&coloured)[b]).collect();
            let mut out = coloured.clone();
            tone::process(&mut out, FS, &full_strength());
            let low_after: Vec<f64> = (0..3).map(|b| ltas(&out)[b]).collect();
            eprintln!(
                "low voice 100/125/160 Hz before {low:.1?} after {low_after:.1?} (relative to 500 Hz-2 kHz)"
            );
        }
    }
}

#[test]
fn strength_scales_the_curve() {
    let Some((_, source)) = speech_sources().into_iter().next() else {
        return;
    };
    let coloured = colour(
        &neutralise(&source),
        Coefficients::high_shelf(FS, 3000.0, 1.0, 6.0),
    );
    let analysis = tone::analyse(&coloured, FS, &full_strength()).unwrap();
    let full = tone::correction(&analysis, &full_strength());
    let half = tone::correction(
        &analysis,
        &ToneConfig {
            strength: 0.5,
            ..full_strength()
        },
    );
    let off = tone::correction(
        &analysis,
        &ToneConfig {
            strength: 0.0,
            ..full_strength()
        },
    );
    for b in 0..BANDS {
        assert!((half[b] - 0.5 * full[b]).abs() < 1e-12);
        assert_eq!(off[b], 0.0);
    }
    assert!(max_abs(&full) > 2.0);
}

/// The filter bank must realise the intended curve within 0.5 dB at each
/// band centre: analytically, and measured with real sines.
#[test]
fn realised_response_matches_intended_curve() {
    // Bands below 200 Hz are never corrected, so curves start at 200 Hz.
    let low = |b: usize| CENTRES_HZ[b] < 200.0;
    let shape = |b: usize, g: f64| if low(b) { 0.0 } else { g };
    let curves: [[f64; BANDS]; 3] = [
        std::array::from_fn(|b| {
            shape(
                b,
                if b < 6 {
                    -4.0
                } else if b > 14 {
                    3.0
                } else {
                    0.0
                },
            )
        }),
        std::array::from_fn(|b| shape(b, 6.0 * ((b as f64) * 0.7).sin())),
        std::array::from_fn(|b| shape(b, if b % 2 == 0 { 2.0 } else { -2.0 })),
    ];
    for wanted in curves {
        let bank = tone::fit_bank(&wanted, FS);
        let mut worst = 0.0f64;
        for (b, &f) in CENTRES_HZ.iter().enumerate().filter(|(b, _)| !low(*b)) {
            let analytic = bank.magnitude_db(FS, f);
            let measured = measured_gain_db(f, |buf| bank.clone().process(buf));
            worst = worst
                .max((analytic - wanted[b]).abs())
                .max((measured - wanted[b]).abs());
        }
        eprintln!(
            "curve {}: worst realised error {worst:.3} dB",
            fmt_curve(&wanted)
        );
        assert!(worst < 0.5, "worst error {worst:.3} dB");
    }
}

#[test]
fn short_speech_falls_back_to_no_correction() {
    let Some(path) = smoke_path("clean.wav") else {
        return;
    };
    let mut x = io::load_wav(&path, Default::default()).unwrap().samples;
    x.truncate(6 * 48_000);
    let orig = x.clone();
    match tone::process(&mut x, FS, &full_strength()) {
        ToneOutcome::Skipped(why) => {
            eprintln!("skipped: {why}");
            assert!(why.contains("speech"));
        }
        other => panic!("expected skip, got {other:?}"),
    }
    assert_eq!(x, orig);
}

/// Band-limited input (AirPods-like, nothing above 7 kHz but hiss): no
/// boost may land above the detected bandwidth.
#[test]
fn never_boosts_above_bandwidth() {
    let Some(path) = smoke_path("degraded_bluetooth.wav") else {
        return;
    };
    let source = io::load_wav(&path, Default::default()).unwrap().samples;
    let mut x = source;
    let ToneOutcome::Applied {
        analysis,
        correction_db,
        ..
    } = tone::process(&mut x, FS, &full_strength())
    else {
        panic!("expected correction on 15 s of speech");
    };
    let top = analysis.top_band.unwrap();
    eprintln!(
        "bluetooth: bandwidth up to {} Hz band, correction {}",
        CENTRES_HZ[top],
        fmt_curve(&correction_db)
    );
    assert!(CENTRES_HZ[top] <= 8000.0);
    assert!(correction_db[top + 1..].iter().all(|&g| g <= 0.0));
}
