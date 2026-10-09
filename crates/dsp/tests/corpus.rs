//! Checks on real recordings placed in `corpus/` (any `*.wav`). The corpus
//! is git-ignored and private, so these tests skip when it is absent.

mod common;

use common::*;
use dsp::config::{ExpanderConfig, HumConfig, PostConfig, PreConfig};
use dsp::{chain, hum, io, pre};
use std::path::PathBuf;

/// Every `corpus/*.wav`, sorted, as (file name, path). Empty (with a skip
/// message) when the folder is missing.
fn corpus() -> Vec<(String, PathBuf)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
    let mut files: Vec<(String, PathBuf)> = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "wav"))
                .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), p))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    if files.is_empty() {
        eprintln!("skipping: no recordings in corpus/ (private, not in the repo)");
    }
    files
}

/// Real laptop-mic recordings measure 50 Hz at +0.6..+2.6 dB and 60 Hz at
/// +3..+5 dB above the local floor: negligible, so no notch may be applied.
#[test]
fn hum_auto_detect_does_not_trigger_on_corpus() {
    for (name, path) in corpus() {
        let audio = io::load_wav(&path, Default::default()).unwrap();
        let report = hum::detect(&audio.samples, FS, &HumConfig::default());
        eprintln!("{name}: {report:?}");
        assert_eq!(report.fundamental_hz, None, "{name}: false hum detection");
    }
}

/// AirPods-style on-device gating sets room tone to exact digital zero. The auto threshold must
/// come from the non-zero frames, never from -inf / -240 dB.
#[test]
fn expander_threshold_ignores_digital_zeros() {
    for (name, path) in corpus() {
        let mut x = io::load_wav(&path, Default::default()).unwrap().samples;
        let zeros = x.iter().filter(|&&s| s == 0.0).count();
        pre::process(&mut x, FS, &PreConfig::default());
        dsp::loudness::align(&mut x, FS_U32, -23.0, 60.0).unwrap();
        let t = dsp::dynamics::expander::auto_threshold(
            &x,
            FS,
            ExpanderConfig::default().auto_margin_db,
        );
        eprintln!("{name}: {zeros} exact-zero samples, auto threshold {t:?}");
        let t = t.expect("speech present");
        assert!((-80.0..=-20.0).contains(&t));
    }
}

#[test]
fn corpus_hits_loudness_and_true_peak_targets() {
    for (name, path) in corpus() {
        let mut x = io::load_wav(&path, Default::default()).unwrap().samples;
        let input = dsp::loudness::measure(&x, FS_U32, false).unwrap();
        pre::process(&mut x, FS, &PreConfig::default());
        let r = chain::process(&mut x, FS_U32, &PostConfig::default(), |_, _| Ok(())).unwrap();
        let m = r.output.unwrap();
        let lim = r.limiter.unwrap().gain_reduction;
        eprintln!(
            "{name}: in {input} -> out {m}\n    compressor {}\n    limiter {lim}",
            r.compressor.unwrap().gain_reduction
        );
        // Regression guard: the limiter should only shave peaks, not do the
        // levelling. Max was 9.8 dB before the phase rotator, 4.4-5.0 dB
        // after it, 3.9-7.9 dB once tone correction removes LF boom (less
        // LF energy = higher crest factor at the same loudness).
        assert!(lim.max_reduction_db < 8.5, "{name}: limiter {lim}");
        assert!(x.iter().all(|s| s.is_finite()));
        assert!((m.integrated_lufs.unwrap() + 14.0).abs() <= 0.3);
        assert!(m.true_peak_dbtp <= -1.0);
    }
}

/// Tone correction on the real recordings: report it, and a band-limited
/// take (e.g. AirPods, nothing above ~12 kHz) must get no boost above 11 kHz.
#[test]
fn corpus_tone_correction() {
    use dsp::tone::{CENTRES_HZ, ToneOutcome};
    for (name, path) in corpus() {
        let mut x = io::load_wav(&path, Default::default()).unwrap().samples;
        pre::process(&mut x, FS, &PreConfig::default());
        dsp::loudness::align(&mut x, FS_U32, -23.0, 60.0).unwrap();
        let ToneOutcome::Applied {
            analysis,
            correction_db,
            ..
        } = dsp::tone::process(&mut x, FS, &Default::default())
        else {
            panic!("{name}: expected a correction (> 8 s of speech)");
        };
        let curve: Vec<String> = correction_db.iter().map(|g| format!("{g:+.1}")).collect();
        eprintln!(
            "{name}: {:.1} s speech, bandwidth {:.0} Hz, boost allowed up to {} Hz band\n    {}\n    {}",
            analysis.speech_seconds,
            analysis.bandwidth_hz,
            CENTRES_HZ[analysis.top_band.unwrap_or(0)],
            curve.join(" "),
            dsp::tone::describe(&correction_db)
        );
        if analysis.bandwidth_hz < BAND_LIMITED_HZ {
            for (b, &f) in CENTRES_HZ.iter().enumerate() {
                if f > 11_000.0 {
                    assert!(
                        correction_db[b] <= 0.0,
                        "{name}: boost {:.1} dB at {f} Hz",
                        correction_db[b]
                    );
                }
            }
        }
    }
}

/// Below this measured bandwidth a take counts as band-limited (Bluetooth).
const BAND_LIMITED_HZ: f64 = 14_000.0;

/// Octave-band levels 125 Hz-8 kHz (sum of three 1/3-octave bands) from
/// the speech-active LTAS, dB relative to 500 Hz-2 kHz.
fn octave_levels(x: &[f32]) -> Vec<f64> {
    let m = dsp::tone::analyse(x, FS, &Default::default())
        .unwrap()
        .measured_db;
    // 1/3-octave bands 100,125,160 -> 125 Hz octave, ... 6.3k,8k,10k -> 8 kHz
    (0..7)
        .map(|o| {
            let p: f64 = (0..3).map(|k| 10f64.powf(m[3 * o + k] / 10.0)).sum();
            10.0 * p.log10()
        })
        .collect()
}

/// Least-squares slope of `y` against octave number, dB/octave.
fn slope(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mx = (n - 1.0) / 2.0;
    let my = y.iter().sum::<f64>() / n;
    let num: f64 = y
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64 - mx) * (v - my))
        .sum();
    let den: f64 = (0..y.len()).map(|i| (i as f64 - mx).powi(2)).sum();
    num / den
}

/// Identity guard: the whole chain may not tilt the voice's spectrum by
/// more than 1.5 dB/octave (125 Hz-8 kHz), tone correction included.
#[test]
fn corpus_tilt_change_is_small() {
    for (name, path) in corpus() {
        let raw = io::load_wav(&path, Default::default()).unwrap().samples;
        let before = octave_levels(&raw);
        let mut tilts = Vec::new();
        for tone_on in [false, true] {
            let mut cfg = PostConfig::default();
            cfg.tone_correction.enabled = tone_on;
            let mut x = raw.clone();
            pre::process(&mut x, FS, &PreConfig::default());
            chain::process(&mut x, FS_U32, &cfg, |_, _| Ok(())).unwrap();
            let after = octave_levels(&x);
            let diff: Vec<f64> = after.iter().zip(&before).map(|(a, b)| a - b).collect();
            tilts.push(slope(&diff));
        }
        eprintln!(
            "{name}: tilt change tone off {:+.2} dB/oct, tone on {:+.2} dB/oct",
            tilts[0], tilts[1]
        );
        assert!(tilts[1].abs() <= 1.5, "{name}: {:.2} dB/oct", tilts[1]);
    }
}
