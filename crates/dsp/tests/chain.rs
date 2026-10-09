//! Whole pre + post chain on the shared smoke clips and on edge-case input.

mod common;

use common::*;
use dsp::config::{PostConfig, PreConfig};
use dsp::{chain, io, pre, rnnoise};

/// Pre stage, optional RNNoise, post chain. Returns the output.
fn run(mut x: Vec<f32>, denoise: bool, post_cfg: &PostConfig) -> (Vec<f32>, chain::PostReport) {
    pre::process(&mut x, FS, &PreConfig::default());
    if denoise {
        x = rnnoise::denoise(&x).0;
    }
    let report = chain::process(&mut x, FS_U32, post_cfg, |_, _| Ok(())).unwrap();
    (x, report)
}

#[test]
fn smoke_files_hit_loudness_and_true_peak_targets() {
    for preset_lufs in [-14.0, -16.0] {
        let mut cfg = PostConfig::default();
        cfg.loudness.target_lufs = preset_lufs;
        for name in SMOKE_FILES {
            let Some(path) = smoke_path(name) else {
                continue;
            };
            for denoise in [false, true] {
                let audio = io::load_wav(&path, Default::default()).unwrap();
                let (out, report) = run(audio.samples, denoise, &cfg);
                let m = dsp::loudness::measure(&out, FS_U32, false).unwrap();
                let reference_tp = db(reference_true_peak(&out));
                eprintln!(
                    "{name:32} rnnoise={denoise:5} -> {m}; reference TP {reference_tp:+.3}; loudness iterations {}",
                    report.loudness.iterations
                );
                let lufs = m.integrated_lufs.unwrap();
                assert!((lufs - preset_lufs).abs() <= 0.3, "{name}: {lufs:.2} LUFS");
                assert!(
                    m.true_peak_dbtp <= -1.0,
                    "{name}: {:.3} dBTP",
                    m.true_peak_dbtp
                );
                assert!(
                    reference_tp <= -1.0,
                    "{name}: reference {reference_tp:.3} dBTP"
                );
            }
        }
    }
}

#[test]
fn silence_in_silence_out_no_nan() {
    let (out, report) = run(vec![0.0; 5 * 48_000], true, &PostConfig::default());
    assert!(out.iter().all(|&s| s == 0.0));
    assert_eq!(report.working_gain_db, None);
    assert_eq!(report.output.unwrap().integrated_lufs, None);
}

#[test]
fn digital_full_scale_input_is_finite_and_within_ceiling() {
    let cases: [(&str, Vec<f32>); 3] = [
        (
            "full-scale square",
            (0..5 * 48_000)
                .map(|n| if (n / 120) % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
        ),
        ("full-scale sine", sine(440.0, 1.0, 5.0)),
        ("DC at full scale", vec![1.0; 5 * 48_000]),
    ];
    for (name, x) in cases {
        for denoise in [false, true] {
            let (out, report) = run(x.clone(), denoise, &PostConfig::default());
            assert!(out.iter().all(|s| s.is_finite()), "{name}: NaN/Inf");
            let m = report.output.unwrap();
            eprintln!("{name:18} rnnoise={denoise:5}: {m}");
            assert!(m.true_peak_dbtp <= -1.0, "{name}: {m}");
        }
    }
}

#[test]
fn tiny_inputs_do_not_panic() {
    for len in [1, 2, 7, 480, 4800] {
        let (out, _) = run(noise(0.5, len, len as u64), true, &PostConfig::default());
        assert_eq!(out.len(), len);
        assert!(out.iter().all(|s| s.is_finite()));
    }
}
