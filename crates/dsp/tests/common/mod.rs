//! Test-signal generators and measurement helpers shared by the
//! integration tests. Measurements are deliberately implemented
//! independently of the code under test (different windows, factors).
#![allow(dead_code)]

use std::f64::consts::PI;
use std::path::PathBuf;

pub const FS: f64 = 48_000.0;
pub const FS_U32: u32 = 48_000;

pub fn sine(freq: f64, amp: f64, secs: f64) -> Vec<f32> {
    sine_phase(freq, amp, 0.0, (secs * FS) as usize)
}

pub fn sine_phase(freq: f64, amp: f64, phase: f64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|n| (amp * (2.0 * PI * freq * n as f64 / FS + phase).sin()) as f32)
        .collect()
}

/// Deterministic white noise in -amp..amp (xorshift64).
pub fn noise(amp: f64, len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (((state as f64 / u64::MAX as f64) * 2.0 - 1.0) * amp) as f32
        })
        .collect()
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Amplitude of the component at `freq` (Hann-windowed single-bin DFT).
/// Accurate for a stationary tone over a window of many cycles.
pub fn amplitude_at(x: &[f32], freq: f64) -> f64 {
    let n = x.len() as f64;
    let (mut re, mut im, mut wsum) = (0.0, 0.0, 0.0);
    for (i, &s) in x.iter().enumerate() {
        let w = 0.5 - 0.5 * (2.0 * PI * i as f64 / n).cos();
        let ph = 2.0 * PI * freq * i as f64 / FS;
        re += f64::from(s) * w * ph.cos();
        im -= f64::from(s) * w * ph.sin();
        wsum += w;
    }
    2.0 * re.hypot(im) / wsum
}

/// Gain (dB) of `process` at `freq`, measured with a 2 s sine and
/// analysing only the last second (after the filter has settled).
pub fn measured_gain_db(freq: f64, mut process: impl FnMut(&mut [f32])) -> f64 {
    let input = sine(freq, 0.5, 2.0);
    let mut output = input.clone();
    process(&mut output);
    let half = input.len() / 2;
    20.0 * (amplitude_at(&output[half..], freq) / amplitude_at(&input[half..], freq)).log10()
}

pub fn db(x: f64) -> f64 {
    20.0 * x.abs().max(1e-12).log10()
}

/// Reference true peak: 16x oversampling with a long Blackman-windowed
/// sinc (64 input samples each side). Independent of `dsp::truepeak`.
pub fn reference_true_peak(x: &[f32]) -> f64 {
    const FACTOR: usize = 16;
    const HALF: isize = 64;
    let len = x.len() as isize;
    let mut peak = x.iter().fold(0.0f64, |m, &s| m.max(f64::from(s).abs()));
    // Pre-compute taps per fractional phase.
    let taps: Vec<Vec<f64>> = (1..FACTOR)
        .map(|p| {
            let mu = p as f64 / FACTOR as f64;
            (-HALF + 1..=HALF)
                .map(|j| {
                    let t = mu - j as f64;
                    let r = (t / HALF as f64).clamp(-1.0, 1.0);
                    let w = 0.42 + 0.5 * (PI * r).cos() + 0.08 * (2.0 * PI * r).cos();
                    let sinc = if t.abs() < 1e-12 {
                        1.0
                    } else {
                        (PI * t).sin() / (PI * t)
                    };
                    sinc * w
                })
                .collect()
        })
        .collect();
    for n in 0..len {
        for phase in &taps {
            let mut acc = 0.0;
            for (k, &c) in phase.iter().enumerate() {
                let idx = n + (k as isize - HALF + 1);
                if (0..len).contains(&idx) {
                    acc += f64::from(x[idx as usize]) * c;
                }
            }
            peak = peak.max(acc.abs());
        }
    }
    peak
}

/// A clip from `testdata/smoke/`, or `None` (with a skip message) when
/// it is absent: test recordings are kept out of the repository.
pub fn smoke_path(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/smoke")
        .join(name);
    if path.exists() {
        Some(path)
    } else {
        eprintln!(
            "skipping: testdata/smoke/{name} not present (test recordings are not in the repo)"
        );
        None
    }
}

pub const SMOKE_FILES: [&str; 3] = [
    "clean.wav",
    "degraded_noise_hum_room.wav",
    "degraded_bluetooth.wav",
];
