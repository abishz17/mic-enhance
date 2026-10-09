//! The post-restore chain, in this order:
//!
//! | # | stage            | why here                                              |
//! |---|------------------|-------------------------------------------------------|
//! | 1 | working level    | set the programme to -23 LUFS so every threshold below is relative to a known level, whatever the mic gain was |
//! | 1b| tone correction  | adaptive, per-file: measure the speech spectrum after restore (noise no longer biases it) and EQ toward neutral; before any dynamics so the compressor sees the corrected balance. The fixed character EQ below is applied on top |
//! | 1c| phase rotator    | all-pass filters (flat magnitude) that make lopsided voice waveforms symmetric: on laptop-mic recordings this halves the limiter's work (max 9.8 -> 4.4 dB) at no tonal cost |
//! | 2 | subtractive EQ   | cut 300 Hz "boxiness" *before* dynamics, so the compressor doesn't react to energy we are about to remove |
//! | 3 | expander         | before compression: the gap between speech and room tone is still at its widest, so the threshold separates them cleanly |
//! | 4 | compressor       | evens out syllables; loudness-neutral make-up keeps the working level |
//! | 5 | additive EQ      | presence/air boosts after compression, so the compressor doesn't push back against them |
//! | 6 | de-esser         | after the boosts: it treats the sibilance the listener will actually hear, including what +2 dB @ 4 kHz and the air shelf added |
//! | 7 | loudness         | gain to -14 LUFS (measured, applied, re-measured after the limiter) |
//! | 8 | true-peak limiter| last, so nothing after it can create a new peak |

use crate::biquad::{Biquad, Cascade, Coefficients};
use crate::config::PostConfig;
use crate::dynamics::compressor::{self, CompressorReport};
use crate::dynamics::deesser::{self, DeesserReport};
use crate::dynamics::expander::{self, ExpanderReport};
use crate::io::AudioError;
use crate::limiter::{self, LimiterReport};
use crate::loudness::{self, LoudnessError, Measurement};
use crate::tone::{self, ToneOutcome};
use crate::util::{apply_gain, db_to_gain};
use thiserror::Error;

/// Maximum loudness/limiter iterations (2 is typical).
const MAX_LOUDNESS_ITERATIONS: usize = 6;

#[derive(Debug, Error)]
pub enum ChainError {
    #[error(transparent)]
    Loudness(#[from] LoudnessError),
    #[error("writing stage dump failed: {0}")]
    Dump(#[from] AudioError),
}

/// Stage names, in order. Used for dump file names (`NN_name.wav`).
pub const STAGES: [&str; 10] = [
    "working_level",
    "tone_correction",
    "phase_rotator",
    "subtractive_eq",
    "expander",
    "compressor",
    "additive_eq",
    "deesser",
    "loudness",
    "limiter",
];

#[derive(Debug, Clone, Default)]
pub struct LoudnessReport {
    /// Gain that moved the programme to the target (before limiting).
    pub gain_db: Option<f64>,
    /// True if `max_gain_db` stopped us reaching the target.
    pub gain_capped: bool,
    pub iterations: usize,
}

#[derive(Debug, Clone, Default)]
pub struct PostReport {
    /// Gain to the working level; `None` if the input was silent.
    pub working_gain_db: Option<f64>,
    pub tone: Option<ToneOutcome>,
    pub expander: Option<ExpanderReport>,
    pub compressor: Option<CompressorReport>,
    pub deesser: Option<DeesserReport>,
    pub loudness: LoudnessReport,
    pub limiter: Option<LimiterReport>,
    pub output: Option<Measurement>,
}

fn run_filters(buf: &mut [f32], filters: impl IntoIterator<Item = Coefficients>) {
    let mut cascade = Cascade::new(filters);
    if !cascade.is_empty() {
        cascade.process(buf);
    }
}

/// Run the whole post chain in place. `on_stage(name, audio)` is called
/// after every stage (used for `--dump-stages`).
pub fn process(
    buf: &mut [f32],
    sample_rate: u32,
    cfg: &PostConfig,
    mut on_stage: impl FnMut(&'static str, &[f32]) -> Result<(), AudioError>,
) -> Result<PostReport, ChainError> {
    let fs = f64::from(sample_rate);
    let mut report = PostReport::default();
    let mut stages = STAGES.iter();
    let mut done = |buf: &[f32]| on_stage(stages.next().expect("one name per stage"), buf);

    // 1. working level
    report.working_gain_db = loudness::align(buf, sample_rate, cfg.working_lufs, 60.0)?;
    done(buf)?;

    // 1b. tone correction (adaptive layer)
    if cfg.tone_correction.enabled && cfg.tone_correction.strength > 0.0 {
        report.tone = Some(tone::process(buf, fs, &cfg.tone_correction));
    }
    done(buf)?;

    // 1c. phase rotator (acts on the corrected tone)
    let pr = &cfg.phase_rotator;
    if pr.enabled {
        run_filters(
            buf,
            (0..pr.sections).map(|_| Coefficients::allpass(fs, pr.freq_hz, pr.q)),
        );
    }
    done(buf)?;

    // 2. subtractive EQ
    let sub = &cfg.subtractive_eq;
    if sub.enabled {
        Biquad::new(Coefficients::peaking(fs, sub.freq_hz, sub.q, sub.gain_db)).process(buf);
    }
    done(buf)?;

    // 3. expander
    if cfg.expander.enabled {
        report.expander = Some(expander::process(buf, sample_rate, &cfg.expander));
    }
    done(buf)?;

    // 4. compressor
    if cfg.compressor.enabled {
        report.compressor = Some(compressor::process(buf, sample_rate, &cfg.compressor)?);
    }
    done(buf)?;

    // 5. additive EQ
    let (presence, air) = (&cfg.additive_eq.presence, &cfg.additive_eq.air);
    run_filters(
        buf,
        [
            presence
                .enabled
                .then(|| Coefficients::peaking(fs, presence.freq_hz, presence.q, presence.gain_db)),
            air.enabled
                .then(|| Coefficients::high_shelf(fs, air.freq_hz, air.slope, air.gain_db)),
        ]
        .into_iter()
        .flatten(),
    );
    done(buf)?;

    // 6. de-esser
    if cfg.deesser.enabled {
        report.deesser = Some(deesser::process(buf, sample_rate, &cfg.deesser));
    }
    done(buf)?;

    // 7 + 8. loudness and limiter (iterated together)
    let (loud, lim) = normalize_and_limit(buf, sample_rate, cfg, &mut done)?;
    report.loudness = loud;
    report.limiter = lim;

    report.output = Some(loudness::measure(buf, sample_rate, cfg.loudness.dual_mono)?);
    Ok(report)
}

/// Two-pass loudness normalisation around the limiter:
/// measure -> gain to target -> limit -> re-measure -> correct the gain by
/// the error -> limit again (from the un-limited signal), until within
/// `tolerance_lu`. The limiter takes a little loudness off the peaks, so a
/// single pass would land slightly under target.
fn normalize_and_limit(
    buf: &mut [f32],
    sample_rate: u32,
    cfg: &PostConfig,
    done: &mut impl FnMut(&[f32]) -> Result<(), AudioError>,
) -> Result<(LoudnessReport, Option<LimiterReport>), ChainError> {
    let target = cfg.loudness.target_lufs;
    let max_gain = cfg.loudness.max_gain_db;
    let dual = cfg.loudness.dual_mono;
    let source = buf.to_vec();
    let mut report = LoudnessReport::default();
    let mut limiter_report = None;

    let Some(measured) = loudness::integrated_lufs(&source, sample_rate, dual)? else {
        // Silent: no gain, but still run the limiter for the peak guarantee.
        done(buf)?;
        if cfg.limiter.enabled {
            limiter_report = Some(limiter::process(buf, sample_rate, &cfg.limiter));
        }
        done(buf)?;
        return Ok((report, limiter_report));
    };

    // Root-find "output loudness(gain) = target". The limiter makes the
    // slope d(out)/d(gain) a bit below 1, so after the first step we use
    // the secant through the last two tries (converges in 2-3 passes).
    let mut gain_db = target - measured;
    let mut previous: Option<(f64, f64)> = None; // (gain, output loudness)
    for iteration in 1..=MAX_LOUDNESS_ITERATIONS {
        report.gain_capped = gain_db > max_gain;
        gain_db = gain_db.min(max_gain);
        buf.copy_from_slice(&source);
        apply_gain(buf, db_to_gain(gain_db));
        report.iterations = iteration;
        if !cfg.limiter.enabled {
            break;
        }
        limiter_report = Some(limiter::process(buf, sample_rate, &cfg.limiter));
        let Some(now) = loudness::integrated_lufs(buf, sample_rate, dual)? else {
            break;
        };
        let error = target - now;
        if error.abs() <= cfg.loudness.tolerance_lu || report.gain_capped {
            break;
        }
        let slope = match previous {
            Some((g, l)) if (gain_db - g).abs() > 1e-6 => {
                ((now - l) / (gain_db - g)).clamp(0.2, 1.0)
            }
            _ => 1.0,
        };
        previous = Some((gain_db, now));
        gain_db += error / slope;
    }
    report.gain_db = Some(gain_db);

    // Dump the gain-only signal, then the limited one.
    let mut gained = source;
    apply_gain(&mut gained, db_to_gain(gain_db));
    done(&gained)?;
    done(buf)?;
    Ok((report, limiter_report))
}
