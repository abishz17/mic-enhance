//! Adaptive tone correction: measure the recording's long-term average
//! speech spectrum (LTAS) and EQ it toward a neutral reference.
//!
//! ## Two EQ layers, two owners
//!
//! * **Correction layer (this module)** owns *neutrality*: whatever the mic
//!   or room did (boxy laptop mic, thin AirPods, an unknown future mic), bring
//!   the voice's 1/3-octave balance to the international LTASS. It adapts
//!   per file and never adds character.
//! * **Character layer** (`post.subtractive_eq`, `post.additive_eq`) owns
//!   the *house sound*: -2.5 dB @ 300 Hz, +2 dB @ 4 kHz, +1.5 dB air.
//!   Fixed, identical for every file, applied on top of the neutral result.
//!
//! Splitting them means the "premium" sound is defined once, relative to a
//! known neutral starting point, instead of being re-tuned per mic.
//!
//! ## Steps
//!
//! 1. Frame the signal (8192-pt FFT, 50 % overlap). Frames of digital
//!    silence (< -100 dBFS, e.g. AirPods gating) are ignored entirely.
//! 2. Speech-active frames: within 20 dB of the loud (95th percentile)
//!    frame level, and at least 10 dB above the quiet (10th percentile)
//!    one. The second rule rejects room-noise frames in noisy takes; it is
//!    capped at "10 dB under loud" because in continuous speech the 10th
//!    percentile is itself quiet speech, not noise.
//! 3. Per 1/3-octave band (100 Hz .. 12.5 kHz): speech level = mean power
//!    over active frames; noise level = mean power over the quietest 10 %
//!    of frames. Bandwidth = highest frequency where speech is still
//!    `min_snr_db` above that noise.
//! 4. Deviation = target - measured, both normalised to their 500 Hz-2 kHz
//!    average; smoothed across bands with a 1/3-octave Gaussian, using
//!    only trustworthy bands (in bandwidth, enough SNR).
//! 5. Only 200 Hz-12.5 kHz is corrected (see `MIN_ADAPTIVE_HZ`), and only
//!    the part of the deviation beyond a +-2 dB deadband (natural talker
//!    variation). Identity beats convergence: a tool that makes the
//!    speaker sound like someone else has failed, however neutral.
//! 6. Limits, applied *after* smoothing so smoothing can't undo them:
//!    cut <= `max_cut_db`; boost <= `max_boost_db` and <= (band SNR -
//!    `min_snr_db`), so a band with less than 10 dB SNR is never boosted
//!    and one with 13 dB gets at most 3 dB; no boost at all above the
//!    detected bandwidth (above the highest band with >= `min_snr_db` SNR
//!    that is less than 20 dB under the target), where it would only
//!    raise hiss and codec junk.
//!    Finally the curve is scaled by `strength` (0..1).
//! 7. Realise the curve with one peaking biquad per band (minimum phase:
//!    no pre-ringing, unlike a linear-phase FIR), with gains solved by
//!    least squares so the *combined* response hits each band centre.

use crate::biquad::{Cascade, Coefficients};
use crate::config::ToneConfig;
use crate::util::{percentile, power_to_db};
use realfft::RealFftPlanner;

/// 1/3-octave band centres (ISO 266 preferred numbers), 100 Hz-12.5 kHz.
pub const CENTRES_HZ: [f64; BANDS] = [
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0,
];
pub const BANDS: usize = 22;

/// Target: the "universal" international long-term average speech
/// spectrum, male and female talkers combined, 1/3-octave band levels in
/// dB SPL at 70 dB overall, from
///
///   D. Byrne et al., "An international comparison of long-term average
///   speech spectra", J. Acoust. Soc. Am. 96(4), 2108-2120 (1994).
///
/// Only the *shape* matters (it is normalised to 500 Hz-2 kHz).
/// NOTE: transcribed from the commonly reproduced ILTASS table; not yet
/// re-checked against the original paper (no access during development).
const LTASS_DB: [f64; BANDS] = [
    54.4, 57.7, 56.8, 60.2, 60.3, 59.0, 62.1, 62.1, 60.5, 56.8, 53.7, 53.0, 52.0, 48.7, 48.1, 46.8,
    45.6, 44.5, 44.3, 43.7, 43.4, 41.3,
];

const FFT_LEN: usize = 8192;
const HOP: usize = FFT_LEN / 2;
const DIGITAL_SILENCE_DB: f64 = -100.0;
const ACTIVE_BELOW_LOUD_DB: f64 = 20.0;
const ACTIVE_ABOVE_QUIET_DB: f64 = 10.0;
/// A band this far *below the target* is not a tonal tilt but missing
/// bandwidth (e.g. AirPods above 12 kHz, Bluetooth above 8 kHz).
const OUT_OF_BAND_DB: f64 = 20.0;
/// Peaking-filter Q for 1/3-octave bandwidth: sqrt(2^N) / (2^N - 1), N = 1/3.
const BAND_Q: f64 = 4.318;
/// No adaptive change below this. The ILTASS averages male and female
/// talkers, who differ most below ~200 Hz (voice fundamental and chest
/// resonance): correcting there pulls a low voice toward the average
/// voice, i.e. changes who it sounds like (a laptop-mic recording lost 6.9 dB
/// at 100 Hz, right on its 123 Hz F0). The high-pass and the character EQ own
/// the low end instead.
pub const MIN_ADAPTIVE_HZ: f64 = 200.0;
/// Width of the cross-band smoothing (in 1/3-octave bands). 1.5 kept
/// more of the talker's own formant detail but left 1.7-2.1 dB spread in
/// the convergence test; 1.0 brings it under 1 dB.
const SMOOTH_SIGMA_BANDS: f64 = 1.0;
/// Bands averaged for normalisation (500 Hz .. 2 kHz).
const REF_BANDS: std::ops::RangeInclusive<usize> = 7..=13;

/// What was measured.
#[derive(Debug, Clone, PartialEq)]
pub struct ToneAnalysis {
    pub speech_seconds: f64,
    /// Speech LTAS per band, dB relative to the 500 Hz-2 kHz average.
    pub measured_db: [f64; BANDS],
    /// Speech-to-noise ratio per band, dB (noise = the quietest 10 % of
    /// non-silent frames; in continuous speech that is quiet speech, which
    /// under-reads SNR and so errs towards fewer boosts).
    pub snr_db: [f64; BANDS],
    /// Highest frequency with usable speech content.
    pub bandwidth_hz: f64,
    /// Highest band that may be boosted (index), `None` if none.
    pub top_band: Option<usize>,
}

/// What the stage did. (Clippy's large-variant lint is silenced: there is
/// one of these per file, so a few hundred bytes don't matter.)
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum ToneOutcome {
    Applied {
        analysis: ToneAnalysis,
        /// Intended correction per band (dB).
        correction_db: [f64; BANDS],
        /// What the filter bank actually does at each band centre (dB).
        realised_db: [f64; BANDS],
    },
    Skipped(String),
}

/// Target shape relative to its own 500 Hz-2 kHz average.
pub fn target_db() -> [f64; BANDS] {
    normalise(&LTASS_DB.map(|db| 10f64.powf(db / 10.0)))
}

/// Neutral speech power *per Hz* at `freq`, dB relative to its average
/// over 500 Hz-2 kHz (LTASS band level minus 10 log10(bandwidth),
/// interpolated on a log-frequency axis, held flat beyond the table).
fn expected_density_db(freq: f64) -> f64 {
    let density: [f64; BANDS] = std::array::from_fn(|b| {
        let (lo, hi) = band_edges(CENTRES_HZ[b]);
        LTASS_DB[b] - 10.0 * (hi - lo).log10()
    });
    let (lo, hi) = (
        band_edges(CENTRES_HZ[*REF_BANDS.start()]).0,
        band_edges(CENTRES_HZ[*REF_BANDS.end()]).1,
    );
    let ref_power: f64 = REF_BANDS.map(|b| 10f64.powf(LTASS_DB[b] / 10.0)).sum();
    let reference = 10.0 * (ref_power / (hi - lo)).log10();
    let at = |b: usize| density[b] - reference;
    let x = freq.clamp(CENTRES_HZ[0], CENTRES_HZ[BANDS - 1]).ln();
    let i = CENTRES_HZ
        .iter()
        .rposition(|c| c.ln() <= x)
        .unwrap_or(0)
        .min(BANDS - 2);
    let t = (x - CENTRES_HZ[i].ln()) / (CENTRES_HZ[i + 1].ln() - CENTRES_HZ[i].ln());
    at(i) + t.clamp(0.0, 1.0) * (at(i + 1) - at(i))
}

/// Powers -> dB relative to the power-average of the reference bands.
fn normalise(power: &[f64; BANDS]) -> [f64; BANDS] {
    let reference: f64 = REF_BANDS.map(|b| power[b]).sum::<f64>() / REF_BANDS.count() as f64;
    power.map(|p| power_to_db(p) - power_to_db(reference))
}

fn band_edges(centre: f64) -> (f64, f64) {
    let half = 2f64.powf(1.0 / 6.0);
    (centre / half, centre * half)
}

/// Measure the speech LTAS, band SNRs and bandwidth. `None` when there is
/// no non-silent audio at all.
pub fn analyse(x: &[f32], sample_rate: f64, cfg: &ToneConfig) -> Option<ToneAnalysis> {
    if x.len() < FFT_LEN {
        return None;
    }
    let bin_hz = sample_rate / FFT_LEN as f64;
    let range_of = |lo: f64, hi: f64, bins: usize| {
        (lo / bin_hz).ceil() as usize..((hi / bin_hz).ceil() as usize).min(bins)
    };
    // Fine 1/6-octave windows from 1 kHz up, for the bandwidth search.
    let fine_centres: Vec<f64> = (0..)
        .map(|k| 1000.0 * 2f64.powf(f64::from(k) / 6.0))
        .take_while(|&f| f * 2f64.powf(1.0 / 12.0) <= sample_rate / 2.0)
        .collect();
    let fine_edges = |f: f64| (f / 2f64.powf(1.0 / 12.0), f * 2f64.powf(1.0 / 12.0));

    let window: Vec<f64> = (0..FFT_LEN)
        .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FFT_LEN as f64).cos())
        .collect();
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(FFT_LEN);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let bins = spectrum.len();

    // One row per non-silent frame: level, 22 band powers, fine powers.
    // ~0.3 KB per 85 ms frame, so an hour of audio needs ~13 MB.
    struct Frame {
        level: f64,
        bands: [f64; BANDS],
        fine: Vec<f64>,
    }
    let mut frames = Vec::new();
    for start in (0..=x.len() - FFT_LEN).step_by(HOP) {
        let chunk = &x[start..start + FFT_LEN];
        let level = crate::util::rms_db(chunk);
        if level < DIGITAL_SILENCE_DB {
            continue; // e.g. AirPods gating room tone to exact zero
        }
        for ((dst, &v), w) in input.iter_mut().zip(chunk).zip(&window) {
            *dst = f64::from(v) * w;
        }
        fft.process(&mut input, &mut spectrum).ok()?;
        let power = |lo: f64, hi: f64| -> f64 {
            spectrum[range_of(lo, hi, bins)]
                .iter()
                .map(|c| c.norm_sqr())
                .sum()
        };
        frames.push(Frame {
            level,
            bands: std::array::from_fn(|b| {
                let (lo, hi) = band_edges(CENTRES_HZ[b]);
                power(lo, hi)
            }),
            fine: fine_centres
                .iter()
                .map(|&f| {
                    let (lo, hi) = fine_edges(f);
                    power(lo, hi)
                })
                .collect(),
        });
    }

    let levels: Vec<f64> = frames.iter().map(|f| f.level).collect();
    let loud = percentile(&levels, 95.0)?;
    let quiet = percentile(&levels, 10.0)?;
    let above_noise = (quiet + ACTIVE_ABOVE_QUIET_DB).min(loud - ACTIVE_ABOVE_QUIET_DB);
    let threshold = (loud - ACTIVE_BELOW_LOUD_DB).max(above_noise);
    let active: Vec<&Frame> = frames.iter().filter(|f| f.level >= threshold).collect();
    let speech_seconds = active.len() as f64 * HOP as f64 / sample_rate;
    let n_active = active.len().max(1) as f64;

    // Speech = mean over active frames. Noise = 10th percentile *per band*
    // over all non-silent frames: per band, because a quiet "s" is a quiet
    // frame overall but loud above 4 kHz, so classifying whole frames as
    // noise would mistake fricatives for hiss.
    let speech_and_noise = |get: &dyn Fn(&Frame) -> f64| -> (f64, f64) {
        let speech = active.iter().map(|f| get(f)).sum::<f64>() / n_active;
        let all: Vec<f64> = frames.iter().map(get).collect();
        (speech, percentile(&all, 10.0).unwrap_or(0.0))
    };
    let mut speech = [0.0; BANDS];
    let mut snr_db = [0.0; BANDS];
    for b in 0..BANDS {
        let (s, n) = speech_and_noise(&|f: &Frame| f.bands[b]);
        speech[b] = s;
        snr_db[b] = power_to_db(s) - power_to_db(n);
    }

    // Bandwidth: the highest fine window where speech (a) stands
    // min_snr_db over the noise, and (b) is no more than OUT_OF_BAND_DB
    // under what a neutral voice would have there. (b) is what catches
    // codecs: AirPods leave residue at 12-14 kHz that follows the speech
    // (so (a) passes) but sits ~25 dB under neutral. A band only counts as
    // in-bandwidth if it lies entirely below the edge, so the AirPods
    // 12.5 kHz band (11.2-14.1 kHz) is never boosted.
    let ref_density = {
        let lo = band_edges(CENTRES_HZ[*REF_BANDS.start()]).0;
        let hi = band_edges(CENTRES_HZ[*REF_BANDS.end()]).1;
        REF_BANDS.map(|b| speech[b]).sum::<f64>() / (hi - lo)
    };
    let bandwidth_hz = fine_centres
        .iter()
        .enumerate()
        .rev()
        .find(|&(k, &f)| {
            let (s, n) = speech_and_noise(&|fr: &Frame| fr.fine[k]);
            let (lo, hi) = fine_edges(f);
            let density_db = power_to_db(s / (hi - lo)) - power_to_db(ref_density);
            s > 0.0
                && power_to_db(s) - power_to_db(n) >= cfg.min_snr_db
                && density_db >= expected_density_db(f) - OUT_OF_BAND_DB
        })
        .map_or(0.0, |(_, &f)| fine_edges(f).1);

    let measured_db = normalise(&speech);
    let target = target_db();
    let top_band = (0..BANDS).rev().find(|&b| {
        band_edges(CENTRES_HZ[b]).1 <= bandwidth_hz
            && snr_db[b] >= cfg.min_snr_db
            && measured_db[b] > target[b] - OUT_OF_BAND_DB
    });

    Some(ToneAnalysis {
        speech_seconds,
        measured_db,
        snr_db,
        bandwidth_hz,
        top_band,
    })
}

/// Target minus measured, per band, before smoothing and limits.
pub fn raw_deviation(analysis: &ToneAnalysis) -> [f64; BANDS] {
    let target = target_db();
    std::array::from_fn(|b| target[b] - analysis.measured_db[b])
}

/// Intended correction per band, with smoothing and all safety limits.
pub fn correction(analysis: &ToneAnalysis, cfg: &ToneConfig) -> [f64; BANDS] {
    let raw = raw_deviation(analysis);
    // Only bands we can trust feed the smoothing: inside the adaptive
    // range, in-bandwidth and with enough SNR. A missing 12.5 kHz band
    // (+18 dB "deviation") must not leak into its neighbours.
    let adaptive = |b: usize| CENTRES_HZ[b] >= MIN_ADAPTIVE_HZ;
    let trusted: [bool; BANDS] = std::array::from_fn(|b| {
        adaptive(b)
            && analysis.top_band.is_some_and(|top| b <= top)
            && analysis.snr_db[b] >= cfg.min_snr_db
    });
    // Gaussian smoothing across bands, sigma = 1 band (1/3 octave).
    // A single talker's spectrum has formant bumps of +-4 dB from band to
    // band that the multi-talker average does not; those are the person's
    // voice, not the mic; smoothing keeps corrections to features about
    // 2/3 octave or wider (tilts, boom, dullness).
    let smoothed: [f64; BANDS] = std::array::from_fn(|b| {
        let (mut sum, mut weight) = (0.0, 0.0);
        for (k, &ok) in trusted.iter().enumerate() {
            let d = k as f64 - b as f64;
            if ok && d.abs() <= 4.0 {
                let w = (-0.5 * (d / SMOOTH_SIGMA_BANDS).powi(2)).exp();
                sum += w * raw[k];
                weight += w;
            }
        }
        if weight > 0.0 { sum / weight } else { 0.0 }
    });
    std::array::from_fn(|b| {
        if !adaptive(b) {
            return 0.0;
        }
        // Deadband: talkers naturally sit within +-2-3 dB of the average
        // spectrum; only the part of the deviation beyond that is treated
        // as mic/room colouration.
        let d = smoothed[b];
        let beyond = d.signum() * (d.abs() - cfg.deadband_db).max(0.0);
        let in_band = analysis.top_band.is_some_and(|top| b <= top);
        let max_boost = if in_band {
            (analysis.snr_db[b] - cfg.min_snr_db).clamp(0.0, cfg.max_boost_db)
        } else {
            0.0
        };
        beyond.clamp(-cfg.max_cut_db, max_boost) * cfg.strength
    })
}

/// Fit one peaking biquad per band (from 200 Hz up) so the *combined*
/// response equals `wanted_db` at every band centre. Neighbouring bells overlap, so the
/// gains are solved as a linear system (dB responses of small bells add
/// almost linearly), then refined twice against the exact response.
pub fn fit_bank(wanted_db: &[f64; BANDS], sample_rate: f64) -> Cascade {
    let bank = |gains: &[f64; BANDS]| {
        Cascade::new(
            CENTRES_HZ
                .iter()
                .zip(gains)
                .filter(|(_, g)| g.abs() > 1e-3)
                .map(|(&f, &g)| Coefficients::peaking(sample_rate, f, BAND_Q, g)),
        )
    };
    // Interaction matrix: response at centre i of a +1 dB bell at centre j.
    let m: Vec<[f64; BANDS]> = (0..BANDS)
        .map(|i| {
            std::array::from_fn(|j| {
                Coefficients::peaking(sample_rate, CENTRES_HZ[j], BAND_Q, 1.0)
                    .magnitude_db(sample_rate, CENTRES_HZ[i])
            })
        })
        .collect();

    let mut gains = [0.0; BANDS];
    for _ in 0..3 {
        let current = bank(&gains);
        let residual: [f64; BANDS] = std::array::from_fn(|i| {
            wanted_db[i] - current.magnitude_db(sample_rate, CENTRES_HZ[i])
        });
        let step = solve(&m, &residual);
        for (g, s) in gains.iter_mut().zip(step) {
            *g = (*g + s).clamp(-15.0, 15.0);
        }
    }
    bank(&gains)
}

/// Gaussian elimination with partial pivoting for the small dense system.
fn solve(m: &[[f64; BANDS]], rhs: &[f64; BANDS]) -> [f64; BANDS] {
    let mut a: Vec<[f64; BANDS + 1]> = (0..BANDS)
        .map(|i| std::array::from_fn(|j| if j < BANDS { m[i][j] } else { rhs[i] }))
        .collect();
    for col in 0..BANDS {
        let pivot = (col..BANDS)
            .max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs()))
            .expect("non-empty range");
        a.swap(col, pivot);
        let diag = a[col][col];
        if diag.abs() < 1e-12 {
            continue;
        }
        let pivot_row = a[col];
        for row in a.iter_mut().skip(col + 1) {
            let factor = row[col] / diag;
            for (v, p) in row.iter_mut().zip(&pivot_row).skip(col) {
                *v -= factor * p;
            }
        }
    }
    let mut x = [0.0; BANDS];
    for row in (0..BANDS).rev() {
        let tail: f64 = (row + 1..BANDS).map(|k| a[row][k] * x[k]).sum();
        let diag = a[row][row];
        x[row] = if diag.abs() < 1e-12 {
            0.0
        } else {
            (a[row][BANDS] - tail) / diag
        };
    }
    x
}

/// Analyse, decide and apply in place.
pub fn process(buf: &mut [f32], sample_rate: f64, cfg: &ToneConfig) -> ToneOutcome {
    let Some(analysis) = analyse(buf, sample_rate, cfg) else {
        return ToneOutcome::Skipped("no audio to measure".into());
    };
    if analysis.speech_seconds < cfg.min_speech_seconds {
        return ToneOutcome::Skipped(format!(
            "only {:.1} s of speech detected (need {:.0} s for a reliable spectrum)",
            analysis.speech_seconds, cfg.min_speech_seconds
        ));
    }
    let correction_db = correction(&analysis, cfg);
    let mut bank = fit_bank(&correction_db, sample_rate);
    let realised_db = CENTRES_HZ.map(|f| bank.magnitude_db(sample_rate, f));
    bank.process(buf);
    ToneOutcome::Applied {
        analysis,
        correction_db,
        realised_db,
    }
}

/// One-line description: the biggest cut and boost regions, e.g.
/// "-2.1 dB @ 250 Hz, +1.8 dB @ 3.15 kHz".
pub fn describe(correction_db: &[f64; BANDS]) -> String {
    // Local extremes with |gain| >= 1 dB.
    let mut parts = Vec::new();
    for b in 0..BANDS {
        let g = correction_db[b];
        let left = if b > 0 { correction_db[b - 1] } else { 0.0 };
        let right = if b + 1 < BANDS {
            correction_db[b + 1]
        } else {
            0.0
        };
        let is_peak = g >= 1.0 && g >= left && g > right;
        let is_dip = g <= -1.0 && g <= left && g < right;
        if is_peak || is_dip {
            let f = CENTRES_HZ[b];
            let freq = if f >= 1000.0 {
                format!("{} kHz", f / 1000.0)
            } else {
                format!("{f} Hz")
            };
            parts.push(format!("{g:+.1} dB @ {freq}"));
        }
    }
    if parts.is_empty() {
        "within +-1 dB of neutral, no change needed".into()
    } else {
        parts.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_is_normalised() {
        let t = target_db();
        let reference: f64 =
            REF_BANDS.map(|b| 10f64.powf(t[b] / 10.0)).sum::<f64>() / REF_BANDS.count() as f64;
        assert!(power_to_db(reference).abs() < 1e-9);
    }

    #[test]
    fn solver_inverts_a_known_system() {
        let m: Vec<[f64; BANDS]> = (0..BANDS)
            .map(|i| std::array::from_fn(|j| if i == j { 2.0 } else { 0.1 }))
            .collect();
        let x: [f64; BANDS] = std::array::from_fn(|i| i as f64 - 5.0);
        let rhs: [f64; BANDS] = std::array::from_fn(|i| (0..BANDS).map(|j| m[i][j] * x[j]).sum());
        let got = solve(&m, &rhs);
        for (a, b) in got.iter().zip(&x) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn limits_hold_after_smoothing() {
        let cfg = ToneConfig {
            strength: 1.0,
            ..ToneConfig::default()
        };
        let target = target_db();
        // Measured 10 dB under target everywhere (wants +10 boost), but
        // band 5 has only 12 dB SNR and bands above 15 are out of band.
        let mut analysis = ToneAnalysis {
            speech_seconds: 20.0,
            measured_db: target.map(|t| t - 10.0),
            snr_db: [40.0; BANDS],
            bandwidth_hz: 4500.0,
            top_band: Some(15),
        };
        analysis.snr_db[5] = 12.0;
        let c = correction(&analysis, &cfg);
        assert!((c[5] - 2.0).abs() < 1e-9, "SNR-limited boost {}", c[5]);
        assert!(
            c[16..].iter().all(|&g| g == 0.0),
            "no boost above bandwidth"
        );
        assert!(c.iter().all(|&g| g <= cfg.max_boost_db));
        // And cuts are limited too (cuts are allowed everywhere).
        analysis.measured_db = target.map(|t| t + 20.0);
        analysis.snr_db = [40.0; BANDS];
        analysis.top_band = Some(BANDS - 1);
        let c = correction(&analysis, &cfg);
        for (b, &g) in c.iter().enumerate() {
            let expected = if CENTRES_HZ[b] < MIN_ADAPTIVE_HZ {
                0.0
            } else {
                -cfg.max_cut_db
            };
            assert!((g - expected).abs() < 1e-9, "band {b}: {g}");
        }
    }

    #[test]
    fn describe_lists_extremes() {
        let mut c = [0.0; BANDS];
        c[4] = -2.1;
        c[15] = 1.8;
        assert_eq!(describe(&c), "-2.1 dB @ 250 Hz, +1.8 dB @ 3.15 kHz");
        assert!(describe(&[0.3; BANDS]).contains("no change"));
    }
}
