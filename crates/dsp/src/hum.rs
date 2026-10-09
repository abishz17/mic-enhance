//! Mains hum: detect 50 vs 60 Hz (and which harmonics), then notch only
//! what is actually there.
//!
//! Detection works on a *median* Welch spectrum: the file is cut into
//! overlapping windows of up to 2.7 s (0.37 Hz resolution), each gets an
//! FFT, and every frequency bin takes the median power across windows.
//! Long windows matter: a steady tone keeps all its energy in one bin
//! while the noise per bin shrinks, so every doubling of the window adds
//! 3 dB of hum-to-noise contrast. Hum is stationary, so it shows up
//! in every window and survives the median; speech harmonics move with the
//! voice pitch and get voted out. A plain mean would let a long sustained
//! vowel at 100 Hz look like hum.
//!
//! A line counts as hum when it stands `detect_threshold_db` above the
//! median of its neighbourhood (6..25 Hz either side). The exact mains
//! frequency (real grids wander +-0.05 Hz) is then refined with a fine
//! DTFT scan, because a narrow notch 0.1 Hz off target loses most of
//! its depth.

use crate::biquad::{Cascade, Coefficients};
use crate::config::{HumConfig, HumMode};
use realfft::RealFftPlanner;

/// Longest analysis window: 2.7 s at 48 kHz, 0.37 Hz bins.
const MAX_SEGMENT_LEN: usize = 1 << 17;
/// Shortest window worth analysing: 0.34 s, 2.9 Hz bins.
const MIN_SEGMENT_LEN: usize = 1 << 14;
/// The median needs at least this many windows to vote out speech.
const MIN_SEGMENTS: usize = 3;
/// Mains frequencies are allowed to be this far off nominal (fraction).
const MAINS_TOLERANCE: f64 = 0.01;
/// Neighbourhood used to estimate the "floor" around a candidate line.
const FLOOR_INNER_HZ: f64 = 6.0;
const FLOOR_OUTER_HZ: f64 = 25.0;
/// How much audio the fine frequency scan looks at.
const REFINE_MAX_SECONDS: f64 = 30.0;

/// One detected spectral line.
#[derive(Debug, Clone, PartialEq)]
pub struct HumLine {
    /// 1 = fundamental, 2 = second harmonic, ...
    pub harmonic: u32,
    pub freq_hz: f64,
    /// Height above the local spectral floor.
    pub prominence_db: f64,
}

/// Result of hum detection.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HumReport {
    /// Refined mains fundamental, `None` if no hum was found.
    pub fundamental_hz: Option<f64>,
    /// Lines that will be notched.
    pub lines: Vec<HumLine>,
}

/// Median-of-segments power spectrum (Welch with median averaging).
struct Spectrum {
    power: Vec<f64>,
    bin_hz: f64,
}

impl Spectrum {
    fn median_welch(x: &[f32], sample_rate: f64) -> Option<Self> {
        // Largest power-of-two window that still gives MIN_SEGMENTS windows
        // at 50 % overlap: (MIN_SEGMENTS + 1) / 2 window lengths of audio.
        let fit = x.len() * 2 / (MIN_SEGMENTS + 1);
        let segment_len = (fit + 1).next_power_of_two() / 2;
        if segment_len < MIN_SEGMENT_LEN {
            return None;
        }
        let segment_len = segment_len.min(MAX_SEGMENT_LEN);
        let hop = segment_len / 2;
        let window: Vec<f64> = (0..segment_len)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / segment_len as f64).cos())
            .collect();
        let mut planner = RealFftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(segment_len);
        let mut input = fft.make_input_vec();
        let mut output = fft.make_output_vec();

        // Only bins below ~1.1 kHz matter for hum; keep memory bounded.
        let max_bin = ((1100.0 / sample_rate) * segment_len as f64) as usize;
        let mut per_bin: Vec<Vec<f64>> = vec![Vec::new(); max_bin];

        for start in (0..=x.len() - segment_len).step_by(hop) {
            for ((dst, &s), w) in input
                .iter_mut()
                .zip(&x[start..start + segment_len])
                .zip(&window)
            {
                *dst = f64::from(s) * w;
            }
            fft.process(&mut input, &mut output).ok()?;
            for (bin, values) in per_bin.iter_mut().enumerate() {
                values.push(output[bin].norm_sqr());
            }
        }
        let power = per_bin
            .into_iter()
            .map(|mut v| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            })
            .collect();
        Some(Self {
            power,
            bin_hz: sample_rate / segment_len as f64,
        })
    }

    fn bin_of(&self, hz: f64) -> usize {
        (hz / self.bin_hz).round() as usize
    }

    /// Prominence (dB) of the strongest bin within `+-tol_hz` of `centre`,
    /// relative to the median of the surrounding floor. `None` if that bin
    /// is not a local maximum: then it is only the skirt of some other
    /// tone next door (e.g. a 125 Hz voice partial leaking towards 120 Hz).
    fn line_prominence(&self, centre: f64, tol_hz: f64) -> Option<f64> {
        let lo = self.bin_of(centre - tol_hz).max(1);
        let hi = self
            .bin_of(centre + tol_hz)
            .min(self.power.len().checked_sub(2)?);
        let (peak_bin, peak) = (lo..=hi)
            .map(|b| (b, self.power[b]))
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        let is_local_max = peak >= self.power[peak_bin - 1] && peak >= self.power[peak_bin + 1];
        if !is_local_max {
            return None;
        }

        let floor_hi = (self.bin_of(centre + FLOOR_OUTER_HZ) + 1).min(self.power.len());
        let mut floor: Vec<f64> = (self.bin_of(centre - FLOOR_OUTER_HZ)..floor_hi)
            .filter(|&b| {
                let d = (b as f64 * self.bin_hz - centre).abs();
                (FLOOR_INNER_HZ..=FLOOR_OUTER_HZ).contains(&d)
            })
            .map(|b| self.power[b])
            .collect();
        if floor.is_empty() {
            return None;
        }
        floor.sort_by(f64::total_cmp);
        let floor = floor[floor.len() / 2].max(1e-30);
        Some(10.0 * (peak.max(1e-30) / floor).log10())
    }
}

/// Lines found for one mains candidate (50 or 60 Hz).
fn scan_candidate(spec: &Spectrum, mains: f64, cfg: &HumConfig, sample_rate: f64) -> Vec<HumLine> {
    (1..=cfg.max_harmonics)
        .filter_map(|k| {
            let f = mains * f64::from(k);
            if f >= sample_rate / 2.0 * 0.9 {
                return None;
            }
            let tol = (f * MAINS_TOLERANCE).max(spec.bin_hz);
            let prominence_db = spec.line_prominence(f, tol)?;
            (prominence_db >= cfg.detect_threshold_db).then_some(HumLine {
                harmonic: k,
                freq_hz: f,
                prominence_db,
            })
        })
        .collect()
}

/// Score that ignores frequencies shared by both grids (300, 600 Hz ...),
/// since a line there says nothing about 50 vs 60.
fn score(lines: &[HumLine]) -> f64 {
    lines
        .iter()
        .filter(|l| !(l.freq_hz.round() as u32).is_multiple_of(300))
        .map(|l| l.prominence_db)
        .sum()
}

/// Magnitude of the DTFT of `x` at `freq`, via the Goertzel recurrence
/// (one multiply-add per sample instead of a sin/cos per sample).
fn dtft_magnitude(x: &[f64], sample_rate: f64, freq: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / sample_rate;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &v in x {
        let s0 = v + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0).sqrt()
}

/// Find the exact peak frequency near `approx` with a fine DTFT scan
/// (0.01 Hz grid over +-tol, then a parabolic fit on the top 3 points).
/// With 30 s of audio the Hann main lobe is ~0.07 Hz wide, so the grid
/// resolves it and the parabola lands within a few mHz.
fn refine_frequency(x: &[f32], sample_rate: f64, approx: f64, tol: f64) -> f64 {
    let len = x.len().min((REFINE_MAX_SECONDS * sample_rate) as usize);
    let windowed: Vec<f64> = x[..len]
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / len as f64).cos();
            f64::from(v) * w
        })
        .collect();
    let step = 0.01;
    let steps = (tol / step).ceil() as i64;
    let freqs: Vec<f64> = (-steps..=steps).map(|i| approx + i as f64 * step).collect();
    let mags: Vec<f64> = freqs
        .iter()
        .map(|&f| dtft_magnitude(&windowed, sample_rate, f))
        .collect();
    let best = mags
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i);
    if best == 0 || best + 1 >= mags.len() {
        return freqs[best];
    }
    let (a, b, c) = (mags[best - 1], mags[best], mags[best + 1]);
    let denom = a - 2.0 * b + c;
    let offset = if denom.abs() > 1e-30 {
        0.5 * (a - c) / denom
    } else {
        0.0
    };
    freqs[best] + offset.clamp(-1.0, 1.0) * step
}

/// Look for mains hum. Pass the signal *before* the high-pass so the
/// fundamental is still visible.
pub fn detect(x: &[f32], sample_rate: f64, cfg: &HumConfig) -> HumReport {
    let candidates: &[f64] = match cfg.mode {
        HumMode::Off => return HumReport::default(),
        HumMode::Auto => &[50.0, 60.0],
        HumMode::Hz50 => &[50.0],
        HumMode::Hz60 => &[60.0],
    };
    let Some(spec) = Spectrum::median_welch(x, sample_rate) else {
        return HumReport::default(); // < 0.7 s of audio: too short to judge
    };

    let best = candidates
        .iter()
        .map(|&mains| (mains, scan_candidate(&spec, mains, cfg, sample_rate)))
        .filter(|(_, lines)| lines.iter().any(|l| l.harmonic <= 4))
        .max_by(|a, b| score(&a.1).total_cmp(&score(&b.1)));
    let Some((mains, mut lines)) = best else {
        return HumReport::default();
    };

    // Refine on the most prominent line, then derive the fundamental.
    let strongest = lines
        .iter()
        .max_by(|a, b| a.prominence_db.total_cmp(&b.prominence_db))
        .cloned()
        .expect("non-empty: filtered on any()");
    let k = f64::from(strongest.harmonic);
    let tol = mains * MAINS_TOLERANCE * k;
    let fundamental = refine_frequency(x, sample_rate, strongest.freq_hz, tol) / k;
    for line in &mut lines {
        line.freq_hz = fundamental * f64::from(line.harmonic);
    }
    HumReport {
        fundamental_hz: Some(fundamental),
        lines,
    }
}

/// Bank of notch filters, one per detected line, constant Q.
pub fn notch_bank(report: &HumReport, sample_rate: f64, cfg: &HumConfig) -> Cascade {
    let Some(f0) = report.fundamental_hz else {
        return Cascade::default();
    };
    let q = f0 / cfg.notch_bandwidth_hz;
    Cascade::new(
        report
            .lines
            .iter()
            .map(|line| Coefficients::notch(sample_rate, line.freq_hz, q)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 48_000.0;

    fn tone(freq: f64, amp: f64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| (amp * (2.0 * std::f64::consts::PI * freq * n as f64 / FS).sin()) as f32)
            .collect()
    }

    /// Deterministic white noise (xorshift), so tests are repeatable.
    fn noise(amp: f64, len: usize) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * amp as f32
            })
            .collect()
    }

    fn mix(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    #[test]
    fn detects_50_hz_and_refines_frequency() {
        let len = 5 * 48_000;
        let x = mix(&tone(50.03, 0.01, len), &noise(0.01, len));
        let r = detect(&x, FS, &HumConfig::default());
        let f0 = r.fundamental_hz.expect("hum should be found");
        assert!((f0 - 50.03).abs() < 0.005, "refined {f0}");
        assert_eq!(r.lines[0].harmonic, 1);
    }

    #[test]
    fn detects_60_hz_harmonics_without_fundamental() {
        let len = 5 * 48_000;
        let hum = mix(&tone(120.0, 0.01, len), &tone(180.0, 0.005, len));
        let x = mix(&hum, &noise(0.01, len));
        let r = detect(&x, FS, &HumConfig::default());
        assert!((r.fundamental_hz.unwrap() - 60.0).abs() < 0.01);
        let harmonics: Vec<u32> = r.lines.iter().map(|l| l.harmonic).collect();
        assert_eq!(harmonics, vec![2, 3]);
    }

    #[test]
    fn no_hum_in_plain_noise() {
        let x = noise(0.1, 5 * 48_000);
        assert_eq!(detect(&x, FS, &HumConfig::default()).fundamental_hz, None);
    }

    #[test]
    fn too_short_or_silent_is_no_hum() {
        assert_eq!(
            detect(&[0.0; 1000], FS, &HumConfig::default()),
            HumReport::default()
        );
        assert_eq!(
            detect(&vec![0.0; 100_000], FS, &HumConfig::default()).fundamental_hz,
            None
        );
    }
}
