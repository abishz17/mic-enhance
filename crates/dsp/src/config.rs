//! Every tunable number in the chain, deserialised from a TOML preset.
//!
//! All structs use `#[serde(default)]`, so a preset only lists what it
//! changes, and `deny_unknown_fields`, so a typo such as `ratoi = 4` is an
//! error instead of a silently ignored setting.
//!
//! Level-dependent thresholds (expander, compressor, de-esser) are in dBFS
//! *at the working level*: the post chain first sets the programme to
//! `post.working_lufs` (default -23 LUFS), so a threshold means the same
//! thing for a whisper-quiet AirPods take and a hot laptop-mic take.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
#[error("invalid config `{field}`: {reason}")]
pub struct ConfigError {
    pub field: &'static str,
    pub reason: String,
}

/// Helper so validation reads as a list of rules.
fn ensure(
    ok: bool,
    field: &'static str,
    reason: impl FnOnce() -> String,
) -> Result<(), ConfigError> {
    if ok {
        Ok(())
    } else {
        Err(ConfigError {
            field,
            reason: reason(),
        })
    }
}

/// A frequency must be a finite number inside (lo, Nyquist).
fn ensure_freq(f: f64, sample_rate: f64, field: &'static str) -> Result<(), ConfigError> {
    ensure(
        f.is_finite() && f > 0.0 && f < sample_rate / 2.0 * 0.98,
        field,
        || {
            format!(
                "{f} Hz must be between 0 and {:.0} Hz",
                sample_rate / 2.0 * 0.98
            )
        },
    )
}

fn ensure_range(v: f64, lo: f64, hi: f64, field: &'static str) -> Result<(), ConfigError> {
    ensure(v.is_finite() && (lo..=hi).contains(&v), field, || {
        format!("{v} must be within {lo}..={hi}")
    })
}

// ---------------------------------------------------------------- pre ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PreConfig {
    /// Subtract the file's mean before filtering (kills a static offset
    /// without the high-pass having to ring it out).
    pub remove_dc: bool,
    pub highpass: HighpassConfig,
    pub hum: HumConfig,
}

impl Default for PreConfig {
    fn default() -> Self {
        Self {
            remove_dc: true,
            highpass: HighpassConfig::default(),
            hum: HumConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HighpassConfig {
    pub enabled: bool,
    pub freq_hz: f64,
    /// Butterworth order; 4 = 24 dB/octave (two biquads).
    pub order: usize,
}

impl Default for HighpassConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            freq_hz: 80.0,
            order: 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HumMode {
    /// Detect 50 vs 60 Hz and notch only the harmonics that are present.
    Auto,
    Off,
    /// Force 50 Hz mains (Europe, Asia, Africa, most of the world).
    #[serde(rename = "50")]
    Hz50,
    /// Force 60 Hz mains (Americas, parts of Japan).
    #[serde(rename = "60")]
    Hz60,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HumConfig {
    pub mode: HumMode,
    /// Highest harmonic considered (10 x 50 Hz = 500 Hz).
    pub max_harmonics: u32,
    /// A spectral line must stand this far above its neighbourhood to count.
    pub detect_threshold_db: f64,
    /// -3 dB width of the notch at the fundamental. Harmonic k gets k times
    /// this width (constant Q), because mains drift scales with k too.
    pub notch_bandwidth_hz: f64,
}

impl Default for HumConfig {
    fn default() -> Self {
        Self {
            mode: HumMode::Auto,
            max_harmonics: 10,
            detect_threshold_db: 10.0,
            notch_bandwidth_hz: 2.0,
        }
    }
}

impl PreConfig {
    pub fn validate(&self, sample_rate: f64) -> Result<(), ConfigError> {
        let hp = &self.highpass;
        ensure_freq(hp.freq_hz, sample_rate, "pre.highpass.freq_hz")?;
        ensure(
            (2..=8).contains(&hp.order) && hp.order.is_multiple_of(2),
            "pre.highpass.order",
            || format!("{} must be 2, 4, 6 or 8", hp.order),
        )?;
        let hum = &self.hum;
        ensure(
            (1..=20).contains(&hum.max_harmonics),
            "pre.hum.max_harmonics",
            || format!("{} must be within 1..=20", hum.max_harmonics),
        )?;
        ensure_range(
            hum.detect_threshold_db,
            3.0,
            60.0,
            "pre.hum.detect_threshold_db",
        )?;
        ensure_range(
            hum.notch_bandwidth_hz,
            0.2,
            20.0,
            "pre.hum.notch_bandwidth_hz",
        )
    }
}

// --------------------------------------------------------------- post ----

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PostConfig {
    /// Loudness the dynamics stages work at (thresholds are relative to it).
    pub working_lufs: f64,
    pub phase_rotator: PhaseRotatorConfig,
    pub tone_correction: ToneConfig,
    pub subtractive_eq: BellConfig,
    pub expander: ExpanderConfig,
    pub compressor: CompressorConfig,
    pub additive_eq: AdditiveEqConfig,
    pub deesser: DeesserConfig,
    pub loudness: LoudnessConfig,
    pub limiter: LimiterConfig,
}

impl Default for PostConfig {
    fn default() -> Self {
        Self {
            working_lufs: -23.0,
            phase_rotator: PhaseRotatorConfig::default(),
            tone_correction: ToneConfig::default(),
            subtractive_eq: BellConfig {
                enabled: true,
                freq_hz: 300.0,
                q: 1.0,
                gain_db: -2.5,
            },
            expander: ExpanderConfig::default(),
            compressor: CompressorConfig::default(),
            additive_eq: AdditiveEqConfig::default(),
            deesser: DeesserConfig::default(),
            loudness: LoudnessConfig::default(),
            limiter: LimiterConfig::default(),
        }
    }
}

/// Adaptive tone correction toward a neutral speech spectrum (see `tone`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToneConfig {
    pub enabled: bool,
    /// Largest cut in any band.
    pub max_cut_db: f64,
    /// Largest boost in any band (further limited by band SNR).
    pub max_boost_db: f64,
    /// Never boost a band whose speech-to-noise ratio is below this; above
    /// it, boost at most (SNR - min_snr_db).
    pub min_snr_db: f64,
    /// Below this much speech the spectrum estimate is too noisy: skip.
    pub min_speech_seconds: f64,
    /// Deviations within +-this are the talker's own voice: left alone.
    pub deadband_db: f64,
    /// Scales the final correction curve (0 = off, 1 = full).
    pub strength: f64,
}

impl Default for ToneConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // Gentle on purpose: preserving the talker's identity matters
            // more than making every mic sound the same. Boosts are also
            // capped by band SNR and never applied above the bandwidth.
            max_cut_db: 6.0,
            max_boost_db: 4.0,
            min_snr_db: 10.0,
            min_speech_seconds: 8.0,
            deadband_db: 2.0,
            // Off by default: in a blind listening comparison the natural,
            // untouched tone was preferred. `--tone-strength` turns it on,
            // e.g. for thin-sounding Bluetooth takes.
            strength: 0.0,
        }
    }
}

/// Chain of all-pass filters that makes the (lopsided) speech waveform
/// more symmetric. Magnitude response is exactly flat; it only lowers the
/// peak-to-loudness ratio so the limiter has less to do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PhaseRotatorConfig {
    pub enabled: bool,
    pub freq_hz: f64,
    pub q: f64,
    pub sections: usize,
}

impl Default for PhaseRotatorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            freq_hz: 300.0,
            q: 0.5,
            sections: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BellConfig {
    pub enabled: bool,
    pub freq_hz: f64,
    pub q: f64,
    pub gain_db: f64,
}

impl Default for BellConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            freq_hz: 1000.0,
            q: 1.0,
            gain_db: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShelfConfig {
    pub enabled: bool,
    pub freq_hz: f64,
    /// RBJ shelf slope S; 1.0 is the steepest monotonic shelf.
    pub slope: f64,
    pub gain_db: f64,
}

impl Default for ShelfConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            freq_hz: 11_000.0,
            slope: 1.0,
            gain_db: 1.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdditiveEqConfig {
    pub presence: BellConfig,
    pub air: ShelfConfig,
}

impl Default for AdditiveEqConfig {
    fn default() -> Self {
        Self {
            // The "character" layer is off by default for the same reason:
            // the natural sound was preferred. Enable per run with
            // `--set post.additive_eq.presence.enabled=true` (and `.air`).
            presence: BellConfig {
                enabled: false,
                freq_hz: 4000.0,
                q: 1.0,
                gain_db: 2.0,
            },
            air: ShelfConfig {
                enabled: false,
                ..ShelfConfig::default()
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExpanderConfig {
    pub enabled: bool,
    /// Pick the threshold from the file's own level histogram
    /// (noise floor + `auto_margin_db`, kept well below speech).
    pub auto_threshold: bool,
    /// Used when `auto_threshold` is false. dBFS RMS at the working level.
    pub threshold_db: f64,
    pub auto_margin_db: f64,
    /// Gate opens at `threshold`, closes only below `threshold - hysteresis`.
    pub hysteresis_db: f64,
    /// Stay open this long after the level drops (protects word tails).
    pub hold_ms: f64,
    /// Downward expansion ratio, 2.0 means 1:2 (1 dB under -> 2 dB down).
    pub ratio: f64,
    /// Maximum attenuation. Keeps it an expander, not a hard gate.
    pub range_db: f64,
    /// Time for the gain to rise when speech starts.
    pub open_ms: f64,
    /// Time for the gain to fall after hold expires (word tails decay).
    pub close_ms: f64,
    pub lookahead_ms: f64,
    pub detector_ms: f64,
}

impl Default for ExpanderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_threshold: true,
            threshold_db: -50.0,
            auto_margin_db: 6.0,
            hysteresis_db: 4.0,
            hold_ms: 150.0,
            ratio: 2.0,
            range_db: 8.0,
            open_ms: 2.0,
            close_ms: 150.0,
            lookahead_ms: 10.0,
            detector_ms: 10.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompressorConfig {
    pub enabled: bool,
    /// dBFS RMS at the working level.
    pub threshold_db: f64,
    pub ratio: f64,
    /// Total width of the soft knee in dB (centred on the threshold).
    pub knee_db: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    /// Averaging time of the RMS level detector.
    pub rms_ms: f64,
    pub lookahead_ms: f64,
    /// Measure loudness before/after and add back exactly what was lost.
    pub auto_makeup: bool,
    /// Used when `auto_makeup` is false.
    pub makeup_db: f64,
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_db: -30.0,
            ratio: 3.0,
            knee_db: 6.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            rms_ms: 5.0,
            lookahead_ms: 10.0,
            auto_makeup: true,
            makeup_db: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DeesserConfig {
    pub enabled: bool,
    /// Detector band (sibilance lives around 5-8 kHz).
    pub detector_low_hz: f64,
    pub detector_high_hz: f64,
    /// Everything above this is turned down while an "s" is detected.
    pub split_hz: f64,
    /// dBFS RMS of the detector band, at the working level.
    pub threshold_db: f64,
    pub ratio: f64,
    pub max_reduction_db: f64,
    pub attack_ms: f64,
    pub release_ms: f64,
    pub rms_ms: f64,
    pub lookahead_ms: f64,
}

impl Default for DeesserConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detector_low_hz: 5000.0,
            detector_high_hz: 8000.0,
            split_hz: 4500.0,
            threshold_db: -32.0,
            ratio: 4.0,
            max_reduction_db: 6.0,
            attack_ms: 1.0,
            release_ms: 60.0,
            rms_ms: 2.0,
            lookahead_ms: 3.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoudnessConfig {
    /// Integrated loudness target (EBU R128 / ITU-R BS.1770).
    pub target_lufs: f64,
    /// Never boost more than this (a silent or noise-only file would
    /// otherwise be turned up by 60 dB).
    pub max_gain_db: f64,
    /// Stop iterating once within this many LU of the target.
    pub tolerance_lu: f64,
    /// Measure as if the mono file is played on both stereo speakers
    /// (+3 dB). Off: plain single-channel BS.1770, as ffmpeg/pyloudnorm do.
    pub dual_mono: bool,
}

impl Default for LoudnessConfig {
    fn default() -> Self {
        Self {
            target_lufs: -14.0,
            max_gain_db: 40.0,
            tolerance_lu: 0.05,
            dual_mono: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LimiterConfig {
    pub enabled: bool,
    pub ceiling_dbtp: f64,
    pub lookahead_ms: f64,
    pub release_ms: f64,
    /// Ceiling used when the output is a video: lossy AAC/Opus encoding
    /// adds a few tenths of a dB of overshoot, so leave more room.
    pub video_ceiling_dbtp: f64,
    /// Low-pass the limiter input here (8th-order Butterworth) so the
    /// true-peak guarantee holds for any meter; 0 disables it.
    pub bandlimit_hz: f64,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ceiling_dbtp: -1.0,
            lookahead_ms: 5.0,
            release_ms: 80.0,
            video_ceiling_dbtp: -1.5,
            bandlimit_hz: 20_000.0,
        }
    }
}

impl PostConfig {
    pub fn validate(&self, sample_rate: f64) -> Result<(), ConfigError> {
        ensure_range(self.working_lufs, -40.0, -10.0, "post.working_lufs")?;

        let pr = &self.phase_rotator;
        ensure_freq(pr.freq_hz, sample_rate, "post.phase_rotator.freq_hz")?;
        ensure_range(pr.q, 0.1, 10.0, "post.phase_rotator.q")?;
        ensure(pr.sections <= 16, "post.phase_rotator.sections", || {
            format!("{} must be at most 16", pr.sections)
        })?;

        let t = &self.tone_correction;
        ensure_range(t.max_cut_db, 0.0, 12.0, "post.tone_correction.max_cut_db")?;
        ensure_range(
            t.max_boost_db,
            0.0,
            12.0,
            "post.tone_correction.max_boost_db",
        )?;
        ensure_range(t.min_snr_db, 0.0, 40.0, "post.tone_correction.min_snr_db")?;
        ensure_range(t.deadband_db, 0.0, 6.0, "post.tone_correction.deadband_db")?;
        ensure_range(t.strength, 0.0, 1.0, "post.tone_correction.strength")?;
        ensure_range(
            t.min_speech_seconds,
            1.0,
            600.0,
            "post.tone_correction.min_speech_seconds",
        )?;

        let eq = &self.subtractive_eq;
        ensure_freq(eq.freq_hz, sample_rate, "post.subtractive_eq.freq_hz")?;
        ensure_range(eq.q, 0.1, 20.0, "post.subtractive_eq.q")?;
        ensure_range(eq.gain_db, -24.0, 24.0, "post.subtractive_eq.gain_db")?;

        let p = &self.additive_eq.presence;
        ensure_freq(p.freq_hz, sample_rate, "post.additive_eq.presence.freq_hz")?;
        ensure_range(p.q, 0.1, 20.0, "post.additive_eq.presence.q")?;
        ensure_range(p.gain_db, -24.0, 24.0, "post.additive_eq.presence.gain_db")?;
        let a = &self.additive_eq.air;
        ensure_freq(a.freq_hz, sample_rate, "post.additive_eq.air.freq_hz")?;
        ensure_range(a.slope, 0.1, 1.0, "post.additive_eq.air.slope")?;
        ensure_range(a.gain_db, -24.0, 24.0, "post.additive_eq.air.gain_db")?;

        let e = &self.expander;
        ensure_range(e.threshold_db, -100.0, 0.0, "post.expander.threshold_db")?;
        ensure_range(e.auto_margin_db, 0.0, 30.0, "post.expander.auto_margin_db")?;
        ensure_range(e.hysteresis_db, 0.0, 20.0, "post.expander.hysteresis_db")?;
        ensure_range(e.hold_ms, 0.0, 2000.0, "post.expander.hold_ms")?;
        ensure_range(e.ratio, 1.0, 10.0, "post.expander.ratio")?;
        ensure_range(e.range_db, 0.0, 60.0, "post.expander.range_db")?;
        ensure_range(e.open_ms, 0.0, 500.0, "post.expander.open_ms")?;
        ensure_range(e.close_ms, 0.0, 5000.0, "post.expander.close_ms")?;
        ensure_range(e.lookahead_ms, 0.0, 100.0, "post.expander.lookahead_ms")?;
        ensure_range(e.detector_ms, 0.1, 500.0, "post.expander.detector_ms")?;

        let c = &self.compressor;
        ensure_range(c.threshold_db, -80.0, 0.0, "post.compressor.threshold_db")?;
        ensure_range(c.ratio, 1.0, 100.0, "post.compressor.ratio")?;
        ensure_range(c.knee_db, 0.0, 24.0, "post.compressor.knee_db")?;
        ensure_range(c.attack_ms, 0.0, 500.0, "post.compressor.attack_ms")?;
        ensure_range(c.release_ms, 0.0, 5000.0, "post.compressor.release_ms")?;
        ensure_range(c.rms_ms, 0.1, 500.0, "post.compressor.rms_ms")?;
        ensure_range(c.lookahead_ms, 0.0, 100.0, "post.compressor.lookahead_ms")?;
        ensure_range(c.makeup_db, -24.0, 24.0, "post.compressor.makeup_db")?;

        let d = &self.deesser;
        ensure_freq(
            d.detector_low_hz,
            sample_rate,
            "post.deesser.detector_low_hz",
        )?;
        ensure_freq(
            d.detector_high_hz,
            sample_rate,
            "post.deesser.detector_high_hz",
        )?;
        ensure(
            d.detector_low_hz < d.detector_high_hz,
            "post.deesser.detector_low_hz",
            || "must be below detector_high_hz".into(),
        )?;
        ensure_freq(d.split_hz, sample_rate, "post.deesser.split_hz")?;
        ensure_range(d.threshold_db, -80.0, 0.0, "post.deesser.threshold_db")?;
        ensure_range(d.ratio, 1.0, 100.0, "post.deesser.ratio")?;
        ensure_range(
            d.max_reduction_db,
            0.0,
            24.0,
            "post.deesser.max_reduction_db",
        )?;
        ensure_range(d.attack_ms, 0.0, 100.0, "post.deesser.attack_ms")?;
        ensure_range(d.release_ms, 0.0, 1000.0, "post.deesser.release_ms")?;
        ensure_range(d.rms_ms, 0.1, 100.0, "post.deesser.rms_ms")?;
        ensure_range(d.lookahead_ms, 0.0, 50.0, "post.deesser.lookahead_ms")?;

        let l = &self.loudness;
        ensure_range(l.target_lufs, -40.0, -5.0, "post.loudness.target_lufs")?;
        ensure_range(l.max_gain_db, 0.0, 60.0, "post.loudness.max_gain_db")?;
        ensure_range(l.tolerance_lu, 0.01, 1.0, "post.loudness.tolerance_lu")?;

        let lim = &self.limiter;
        ensure_range(lim.ceiling_dbtp, -20.0, 0.0, "post.limiter.ceiling_dbtp")?;
        ensure_range(lim.lookahead_ms, 0.5, 50.0, "post.limiter.lookahead_ms")?;
        ensure_range(lim.release_ms, 1.0, 2000.0, "post.limiter.release_ms")?;
        ensure_range(
            lim.video_ceiling_dbtp,
            -20.0,
            0.0,
            "post.limiter.video_ceiling_dbtp",
        )?;
        if lim.bandlimit_hz != 0.0 {
            ensure_freq(lim.bandlimit_hz, sample_rate, "post.limiter.bandlimit_hz")?;
            ensure_range(
                lim.bandlimit_hz,
                8000.0,
                24_000.0,
                "post.limiter.bandlimit_hz",
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        PreConfig::default().validate(48_000.0).unwrap();
        PostConfig::default().validate(48_000.0).unwrap();
    }

    #[test]
    fn partial_toml_keeps_defaults() {
        let cfg: PostConfig = toml::from_str("[compressor]\nratio = 4.0\n").unwrap();
        assert_eq!(cfg.compressor.ratio, 4.0);
        assert_eq!(cfg.compressor.attack_ms, 10.0);
        assert_eq!(cfg.loudness.target_lufs, -14.0);
    }

    #[test]
    fn typo_is_rejected() {
        let err = toml::from_str::<PostConfig>("[compressor]\nratoi = 4.0\n").unwrap_err();
        assert!(err.to_string().contains("ratoi"), "{err}");
    }

    #[test]
    fn hum_mode_parses_numbers_as_strings() {
        let cfg: PreConfig = toml::from_str("[hum]\nmode = \"60\"\n").unwrap();
        assert_eq!(cfg.hum.mode, HumMode::Hz60);
    }

    #[test]
    fn out_of_range_values_are_reported_with_field() {
        let mut cfg = PostConfig::default();
        cfg.compressor.ratio = 0.5;
        assert_eq!(
            cfg.validate(48_000.0).unwrap_err().field,
            "post.compressor.ratio"
        );
        let mut cfg = PostConfig::default();
        cfg.limiter.ceiling_dbtp = f64::NAN;
        assert_eq!(
            cfg.validate(48_000.0).unwrap_err().field,
            "post.limiter.ceiling_dbtp"
        );
        let mut cfg = PreConfig::default();
        cfg.highpass.freq_hz = 30_000.0;
        assert!(cfg.validate(48_000.0).is_err());
    }
}
