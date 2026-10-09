//! Pure-Rust RNNoise (`nnnoiseless`) denoiser: the offline fallback for
//! the restore stage when the Python ML stage is unavailable.
//!
//! RNNoise is a mask-based denoiser: a small recurrent network predicts a
//! gain for each of 22 frequency bands every 10 ms and the spectrum is
//! multiplied by those gains. It only ever attenuates, so it fits the
//! "clean" mode promise of never inventing audio.

use nnnoiseless::DenoiseState;

/// RNNoise works on 480-sample (10 ms) frames at 48 kHz.
pub const FRAME: usize = DenoiseState::FRAME_SIZE;
/// Output delay of the overlap-add synthesis, in samples (one frame).
/// Verified by `tests::output_is_time_aligned_with_input`.
pub const LATENCY: usize = FRAME;
/// RNNoise was trained on 16-bit PCM levels, not +-1.0 floats.
const PCM_SCALE: f32 = 32_768.0;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RnnoiseReport {
    /// Mean voice-activity probability reported by the network (0..1).
    pub mean_vad: f64,
}

/// Denoise a 48 kHz mono buffer. Output has the same length and timing
/// as the input (the one-frame delay is removed).
pub fn denoise(input: &[f32]) -> (Vec<f32>, RnnoiseReport) {
    let mut state = DenoiseState::new();
    // Pad so the delayed tail of the real signal still gets flushed out.
    let padded_len = (input.len() + LATENCY).div_ceil(FRAME) * FRAME;
    let mut out = Vec::with_capacity(padded_len);
    let mut frame_in = [0.0f32; FRAME];
    let mut frame_out = [0.0f32; FRAME];
    let mut vad_sum = 0.0f64;
    let frames = padded_len / FRAME;

    for f in 0..frames {
        let start = f * FRAME;
        for (i, slot) in frame_in.iter_mut().enumerate() {
            *slot = input.get(start + i).map_or(0.0, |&s| s * PCM_SCALE);
        }
        vad_sum += f64::from(state.process_frame(&mut frame_out, &frame_in));
        out.extend(frame_out.iter().map(|&s| s / PCM_SCALE));
    }

    let aligned = out[LATENCY..LATENCY + input.len()].to_vec();
    let report = RnnoiseReport {
        mean_vad: if frames > 0 {
            vad_sum / frames as f64
        } else {
            0.0
        },
    };
    (aligned, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_silent_inputs_are_fine() {
        assert!(denoise(&[]).0.is_empty());
        let (out, _) = denoise(&[0.0; 1234]);
        assert_eq!(out.len(), 1234);
        assert!(out.iter().all(|s| s.is_finite()));
    }

    /// RNNoise passes a frame through untouched when it considers it
    /// "silent" (very low energy), which makes the delay directly visible:
    /// a lone small click must come out exactly LATENCY samples later,
    /// i.e. at the same index after our alignment.
    #[test]
    fn output_is_time_aligned_with_input() {
        let mut x = vec![0.0f32; 48_00];
        x[2000] = 1e-5;
        let (out, _) = denoise(&x);
        let peak_at = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|(i, _)| i)
            .unwrap();
        assert!((peak_at as i64 - 2000).abs() <= 1, "peak at {peak_at}");
    }
}
