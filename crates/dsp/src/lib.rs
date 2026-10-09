//! Classic (non-generative) DSP for making laptop / Bluetooth-mic speech
//! sound production-ready.
//!
//! Signal flow, as orchestrated by the `mic-enhance` CLI:
//!
//! ```text
//! io::load_wav ─► pre::process ─► [restore: ML / rnnoise / none] ─► chain::process ─► io::write_wav
//!   (any WAV ->      (DC, 80 Hz HPF,                                  (EQ, expander, compressor,
//!    48 kHz mono)     hum notches)                                     de-esser, loudness, limiter)
//! ```
//!
//! Every stage works on a whole in-memory `&mut [f32]` buffer at 48 kHz.
//! Being offline lets stages look ahead freely (lookahead dynamics,
//! zero-phase filtering, two-pass loudness) with no latency cost.

pub mod biquad;
pub mod chain;
pub mod config;
pub mod dynamics;
pub mod hum;
pub mod io;
pub mod limiter;
pub mod loudness;
pub mod pre;
pub mod rnnoise;
pub mod tone;
pub mod truepeak;
pub mod util;

pub use io::SAMPLE_RATE;
