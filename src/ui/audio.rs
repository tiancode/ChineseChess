//! Procedurally-synthesised piece-drop sound. No asset files: the waveform is
//! generated in code so the binary stays self-contained and works offline.
//! Audio is best-effort — if no output device is available the game runs
//! silently (see [`Audio::new`] returning `None`).

use rodio::buffer::SamplesBuffer;
use rodio::{OutputStream, OutputStreamHandle};

/// Output sample rate for the synthesised clip.
const SR: u32 = 44_100;

/// Holds the audio device alive for the lifetime of the app. `OutputStream`
/// is `!Send`, which is fine: `XiangqiApp` lives on (and plays from) the eframe
/// main thread; only the search engine crosses threads.
pub struct Audio {
    // Dropping the stream closes the device, so it must outlive `handle`.
    _stream: OutputStream,
    handle: OutputStreamHandle,
    // The synthesis is fully deterministic, so the two clips are built once
    // here and cloned per play instead of re-synthesised every move.
    quiet: Vec<f32>,
    capture: Vec<f32>,
}

impl Audio {
    /// Open the default output device. Returns `None` (silent fallback) if no
    /// device is available, so callers never have to handle audio failure.
    pub fn new() -> Option<Audio> {
        let (stream, handle) = OutputStream::try_default().ok()?;
        Some(Audio {
            _stream: stream,
            handle,
            quiet: move_sound_samples(SR, false),
            capture: move_sound_samples(SR, true),
        })
    }

    /// Play one piece-drop "clack". Fire-and-forget: `play_raw` detaches the
    /// source and any error is ignored (a missed sound must never disrupt play).
    pub fn play_move(&self, capture: bool) {
        // `SamplesBuffer<f32>` is `Source<Item = f32> + Send + 'static`, which
        // is exactly what `play_raw` wants; it detaches and plays to the end.
        // The buffer must be owned, so clone the cached clip (cheap memcpy,
        // no synthesis).
        let clip = if capture { &self.capture } else { &self.quiet };
        let buf = SamplesBuffer::new(1, SR, clip.clone());
        let _ = self.handle.play_raw(buf);
    }
}

/// Synthesise a short wooden piece-drop sound as mono `f32` samples.
///
/// Pure and deterministic (the noise transient uses a fixed-seed LCG) so it is
/// unit-testable without an audio device. The clip is a brief filtered-noise
/// attack plus exponentially-decaying sine partials. `capture` swaps the voice:
///
/// - quiet move (~85 ms): a light "tick" — short noise burst, crisp 1100 Hz
///   body over a 210 Hz woody resonance, normalised to ~0.5.
/// - capture (~130 ms): a heavier "thwack" — a longer, louder collision
///   transient, an added bright 1600 Hz crack, and a lower 150 Hz body for
///   weight, normalised hotter (~0.65) so it clearly reads as different.
fn move_sound_samples(sample_rate: u32, capture: bool) -> Vec<f32> {
    let sr = sample_rate as f32;
    let (dur, transient_tau, transient_gain) = if capture {
        (0.13, 0.012, 0.85)
    } else {
        (0.085, 0.004, 0.55)
    };
    let n = (sr * dur) as usize;
    let mut out = Vec::with_capacity(n);

    // Deterministic white noise (SplitMix64-style), mapped to [-1, 1).
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut noise = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    };

    let two_pi = std::f32::consts::TAU;
    for i in 0..n {
        let t = i as f32 / sr;
        // Collision/contact noise: louder and longer for a capture.
        let transient = transient_gain * noise() * (-t / transient_tau).exp();
        let mut s = if capture {
            // Bright crack over a heavy low body.
            let crack = (two_pi * 1600.0 * t).sin() * (-t / 0.03).exp();
            let body = (two_pi * 150.0 * t).sin() * (-t / 0.07).exp();
            transient + 0.5 * crack + 0.55 * body
        } else {
            let tick = (two_pi * 1100.0 * t).sin() * (-t / 0.022).exp();
            let thunk = (two_pi * 210.0 * t).sin() * (-t / 0.05).exp();
            transient + 0.5 * tick + 0.45 * thunk
        };
        // ~1.5 ms raised attack so the onset itself is not a hard click.
        s *= (t / 0.0015).min(1.0);
        out.push(s);
    }

    // Peak-normalise: a capture sits a touch hotter so it reads as heavier.
    let target = if capture { 0.65 } else { 0.5 };
    let peak = out.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    if peak > 1e-6 {
        let g = target / peak;
        for s in &mut out {
            *s *= g;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_are_finite_bounded_and_nonempty() {
        for capture in [false, true] {
            let s = move_sound_samples(44_100, capture);
            assert!(!s.is_empty());
            assert!(s.iter().all(|x| x.is_finite()));
            let peak = s.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
            assert!(peak <= 1.0, "peak {peak} should stay within [-1, 1]");
            assert!(peak > 0.4, "peak {peak} should be normalised");
        }
    }

    #[test]
    fn synthesis_is_deterministic() {
        for capture in [false, true] {
            assert_eq!(
                move_sound_samples(22_050, capture),
                move_sound_samples(22_050, capture)
            );
        }
    }

    #[test]
    fn capture_sound_differs_from_quiet_move() {
        let quiet = move_sound_samples(44_100, false);
        let cap = move_sound_samples(44_100, true);
        // The capture clip is longer and hotter, so it is plainly distinct.
        assert!(cap.len() > quiet.len(), "capture clip should be longer");
        let pk = |v: &[f32]| v.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        assert!(pk(&cap) > pk(&quiet), "capture should be louder");
    }
}
