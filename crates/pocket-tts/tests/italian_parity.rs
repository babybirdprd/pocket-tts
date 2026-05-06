//! Italian end-to-end parity gate (Phase 3).
//!
//! Validates that the structural changes from Phase 2 are correct by
//! actually loading the upstream Italian model from HuggingFace and
//! generating audio with the language's default voice (giovanni).
//!
//! This test requires `HF_TOKEN` to be set. When unset, every test
//! self-skips with a `println!` so workspace test runs don't fail.
//!
//! Run with:
//!   HF_TOKEN=... cargo test --release -p pocket-tts --test italian_parity -- --nocapture

use anyhow::Result;
use pocket_tts::TTSModel;

fn hf_token_set() -> bool {
    std::env::var("HF_TOKEN")
        .map(|s| !s.is_empty())
        .unwrap_or(false)
}

fn skip_if_no_token(test_name: &str) -> bool {
    if !hf_token_set() {
        eprintln!("[skip] {test_name}: HF_TOKEN not set");
        return true;
    }
    false
}

/// Phase 3.0: just load the Italian config + safetensors weights.
/// The point is to validate that the structural Phase 2 changes
/// (inner_dim/outer_dim downsample shapes, bos_before_voice loading,
/// speaker_proj_weight (1024, 32) shape) don't reject the safetensors.
#[test]
fn italian_model_loads() {
    if skip_if_no_token("italian_model_loads") {
        return;
    }
    let model = TTSModel::load("italian").expect("italian model should load");
    assert_eq!(model.sample_rate, 24000, "Italian config: 24 kHz");
    eprintln!(
        "Italian loaded: dim={} ldim={} sample_rate={}",
        model.dim, model.ldim, model.sample_rate
    );
}

/// Phase 3.1a: load the giovanni predefined voice via the new
/// language-aware path
/// `kyutai/pocket-tts-without-voice-cloning/languages/italian/embeddings/giovanni.safetensors`.
/// Failure means either: HF token lacks repo access, or the predefined
/// voice path / revision is wrong.
#[test]
fn italian_voice_giovanni_loads() {
    if skip_if_no_token("italian_voice_giovanni_loads") {
        return;
    }
    let model = TTSModel::load("italian").expect("italian model should load");

    // Re-implement the voice-resolution path inline to keep this test in
    // the `pocket-tts` crate (not pocket-tts-cli).
    let hf_path = "hf://kyutai/pocket-tts-without-voice-cloning/languages/italian/embeddings/giovanni.safetensors@e041936c75475d350b405bc870bcf7c22da4e9e6";
    let local = pocket_tts::weights::download_if_necessary(hf_path)
        .expect("giovanni embeddings should download");
    let _state = model
        .get_voice_state_from_prompt_file(&local)
        .expect("giovanni embeddings should load into a voice state");
}

/// Phase 3.1b: full end-to-end Italian generation.
/// Generates a short Italian phrase with the giovanni voice and
/// asserts the audio is non-trivial (length, amplitude, no NaN).
#[test]
fn italian_generates_audio_end_to_end() -> Result<()> {
    if skip_if_no_token("italian_generates_audio_end_to_end") {
        return Ok(());
    }
    let model = TTSModel::load("italian")?;

    let hf_path = "hf://kyutai/pocket-tts-without-voice-cloning/languages/italian/embeddings/giovanni.safetensors@e041936c75475d350b405bc870bcf7c22da4e9e6";
    let local = pocket_tts::weights::download_if_necessary(hf_path)?;
    let voice_state = model.get_voice_state_from_prompt_file(&local)?;

    // Default Italian greeting from language_defaults::default_text("italian").
    let text = "Ciao mondo. Sono il Pocket TTS di Kyutai.";
    let audio = model.generate(text, &voice_state)?;

    // Audio shape sanity: [C, T] after squeezing batch.
    let dims = audio.dims();
    assert!(
        dims.len() == 2,
        "expected [C, T] tensor, got dims {:?}",
        dims
    );
    let samples = dims[1];

    // Should be at least 1 second (the phrase is ~3s natural speech).
    let sr = model.sample_rate;
    let duration_sec = samples as f32 / sr as f32;
    assert!(
        duration_sec > 1.0,
        "audio too short: {:.2}s ({} samples @ {} Hz)",
        duration_sec,
        samples,
        sr
    );

    // Peak amplitude sanity: must not be silent and must not clip hard.
    let peak = audio
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        peak > 0.01,
        "audio is essentially silent (peak={:.4})",
        peak
    );
    assert!(
        peak.is_finite(),
        "audio peak is non-finite (peak={:?}); NaN/Inf in generation",
        peak
    );

    // Save to a path the developer can listen to manually.
    let out = std::env::temp_dir().join("pocket_tts_italian_parity.wav");
    pocket_tts::audio::write_wav(&out, &audio, sr as u32)?;
    eprintln!(
        "Italian e2e: {:.2}s, peak={:.4}, written to {}",
        duration_sec,
        peak,
        out.display()
    );

    Ok(())
}
