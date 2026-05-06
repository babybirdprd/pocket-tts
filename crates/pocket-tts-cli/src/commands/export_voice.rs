//! `export-voice` command: encode an audio prompt and save the resulting
//! per-layer KV-cache state to a `.safetensors` file.
//!
//! Mirrors Python's `pocket-tts export-voice` (see
//! `pocket_tts/main.py::export_voice`). The output format matches Python's
//! `export_model_state`, so files produced by this command can be imported
//! by both the Rust and Python implementations.

use anyhow::{Context, Result};
use clap::Parser;
use owo_colors::OwoColorize;
use pocket_tts::{TTSModel, config::defaults, voice_state::export_model_state_to_file};
use std::path::PathBuf;

#[derive(Parser, Debug)]
pub struct ExportVoiceArgs {
    /// Path to the input audio file (.wav). Voice cloning will encode this
    /// audio and capture the resulting flow-LM state.
    pub audio_path: PathBuf,

    /// Output `.safetensors` path. The format matches Python's
    /// `export_model_state` and is round-trip compatible with Rust's
    /// `voice_state::import_model_state_from_file`.
    pub export_path: PathBuf,

    /// Language model to load. One of the bundled language YAML stems.
    /// Mutually exclusive with `--variant`.
    #[arg(long, default_value = defaults::DEFAULT_LANGUAGE, conflicts_with = "variant")]
    pub language: String,

    /// Legacy: directly select a model YAML stem (e.g. "b6369a24").
    #[arg(long)]
    pub variant: Option<String>,

    /// Suppress informational output.
    #[arg(short, long)]
    pub quiet: bool,
}

pub fn run(args: ExportVoiceArgs) -> Result<()> {
    let stem = args.variant.clone().unwrap_or_else(|| args.language.clone());

    if !args.quiet {
        println!("{} Loading: {}", "▶".cyan(), stem.yellow());
    }
    let model = TTSModel::load(&stem)
        .with_context(|| format!("Failed to load model for stem '{}'", stem))?;

    if !args.quiet {
        println!(
            "  {} Model loaded (sample rate: {}Hz)",
            "✓".green(),
            model.sample_rate
        );
        println!(
            "{} Encoding voice: {}",
            "▶".cyan(),
            args.audio_path.display().yellow()
        );
    }

    let state = model
        .get_voice_state(&args.audio_path)
        .with_context(|| format!("Failed to encode {}", args.audio_path.display()))?;

    if !args.quiet {
        println!(
            "  {} Encoded {} module(s)",
            "✓".green(),
            state.len()
        );
        println!(
            "{} Saving to: {}",
            "▶".cyan(),
            args.export_path.display().yellow()
        );
    }

    // Use "flow_lm" as the prefix so produced files match upstream's layout.
    export_model_state_to_file(&state, &args.export_path, "flow_lm")
        .with_context(|| format!("Failed to write {}", args.export_path.display()))?;

    if !args.quiet {
        println!(
            "  {} Voice exported to {}",
            "✓".green().bold(),
            args.export_path.display().cyan()
        );
    }

    Ok(())
}
