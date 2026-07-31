//! Export-voice command implementation
//!
//! Ports upstream's `export_voice`: convert an audio prompt (or every audio
//! file in a directory) into a `.safetensors` voice state that loads much
//! faster than re-encoding the audio, and is interchangeable with the Python
//! implementation.

use anyhow::{Context, Result};
use clap::Parser;
use pocket_tts::{TTSModel, export_model_state};
use std::path::{Path, PathBuf};

use crate::voice::resolve_voice;

#[derive(Parser, Debug)]
pub struct ExportVoiceArgs {
    /// Audio file (or directory of audio files) to convert and export
    pub audio_path: PathBuf,

    /// Output .safetensors file (or directory when the input is a directory)
    pub export_path: PathBuf,

    /// Language for the TTS model (english, french_24l, ...); incompatible
    /// with --config. Default: english
    #[arg(long)]
    pub language: Option<String>,

    /// Path to a locally-saved model config .yaml file; incompatible with
    /// --language
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Back-compat alias of --language
    #[arg(long, hide = true)]
    pub variant: Option<String>,

    /// Use Metal acceleration (macOS only)
    #[arg(long)]
    pub use_metal: bool,

    /// Suppress all output except errors
    #[arg(short, long)]
    pub quiet: bool,
}

pub fn run(args: ExportVoiceArgs) -> Result<()> {
    let device = if args.use_metal {
        #[cfg(feature = "metal")]
        {
            candle_core::Device::new_metal(0)?
        }
        #[cfg(not(feature = "metal"))]
        {
            anyhow::bail!("Metal feature not enabled. Rebuild with --features metal");
        }
    } else {
        candle_core::Device::Cpu
    };

    let model_spec = super::generate::resolve_model_spec(
        args.language.as_deref(),
        args.config.as_deref(),
        args.variant.as_deref(),
    )?;

    if !args.quiet {
        println!("Loading model {model_spec}...");
    }
    let model = TTSModel::load_with_params_device(
        &model_spec,
        None,
        pocket_tts::config::defaults::LSD_DECODE_STEPS,
        pocket_tts::config::defaults::EOS_THRESHOLD,
        None,
        &device,
    )?;

    if args.audio_path.is_dir() {
        std::fs::create_dir_all(&args.export_path)?;
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&args.audio_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| is_audio_file(p))
            .collect();
        entries.sort();
        if entries.is_empty() {
            anyhow::bail!("no audio files found in {:?}", args.audio_path);
        }
        for path in entries {
            let dest = args
                .export_path
                .join(path.file_stem().unwrap_or_default())
                .with_extension("safetensors");
            export_one(&model, &path, &dest, args.quiet)?;
        }
        Ok(())
    } else {
        export_one(&model, &args.audio_path, &args.export_path, args.quiet)
    }
}

fn export_one(model: &TTSModel, audio: &Path, dest: &Path, quiet: bool) -> Result<()> {
    let state = resolve_voice(model, Some(&audio.to_string_lossy()))
        .with_context(|| format!("failed to build voice state from {audio:?}"))?;
    export_model_state(&state, dest)?;
    if !quiet {
        println!("Exported {audio:?} -> {dest:?}");
    }
    Ok(())
}

fn is_audio_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("wav" | "wave" | "mp3" | "flac" | "ogg" | "m4a")
    )
}
