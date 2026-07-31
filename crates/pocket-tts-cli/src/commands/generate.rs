//! Generate command implementation
//!
//! Provides `pocket-tts generate` for text-to-speech synthesis.

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use owo_colors::OwoColorize;
use pocket_tts::TTSModel;
use std::path::PathBuf;

use crate::voice::{PREDEFINED_VOICES, default_voice_for, resolve_voice};

/// Default text shown when user runs without --text
pub const DEFAULT_TEXT: &str =
    "Hello world! I am Pocket TTS, running blazingly fast in Rust. I hope you'll like me.";

/// Per-language demo texts, matching Python's DEFAULT_TEXT_FOR_LANGUAGE
/// (matched as a substring of the variant name).
const DEFAULT_TEXT_FOR_LANGUAGE: &[(&str, &str)] = &[
    (
        "french",
        "Bonjour le monde. Je suis le TTS de poche de Kyutai. \
         Je suis assez rapide pour fonctionner sur de petits CPU. \
         J'espère que vous m'aimerez.",
    ),
    (
        "german",
        "Hallo Welt. Ich bin Pocket TTS von Kyutai. \
         Ich bin schnell genug, um auch auf kleinen CPUs zu laufen. \
         Ich hoffe, ich gefalle dir.",
    ),
    (
        "italian",
        "Ciao mondo. Sono il Pocket TTS di Kyutai. \
         Sono abbastanza veloce da funzionare su piccole CPU. \
         Spero che ti piacerò.",
    ),
    (
        "spanish",
        "Hola mundo. Soy el Pocket TTS de Kyutai. \
         Soy lo suficientemente rápido para funcionar en pequeñas CPU. \
         Espero que te guste.",
    ),
    (
        "portuguese",
        "Olá mundo. Eu sou o Pocket TTS da Kyutai. \
         Sou rápido o suficiente para rodar em CPUs pequenas. \
         Espero que você goste de mim.",
    ),
];

/// Pick the demo text for a variant when the user gives no --text.
fn default_text_for_variant(variant: &str) -> &'static str {
    DEFAULT_TEXT_FOR_LANGUAGE
        .iter()
        .find(|(lang, _)| variant.contains(lang))
        .map(|(_, text)| *text)
        .unwrap_or(DEFAULT_TEXT)
}

/// Resolve the model spec (language name or config path) from the
/// upstream-style --language/--config arguments; --variant stays as a
/// back-compat alias of --language.
pub fn resolve_model_spec(
    language: Option<&str>,
    config: Option<&std::path::Path>,
    variant: Option<&str>,
) -> Result<String> {
    if config.is_some() && (language.is_some() || variant.is_some()) {
        anyhow::bail!("Cannot specify both config and language, please choose one or the other.");
    }
    if language.is_some() && variant.is_some() {
        anyhow::bail!("--variant is an alias of --language; pass only one of them.");
    }
    if let Some(path) = config {
        return Ok(path.to_string_lossy().into_owned());
    }
    let language = language.or(variant).unwrap_or(pocket_tts::config::defaults::DEFAULT_VARIANT);
    if language == "french" {
        anyhow::bail!(
            "For technical reasons, only a larger 24-layer model is available for French. \
             Please use the 'french_24l' language instead."
        );
    }
    Ok(language.to_string())
}

/// Read the text argument, resolving "-" to stdin like upstream.
pub fn resolve_text(text: Option<&str>, model_spec: &str) -> Result<String> {
    let text = match text {
        Some("-") => {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
        Some(t) => t.to_string(),
        None => default_text_for_variant(model_spec).to_string(),
    };
    if text.trim().is_empty() {
        anyhow::bail!("No input text received.");
    }
    Ok(text)
}

#[derive(Parser, Debug)]
pub struct GenerateArgs {
    /// Text to synthesize (defaults to a greeting in the model's language);
    /// "-" reads from stdin
    #[arg(short, long)]
    pub text: Option<String>,

    /// Voice for synthesis. Can be:
    /// - Predefined name: alba, marius, ... (estelle, giovanni, lola, juergen,
    ///   rafael for the language models); defaults to the model's language voice
    /// - Path to .wav file for voice cloning
    /// - Path to .safetensors embeddings file
    /// - HuggingFace URL: hf://owner/repo/file.wav
    #[arg(short, long)]
    pub voice: Option<String>,

    /// Output audio file path
    #[arg(short, long, default_value = "output.wav")]
    pub output: PathBuf,

    /// Language for the TTS model: english, english_2026-01, english_2026-04,
    /// french_24l, german(_24l), italian(_24l), portuguese(_24l), spanish(_24l).
    /// The historical "b6369a24" name (= english_2026-01) still works.
    /// Incompatible with --config. Default: english
    #[arg(long)]
    pub language: Option<String>,

    /// Path to a locally-saved model config .yaml file; incompatible with
    /// --language
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Back-compat alias of --language
    #[arg(long, hide = true)]
    pub variant: Option<String>,

    /// Maximum number of tokens per generation chunk
    #[arg(long, default_value_t = pocket_tts::config::defaults::MAX_TOKEN_PER_CHUNK)]
    pub max_tokens: usize,

    /// Sampling temperature (higher = more variation); defaults to the
    /// model's recommended temperature from its config
    #[arg(long)]
    pub temperature: Option<f32>,

    /// LSD decode steps (more steps = better quality, slower)
    #[arg(long, default_value = "1")]
    pub lsd_decode_steps: usize,

    /// EOS threshold (more negative = longer audio)
    #[arg(long, default_value = "-4.0")]
    pub eos_threshold: f32,

    /// Noise clamp value (optional)
    #[arg(long)]
    pub noise_clamp: Option<f32>,

    /// Frames to generate after EOS detection (optional, auto-estimated if not set)
    #[arg(long)]
    pub frames_after_eos: Option<usize>,

    /// Stream raw PCM audio to stdout (for piping to audio players)
    #[arg(long)]
    pub stream: bool,

    /// Apply int8 quantization to reduce memory usage (upstream --quantize;
    /// --quantized kept as an alias)
    #[arg(long, alias = "quantized")]
    pub quantize: bool,

    /// Use Metal acceleration (macOS only)
    #[arg(long)]
    pub use_metal: bool,

    /// Suppress all output except errors
    #[arg(short, long)]
    pub quiet: bool,
}

/// Print styled message (respects quiet mode)
macro_rules! info {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            println!($($arg)*);
        }
    };
}

pub fn run(args: GenerateArgs) -> Result<()> {
    let quiet = args.quiet || args.stream;

    // Print banner
    if !quiet {
        print_banner();
    }

    // Set up device
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

    if !quiet {
        println!("  {} Using device: {:?}", "▶".cyan(), device);
    }

    // Load model
    info!(quiet, "{} Loading model...", "▶".cyan());

    let model_spec = resolve_model_spec(
        args.language.as_deref(),
        args.config.as_deref(),
        args.variant.as_deref(),
    )?;

    let model = if args.quantize {
        TTSModel::load_quantized_with_params_device(
            &model_spec,
            args.temperature,
            args.lsd_decode_steps,
            args.eos_threshold,
            args.noise_clamp,
            &device,
        )?
    } else {
        TTSModel::load_with_params_device(
            &model_spec,
            args.temperature,
            args.lsd_decode_steps,
            args.eos_threshold,
            args.noise_clamp,
            &device,
        )?
    };

    info!(
        quiet,
        "  {} Model loaded (sample rate: {}Hz)",
        "✓".green(),
        model.sample_rate
    );

    // Resolve voice
    let voice_display = args
        .voice
        .clone()
        .unwrap_or_else(|| format!("{} (default)", default_voice_for(&model)));
    info!(
        quiet,
        "{} Using voice: {}",
        "▶".cyan(),
        voice_display.yellow()
    );

    let voice_state = resolve_voice(&model, args.voice.as_deref())?;

    info!(quiet, "  {} Voice ready", "✓".green());

    let origin = model.origin.clone().unwrap_or_else(|| model_spec.clone());
    let text = resolve_text(args.text.as_deref(), &origin)?;

    let options = pocket_tts::GenerateOptions {
        max_tokens: args.max_tokens,
        frames_after_eos: args.frames_after_eos,
    };

    // Generate
    if args.stream {
        run_streaming(&model, &text, &voice_state, options)
    } else {
        run_to_file(&model, &args, &text, &voice_state, options, quiet)
    }
}

/// Run streaming generation to stdout
fn run_streaming(
    model: &TTSModel,
    text: &str,
    voice_state: &pocket_tts::ModelState,
    options: pocket_tts::GenerateOptions,
) -> Result<()> {
    use std::io::Write;
    let mut stdout = std::io::stdout();

    for chunk_res in model.generate_stream_long_opts(text, voice_state, options) {
        let chunk = chunk_res?;
        // Convert tensor to 16-bit PCM
        let chunk = chunk.squeeze(0)?;
        let bytes = pocket_tts::audio::pcm_i16_le_bytes(&chunk)?;
        stdout.write_all(&bytes)?;
        stdout.flush()?;
    }

    Ok(())
}

/// Run generation to file with progress bar
fn run_to_file(
    model: &TTSModel,
    args: &GenerateArgs,
    text: &str,
    voice_state: &pocket_tts::ModelState,
    options: pocket_tts::GenerateOptions,
    quiet: bool,
) -> Result<()> {
    use candle_core::Tensor;

    info!(
        quiet,
        "{} Generating: \"{}\"",
        "▶".cyan(),
        truncate_text(text, 60).italic()
    );

    let total_steps = model.estimate_generation_steps(text) as u64;

    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(total_steps);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
                )
                .unwrap()
                .progress_chars("█▓░"),
        );
        pb.set_message("generating...");
        pb
    };

    let mut audio_chunks = Vec::new();
    let mut total_samples = 0;

    for chunk_res in model.generate_stream_long_opts(text, voice_state, options) {
        let chunk = chunk_res?;
        let dims = chunk.dims();
        let samples = if dims.len() == 2 { dims[1] } else { dims[0] };
        total_samples += samples;

        audio_chunks.push(chunk);
        pb.inc(1);
        pb.set_message(format!(
            "{:.2}s generated",
            total_samples as f32 / model.sample_rate as f32
        ));
    }

    pb.finish_and_clear();

    // Concatenate all audio chunks
    if audio_chunks.is_empty() {
        anyhow::bail!("No audio generated - text may be too short or invalid");
    }
    let audio = Tensor::cat(&audio_chunks, 2)?;
    let audio = audio.squeeze(0)?; // Remove batch dimension

    let dims = audio.dims();
    let num_samples = if dims.len() == 2 { dims[1] } else { dims[0] };
    let duration_sec = num_samples as f32 / model.sample_rate as f32;

    // Save to file
    info!(
        quiet,
        "{} Saving to: {}",
        "▶".cyan(),
        args.output.display().yellow()
    );
    pocket_tts::audio::write_wav(&args.output, &audio, model.sample_rate as u32)?;

    // Success message
    if !quiet {
        println!();
        println!(
            "  {} {}",
            "✓".green().bold(),
            "Audio generated successfully!".green().bold()
        );
        println!(
            "    Duration: {:.2}s ({} samples @ {}Hz)",
            duration_sec, num_samples, model.sample_rate
        );
        println!("    Output:   {}", args.output.display().cyan());
        println!();
        println!(
            "  {} {}",
            "💡".dimmed(),
            format!("Play with: ffplay -autoexit {:?}", args.output).dimmed()
        );
    }

    Ok(())
}

/// Print startup banner
fn print_banner() {
    println!();
    println!("  {}  {}", "🗣️".bold(), "Pocket TTS".bold().cyan());
    println!(
        "      {} {}",
        "Rust/Candle port".dimmed(),
        format!("v{}", env!("CARGO_PKG_VERSION")).dimmed()
    );
    println!();
}

/// Truncate text for display
fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len - 3])
    }
}

/// Print available voices (for help text)
pub fn available_voices_help() -> String {
    format!("Predefined voices: {}", PREDEFINED_VOICES.join(", "))
}
