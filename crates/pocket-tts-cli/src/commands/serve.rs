//! Serve command implementation
//!
//! Provides `pocket-tts serve` for HTTP API server.

use anyhow::Result;
use clap::{ArgAction, Parser, ValueEnum};
use owo_colors::OwoColorize;

use crate::voice::PREDEFINED_VOICES;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum UiMode {
    /// Existing server-backed React UI.
    Standard,
    /// Experimental browser-side WASM inference UI.
    WasmExperimental,
}

impl UiMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::WasmExperimental => "wasm-experimental",
        }
    }
}

#[derive(Parser, Debug, Clone)]
pub struct ServeArgs {
    /// Host address to bind (default: 127.0.0.1)
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port number to listen on (default: 8000)
    #[arg(short, long, default_value_t = 8000)]
    pub port: u16,

    /// Default voice for API requests (can be overridden per-request);
    /// defaults to the model's language voice (estelle for French, alba
    /// for English, ...)
    #[arg(long)]
    pub voice: Option<String>,

    /// Language for the TTS model (english, french_24l, ...); incompatible
    /// with --config. Default: english
    #[arg(long)]
    pub language: Option<String>,

    /// Path to a locally-saved model config .yaml file; incompatible with
    /// --language
    #[arg(long)]
    pub config: Option<std::path::PathBuf>,

    /// Back-compat alias of --language
    #[arg(long, hide = true)]
    pub variant: Option<String>,

    /// Sampling temperature; defaults to the model's recommended
    /// temperature from its config
    #[arg(long)]
    pub temperature: Option<f32>,

    /// LSD decode steps
    #[arg(long, default_value = "1")]
    pub lsd_decode_steps: usize,

    /// EOS threshold
    #[arg(long, default_value = "-4.0")]
    pub eos_threshold: f32,

    /// Apply int8 quantization to reduce memory usage (upstream --quantize;
    /// --quantized kept as an alias)
    #[arg(long, alias = "quantized")]
    pub quantize: bool,

    /// Maximum number of resolved voice states to keep in server LRU cache.
    #[arg(long, default_value_t = 64)]
    pub voice_cache_capacity: usize,

    /// Comma-separated voices to prewarm at startup (e.g. "alba,marius").
    /// The default voice is always prewarmed; empty prewarms nothing else.
    #[arg(long, default_value = "")]
    pub prewarm_voices: String,

    /// Run a tiny startup warmup generation to reduce first-request latency.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub warmup: bool,

    /// Override OMP_NUM_THREADS before model load.
    #[arg(long)]
    pub omp_threads: Option<usize>,

    /// Override MKL_NUM_THREADS before model load.
    #[arg(long)]
    pub mkl_threads: Option<usize>,

    /// Web UI mode to serve.
    #[arg(long, value_enum, default_value_t = UiMode::Standard)]
    pub ui: UiMode,
}

pub async fn run(args: ServeArgs) -> Result<()> {
    print_banner();

    let model_spec = crate::commands::generate::resolve_model_spec(
        args.language.as_deref(),
        args.config.as_deref(),
        args.variant.as_deref(),
    )?;
    println!("{} Loading model: {}", "->".cyan(), model_spec.yellow());

    println!("{} UI mode: {}", "->".cyan(), args.ui.as_str().yellow());

    let server_args = args.clone();
    crate::server::start_server(server_args).await
}

fn print_banner() {
    println!();
    println!(
        "  {}  {} {}",
        "[]".bold(),
        "Pocket TTS".bold().cyan(),
        "API Server".bold()
    );
    println!(
        "      {} {}",
        "Rust/Candle port".dimmed(),
        format!("v{}", env!("CARGO_PKG_VERSION")).dimmed()
    );
    println!();
}

/// Print endpoint information after server starts
pub fn print_endpoints(host: &str, port: u16, ui_mode: UiMode) {
    let base = format!("http://{}:{}", host, port);

    println!();
    println!(
        "  {} {}",
        "[ok]".green().bold(),
        "Server is running!".green().bold()
    );
    println!();
    println!("  {}", "Endpoints:".bold());
    println!(
        "    {} {}  {}",
        "GET".cyan(),
        format!("{}/", base).white(),
        "Web interface".dimmed()
    );
    println!(
        "    {} {}  {}",
        "GET".cyan(),
        format!("{}/health", base).white(),
        "Health check".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/generate", base).white(),
        "Generate audio (JSON body)".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/stream", base).white(),
        "Streaming generation".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/tts", base).white(),
        "Python-compatible endpoint (form data)".dimmed()
    );
    println!(
        "    {} {}  {}",
        "POST".yellow(),
        format!("{}/v1/audio/speech", base).white(),
        "OpenAI-compatible".dimmed()
    );
    println!();
    println!(
        "  {} Active web UI mode: {}",
        "Mode:".dimmed(),
        ui_mode.as_str().dimmed()
    );
    println!(
        "  {} Available voices: {}",
        "Voices:".dimmed(),
        PREDEFINED_VOICES.join(", ").dimmed()
    );
    println!();
    println!(
        "  {} curl -X POST {}/generate -H 'Content-Type: application/json' -d '{{\"text\": \"Hello world\"}}' --output test.wav",
        "Example:".dimmed(),
        base
    );
    println!();
}
