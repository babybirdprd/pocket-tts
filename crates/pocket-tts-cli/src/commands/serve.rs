//! Serve command implementation
//!
//! Provides `pocket-tts serve` for HTTP API server.

use anyhow::Result;
use clap::{ArgAction, Parser, ValueEnum};
use owo_colors::OwoColorize;

use crate::voice::PREDEFINED_VOICES;
use pocket_tts::config::defaults;

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

    /// Default voice for API requests (can be overridden per-request).
    /// If left empty, the language's recommended voice is used.
    #[arg(long, default_value = "")]
    pub voice: String,

    /// Language model to load. One of: english, italian, french_24l, german,
    /// spanish, portuguese, etc. Mutually exclusive with `--variant`.
    #[arg(long, default_value = defaults::DEFAULT_LANGUAGE, conflicts_with = "variant")]
    pub language: String,

    /// Legacy: directly select a model YAML stem (e.g. "b6369a24").
    /// Mutually exclusive with `--language`.
    #[arg(long)]
    pub variant: Option<String>,

    /// Sampling temperature
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// LSD decode steps
    #[arg(long, default_value = "1")]
    pub lsd_decode_steps: usize,

    /// EOS threshold
    #[arg(long, default_value = "-4.0")]
    pub eos_threshold: f32,

    /// Use simulated int8 quantization for inference
    #[arg(long)]
    pub quantized: bool,

    /// Maximum number of resolved voice states to keep in server LRU cache.
    #[arg(long, default_value_t = 64)]
    pub voice_cache_capacity: usize,

    /// Comma-separated voices to prewarm at startup (e.g. "alba,marius").
    #[arg(long, default_value = "alba")]
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

impl ServeArgs {
    /// Resolve which YAML stem to load. Returns the chosen stem and, when in
    /// language mode, the language string for voice resolution.
    pub fn resolved_stem(&self) -> (String, Option<&str>) {
        match &self.variant {
            Some(v) => (v.clone(), None),
            None => (self.language.clone(), Some(self.language.as_str())),
        }
    }

    /// Resolve which voice spec to use as the default. Empty `--voice`
    /// becomes the language's recommended voice.
    pub fn resolved_default_voice(&self) -> String {
        if self.voice.trim().is_empty() {
            let (_, lang) = self.resolved_stem();
            pocket_tts::config::language_defaults::default_voice(lang).to_string()
        } else {
            self.voice.clone()
        }
    }
}

pub async fn run(args: ServeArgs) -> Result<()> {
    print_banner();

    let (stem, _lang) = args.resolved_stem();
    println!("{} Loading model: {}", "->".cyan(), stem.yellow());
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Helper to construct ServeArgs with parser defaults applied.
    fn parse_args(extra: &[&str]) -> ServeArgs {
        let mut argv = vec!["test"];
        argv.extend_from_slice(extra);
        ServeArgs::try_parse_from(argv).expect("ServeArgs should parse")
    }

    #[test]
    fn test_default_resolves_to_english_language() {
        let args = parse_args(&[]);
        let (stem, lang) = args.resolved_stem();
        assert_eq!(stem, "english");
        assert_eq!(lang, Some("english"));
        assert_eq!(args.resolved_default_voice(), "alba");
    }

    #[test]
    fn test_explicit_language_italian() {
        let args = parse_args(&["--language", "italian"]);
        let (stem, lang) = args.resolved_stem();
        assert_eq!(stem, "italian");
        assert_eq!(lang, Some("italian"));
        assert_eq!(args.resolved_default_voice(), "giovanni");
    }

    #[test]
    fn test_legacy_variant_disables_language_aware_voice() {
        let args = parse_args(&["--variant", "b6369a24"]);
        let (stem, lang) = args.resolved_stem();
        assert_eq!(stem, "b6369a24");
        assert!(lang.is_none());
        // empty voice -> default voice fallback (no language => alba)
        assert_eq!(args.resolved_default_voice(), "alba");
    }

    #[test]
    fn test_language_and_variant_conflict() {
        let res = ServeArgs::try_parse_from(["test", "--language", "italian", "--variant", "b6369a24"]);
        assert!(res.is_err(), "language and variant must be mutually exclusive");
    }

    #[test]
    fn test_explicit_voice_overrides_language_default() {
        let args = parse_args(&["--language", "italian", "--voice", "alba"]);
        assert_eq!(args.resolved_default_voice(), "alba");
    }
}
