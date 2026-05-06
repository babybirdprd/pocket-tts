use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use crate::commands::serve::{ServeArgs, UiMode};
use pocket_tts::config::defaults;

#[derive(Parser, Debug, Clone)]
pub struct WasmDemoArgs {
    /// Host address to bind (default: 127.0.0.1)
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port to listen on.
    #[arg(short, long, default_value_t = 8080)]
    pub port: u16,

    /// Deprecated option retained for compatibility. Ignored.
    #[arg(long)]
    pub root: Option<PathBuf>,

    /// Deprecated option retained for compatibility. Ignored.
    #[arg(short, long)]
    pub models: Option<PathBuf>,
}

pub async fn run(args: WasmDemoArgs) -> Result<()> {
    println!(
        "[deprecated] `wasm-demo` is deprecated. Use: pocket-tts serve --ui wasm-experimental [--port {}]",
        args.port
    );
    if args.root.is_some() || args.models.is_some() {
        println!("[note] `--root` and `--models` are ignored by the new unified UI path.");
    }

    let serve_args = ServeArgs {
        host: args.host,
        port: args.port,
        voice: String::new(), // empty -> language default voice
        language: defaults::DEFAULT_LANGUAGE.to_string(),
        variant: None,
        temperature: 0.7,
        lsd_decode_steps: 1,
        eos_threshold: -4.0,
        quantized: false,
        voice_cache_capacity: 64,
        prewarm_voices: "alba".to_string(),
        warmup: true,
        omp_threads: None,
        mkl_threads: None,
        ui: UiMode::WasmExperimental,
    };

    crate::commands::serve::run(serve_args).await
}
