//! HTTP API Server
//!
//! Axum-based server providing TTS generation endpoints.

use anyhow::Result;
use pocket_tts::TTSModel;

use crate::commands::serve::{ServeArgs, UiMode, print_endpoints};
use crate::voice::{resolve_voice_for_language, voice_cache_key};

pub mod handlers;
pub mod routes;
pub mod state;

pub async fn start_server(args: ServeArgs) -> Result<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt::try_init();

    let wasm_pkg_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("pocket-tts")
        .join("pkg");

    if matches!(args.ui, UiMode::WasmExperimental) {
        let wasm_js = wasm_pkg_dir.join("pocket_tts.js");
        let wasm_bin = wasm_pkg_dir.join("pocket_tts_bg.wasm");
        if !wasm_js.is_file() || !wasm_bin.is_file() {
            anyhow::bail!(
                "WASM UI mode requires built WASM assets at {:?}. Run scripts/build-wasm.ps1 (Windows) or scripts/build-wasm.sh (Unix).",
                wasm_pkg_dir
            );
        }
    }

    if let Some(omp_threads) = args.omp_threads {
        // SAFETY: environment is configured at startup before serving requests.
        unsafe {
            std::env::set_var("OMP_NUM_THREADS", omp_threads.to_string());
        }
        println!("  Set OMP_NUM_THREADS={omp_threads}");
    }
    if let Some(mkl_threads) = args.mkl_threads {
        // SAFETY: environment is configured at startup before serving requests.
        unsafe {
            std::env::set_var("MKL_NUM_THREADS", mkl_threads.to_string());
        }
        println!("  Set MKL_NUM_THREADS={mkl_threads}");
    }

    let (model_stem, language_opt) = args.resolved_stem();
    let language_owned: Option<String> = language_opt.map(str::to_string);
    let language_ref: Option<&str> = language_owned.as_deref();
    let default_voice = args.resolved_default_voice();

    // Load model with configured parameters
    let model = if args.quantized {
        #[cfg(feature = "quantized")]
        {
            TTSModel::load_quantized_with_params(
                &model_stem,
                args.temperature,
                args.lsd_decode_steps,
                args.eos_threshold,
            )?
        }
        #[cfg(not(feature = "quantized"))]
        {
            anyhow::bail!("Quantization feature not enabled. Rebuild with --features quantized");
        }
    } else {
        TTSModel::load_with_params(
            &model_stem,
            args.temperature,
            args.lsd_decode_steps,
            args.eos_threshold,
        )?
    };

    println!("  ✓ Model loaded (sample rate: {}Hz)", model.sample_rate);

    // Pre-load default voice
    println!("  Loading default voice: {}...", default_voice);
    let default_voice_state = resolve_voice_for_language(&model, Some(&default_voice), language_ref)?;
    println!("  ✓ Default voice ready");

    let state = state::AppState::new(
        model,
        default_voice_state,
        args.voice_cache_capacity,
        args.ui,
        wasm_pkg_dir,
        language_owned.clone(),
    );
    {
        let mut cache = state
            .voice_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("voice cache lock poisoned"))?;
        cache.put(
            voice_cache_key(&default_voice),
            state.default_voice_state.clone(),
        );
    }

    for voice in args
        .prewarm_voices
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let key = voice_cache_key(voice);
        let already_cached = {
            let mut cache = state
                .voice_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("voice cache lock poisoned"))?;
            cache.get(&key).is_some()
        };
        if already_cached {
            continue;
        }

        println!("  Prewarming voice: {voice}...");
        match resolve_voice_for_language(&state.model, Some(voice), language_ref) {
            Ok(vs) => {
                let mut cache = state
                    .voice_cache
                    .lock()
                    .map_err(|_| anyhow::anyhow!("voice cache lock poisoned"))?;
                cache.put(key, std::sync::Arc::new(vs));
                println!("  - Voice prewarmed: {voice}");
            }
            Err(e) => {
                println!("  !! Failed to prewarm voice '{voice}': {e}");
            }
        }
    }

    if args.warmup {
        println!("  Running startup warmup...");
        let mut warmup_iter = state
            .model
            .generate_stream_long("warmup", &state.default_voice_state);
        if let Some(frame_res) = warmup_iter.next() {
            frame_res?;
        }
        println!("  - Warmup complete");
    }

    let app = routes::create_router(state);

    let addr = format!("{}:{}", args.host, args.port);

    print_endpoints(&args.host, args.port, args.ui);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    println!("  👋 Server stopped gracefully");

    Ok(())
}

/// Wait for Ctrl+C or SIGTERM signal
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            println!("\n  ⚠️  Received Ctrl+C, shutting down...");
        },
        _ = terminate => {
            println!("\n  ⚠️  Received SIGTERM, shutting down...");
        },
    }
}
