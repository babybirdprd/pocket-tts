//! Main TTSModel struct - orchestrates the TTS pipeline
//!
//! This is the high-level API for text-to-speech generation,
//! matching Python's `pocket_tts/models/tts_model.py`.

use crate::ModelState;
use crate::conditioners::text::LUTConditioner;
use crate::config::{Config, defaults, load_config};
use crate::models::flow_lm::FlowLMModel;
use crate::models::mimi::MimiModel;
use crate::models::seanet::{SEANetDecoder, SEANetEncoder};
use crate::models::transformer::{ProjectedTransformer, StreamingTransformer};
use crate::modules::mlp::SimpleMLPAdaLN;
use crate::voice_state::{
    ATTN_K_BUF_KEY, ATTN_LEN_KEY, ATTN_POS_KEY, ATTN_V_BUF_KEY, increment_steps, init_states,
};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

/// Per-call generation options (upstream's `max_tokens` and
/// `frames_after_eos` keyword arguments).
#[derive(Debug, Clone, Copy)]
pub struct GenerateOptions {
    /// Per-chunk token budget for the text splitter.
    pub max_tokens: usize,
    /// Explicit frames to generate after EOS; `None` uses the model
    /// recommendation from its config, then the length heuristic.
    pub frames_after_eos: Option<usize>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: defaults::MAX_TOKEN_PER_CHUNK,
            frames_after_eos: None,
        }
    }
}

/// Main TTS model that orchestrates the entire pipeline
#[derive(Clone)]
pub struct TTSModel {
    /// Flow language model for latent generation
    pub flow_lm: FlowLMModel,
    /// Mimi neural audio codec
    pub mimi: MimiModel,
    /// Text conditioner (tokenizer + embeddings)
    pub conditioner: LUTConditioner,
    /// Speaker projection weight for voice cloning
    pub speaker_proj_weight: Tensor,
    /// Generation temperature
    pub temp: f32,
    /// Number of LSD decode steps
    pub lsd_decode_steps: usize,
    /// End-of-sequence threshold
    pub eos_threshold: f32,
    pub noise_clamp: Option<f32>,
    /// Optional override for voice-conditioning Mimi chunk size (in frames).
    /// If `None`, an adaptive heuristic is used.
    pub voice_prompt_chunk_frames: Option<usize>,
    /// Sample rate
    pub sample_rate: usize,
    /// Mimi frame rate (latent frames per second, 12.5 for all checkpoints)
    pub frame_rate: f64,
    /// Model dimension
    pub dim: usize,
    /// Latent dimension
    pub ldim: usize,
    /// Device
    pub device: Device,
    /// Prepend 8 spaces to very short inputs (English checkpoints only).
    pub pad_with_spaces_for_short_inputs: bool,
    /// Replace `;` with `,` before tokenisation (multilingual checkpoints).
    pub remove_semicolons: bool,
    /// Model-recommended frames after EOS; overrides the heuristic when set.
    pub model_recommended_frames_after_eos: Option<usize>,
    /// Config stem the model was loaded from (e.g. "french_24l"), like
    /// Python's `origin`. Used to scope predefined voices per language;
    /// `None` when loaded from raw bytes.
    pub origin: Option<String>,
    /// False when the without-voice-cloning checkpoint was loaded (its Mimi
    /// encoder weights are shipped zeroed): audio-prompt cloning would
    /// silently produce silence, so it is refused, like Python's
    /// `has_voice_cloning`.
    pub has_voice_cloning: bool,
}

impl TTSModel {
    /// Load a pre-trained TTS model from HuggingFace
    ///
    /// # Arguments
    /// * `variant` - Model variant (e.g., "b6369a24")
    ///
    /// # Returns
    /// Fully initialized TTSModel ready for generation
    pub fn load(variant: &str) -> Result<Self> {
        Self::load_with_params(
            variant,
            None,
            defaults::LSD_DECODE_STEPS,
            defaults::EOS_THRESHOLD,
        )
    }

    /// Load with custom generation parameters.
    ///
    /// `temp: None` uses the model's `default_temperature` from its config
    /// (upstream #223: per-checkpoint recommended temperature).
    pub fn load_with_params(
        variant: &str,
        temp: Option<f32>,
        lsd_decode_steps: usize,
        eos_threshold: f32,
    ) -> Result<Self> {
        Self::load_with_params_device(
            variant,
            temp,
            lsd_decode_steps,
            eos_threshold,
            None,
            &Device::Cpu,
        )
    }

    /// Load with custom generation parameters and specific device
    pub fn load_with_params_device(
        variant: &str,
        temp: Option<f32>,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        noise_clamp: Option<f32>,
        device: &Device,
    ) -> Result<Self> {
        // Find config file - look relative to the Rust crate, then fall back to Python location
        let config_path = find_config_path(variant)?;
        let config = load_config(&config_path)?;

        let mut model = Self::from_config(
            config,
            temp,
            lsd_decode_steps,
            eos_threshold,
            noise_clamp,
            device,
        )?;
        // Like Python's origin: the config stem ("french_24l" both for the
        // bundled variant name and a local path ending in french_24l.yaml).
        model.origin = config_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .or_else(|| Some(variant.to_string()));
        Ok(model)
    }

    /// Load model with quantized weights for reduced memory footprint
    ///
    /// This applies simulated int8 quantization to applicable layers,
    /// reducing memory usage while maintaining acceptable quality.
    ///
    /// # Arguments
    /// * `variant` - Model variant (e.g., "b6369a24")
    ///
    /// # Returns
    /// TTSModel with quantized weights
    ///
    /// # Note
    /// Quantization uses 256 discrete levels (int8-equivalent).
    /// Some layers (embeddings, output projections) are kept in full precision.
    pub fn load_quantized(variant: &str) -> Result<Self> {
        Self::load_quantized_with_params(
            variant,
            None,
            defaults::LSD_DECODE_STEPS,
            defaults::EOS_THRESHOLD,
        )
    }

    /// Load quantized model with custom generation parameters
    pub fn load_quantized_with_params(
        variant: &str,
        temp: Option<f32>,
        lsd_decode_steps: usize,
        eos_threshold: f32,
    ) -> Result<Self> {
        Self::load_quantized_with_params_device(
            variant,
            temp,
            lsd_decode_steps,
            eos_threshold,
            None,
            &Device::Cpu,
        )
    }

    /// Load quantized model with custom generation parameters and specific device
    pub fn load_quantized_with_params_device(
        variant: &str,
        temp: Option<f32>,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        noise_clamp: Option<f32>,
        device: &Device,
    ) -> Result<Self> {
        let mut model = Self::load_with_params_device(
            variant,
            temp,
            lsd_decode_steps,
            eos_threshold,
            noise_clamp,
            device,
        )?;
        model.apply_dynamic_int8(crate::quantize::RECOMMENDED_CONFIG)?;
        Ok(model)
    }

    /// Upstream `apply_dynamic_int8`: quantize the FlowLM transformer's
    /// requested layer groups to int8. Flow net and Mimi stay float32.
    pub fn apply_dynamic_int8(&mut self, groups: &[crate::quantize::QuantizeGroup]) -> Result<()> {
        if groups.contains(&crate::quantize::QuantizeGroup::FlowNet) {
            anyhow::bail!(
                "flow_net quantization is not ported (upstream's recommended config excludes it)"
            );
        }
        self.flow_lm.transformer.quantize_int8(groups)
    }

    /// Check if this model was loaded with quantization
    pub fn is_quantized(&self) -> bool {
        self.flow_lm.transformer.is_quantized()
    }

    /// Create model from configuration
    fn from_config(
        config: Config,
        temp: Option<f32>,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        noise_clamp: Option<f32>,
        device: &Device,
    ) -> Result<Self> {
        let dtype = DType::F32;

        // Download weights
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Prefer the voice-cloning weights; fall back to the
            // without-voice-cloning checkpoint (not HF-gated) so French works
            // without an HF token. Predefined-voice prompts still work there,
            // but its Mimi encoder weights are shipped zeroed, so cloning is
            // disabled (has_voice_cloning, like Python).
            let (weights_file, has_voice_cloning) = match config.weights_path.as_deref() {
                Some(p) => match crate::weights::download_if_necessary(p) {
                    Ok(f) => (f, true),
                    Err(primary_err) => {
                        match config.weights_path_without_voice_cloning.as_deref() {
                            Some(fallback) => {
                                tracing::warn!(
                                    error = %primary_err,
                                    "voice-cloning weights unavailable, falling back to without-voice-cloning checkpoint"
                                );
                                (crate::weights::download_if_necessary(fallback)?, false)
                            }
                            None => return Err(primary_err),
                        }
                    }
                },
                None => {
                    let fallback = config
                        .weights_path_without_voice_cloning
                        .as_deref()
                        .ok_or_else(|| anyhow::anyhow!("weights_path not specified in config"))?;
                    (crate::weights::download_if_necessary(fallback)?, false)
                }
            };

            // Load safetensors with VarBuilder
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], dtype, device)? };

            // Download tokenizer
            let tokenizer_path =
                crate::weights::download_if_necessary(&config.flow_lm.lookup_table.tokenizer_path)?;

            // Build conditioner
            let conditioner = LUTConditioner::new(
                config.flow_lm.lookup_table.n_bins,
                &tokenizer_path,
                config.flow_lm.lookup_table.dim,
                config.flow_lm.transformer.d_model,
                vb.pp("flow_lm.conditioner"),
            )?;

            let mut model = Self::from_config_and_vb(
                config,
                temp,
                lsd_decode_steps,
                eos_threshold,
                noise_clamp,
                conditioner,
                vb,
            )?;
            model.has_voice_cloning = has_voice_cloning;
            Ok(model)
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ = (config, temp, lsd_decode_steps, eos_threshold, device, dtype);
            anyhow::bail!(
                "WASM requires from_bytes or providing a pre-built VarBuilder. Use load_from_bytes instead."
            );
        }
    }

    /// Load model from byte slices (useful for WASM)
    pub fn load_from_bytes(
        config_yaml: &[u8],
        weights_bytes: &[u8],
        tokenizer_bytes: &[u8],
    ) -> Result<Self> {
        let config: Config = serde_yaml::from_slice(config_yaml)?;
        let device = Device::Cpu;
        let dtype = DType::F32;

        let tensors = candle_core::safetensors::load_buffer(weights_bytes, &device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);

        // On WASM, LUTConditioner::new needs a path, but we've updated it to
        // eventually support bytes. For now, we'll need to adapt it.
        // Actually, my recent change to conditioners/text.rs still uses Tokenizer::from_file on WASM.
        // I should probably fix that to support bytes too.

        // For now, let's keep it simple and assume we have a path for the tokenizer or a way to load it.
        // This is a placeholder for real WASM loading.

        let conditioner = LUTConditioner::new_from_bytes(
            config.flow_lm.lookup_table.n_bins,
            tokenizer_bytes,
            config.flow_lm.lookup_table.dim,
            config.flow_lm.transformer.d_model,
            vb.pp("flow_lm.conditioner"),
        )?;

        Self::from_config_and_vb(
            config,
            None,
            defaults::LSD_DECODE_STEPS,
            defaults::EOS_THRESHOLD,
            None,
            conditioner,
            vb,
        )
    }

    /// Internal helper to build model from config and VarBuilder
    fn from_config_and_vb(
        config: Config,
        temp: Option<f32>,
        lsd_decode_steps: usize,
        eos_threshold: f32,
        noise_clamp: Option<f32>,
        conditioner: LUTConditioner,
        vb: VarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();

        // Build FlowLM components
        let dim = config.flow_lm.transformer.d_model;
        let ldim = config.mimi.quantizer.dimension;
        let hidden_dim = dim * config.flow_lm.transformer.hidden_scale;

        // SimpleMLPAdaLN::new(in_channels, model_channels, out_channels, cond_channels, num_res_blocks, num_time_conds, max_period, vb)
        let flow_net = SimpleMLPAdaLN::new(
            ldim,                      // in_channels (input is latent dim)
            config.flow_lm.flow.dim,   // model_channels
            ldim,                      // out_channels (output is also latent dim)
            dim,                       // cond_channels (conditioning from transformer)
            config.flow_lm.flow.depth, // num_res_blocks
            2,                         // num_time_conds (s and t)
            config.flow_lm.transformer.max_period as f32,
            vb.pp("flow_lm.flow_net"),
        )?;

        // StreamingTransformer::new(d_model, num_heads, num_layers, layer_scale, dim_feedforward, context, max_period, kind, name, vb)
        let transformer = StreamingTransformer::new(
            dim,
            config.flow_lm.transformer.num_heads,
            config.flow_lm.transformer.num_layers,
            None,       // layer_scale
            hidden_dim, // dim_feedforward
            None,       // context (causal)
            config.flow_lm.transformer.max_period as f32,
            "kv",
            "flow_lm.transformer",
            vb.pp("flow_lm.transformer"),
        )?;

        let mut flow_lm = FlowLMModel::new(
            flow_net,
            transformer,
            ldim,
            dim,
            config.flow_lm.insert_bos_before_voice,
            vb.pp("flow_lm"),
        )?;
        flow_lm.noise_clamp = noise_clamp;

        // Build Mimi components
        let seanet_cfg = &config.mimi.seanet;
        let encoder = SEANetEncoder::new(
            seanet_cfg.channels,
            seanet_cfg.dimension,
            seanet_cfg.n_filters,
            seanet_cfg.n_residual_layers,
            &seanet_cfg.ratios,
            seanet_cfg.kernel_size,
            seanet_cfg.residual_kernel_size,
            seanet_cfg.last_kernel_size,
            seanet_cfg.dilation_base,
            &seanet_cfg.pad_mode,
            seanet_cfg.compress,
            "mimi.encoder",
            vb.pp("mimi.encoder"),
        )?;

        let decoder = SEANetDecoder::new(
            seanet_cfg.channels,
            seanet_cfg.dimension,
            seanet_cfg.n_filters,
            seanet_cfg.n_residual_layers,
            &seanet_cfg.ratios,
            seanet_cfg.kernel_size,
            seanet_cfg.residual_kernel_size,
            seanet_cfg.last_kernel_size,
            seanet_cfg.dilation_base,
            &seanet_cfg.pad_mode,
            seanet_cfg.compress,
            "mimi.decoder",
            vb.pp("mimi.decoder"),
        )?;

        let mimi_tr_cfg = &config.mimi.transformer;
        // ProjectedTransformer::new(input_dimension, output_dimensions, d_model, num_heads, num_layers, layer_scale, context, max_period, dim_feedforward, name, vb)
        let encoder_transformer = ProjectedTransformer::new(
            mimi_tr_cfg.input_dimension,
            mimi_tr_cfg.output_dimensions.clone(),
            mimi_tr_cfg.d_model,
            mimi_tr_cfg.num_heads,
            mimi_tr_cfg.num_layers,
            mimi_tr_cfg.layer_scale as f32,
            mimi_tr_cfg.context,
            mimi_tr_cfg.max_period as f32,
            mimi_tr_cfg.dim_feedforward,
            "mimi.encoder_transformer",
            vb.pp("mimi.encoder_transformer"),
        )?;

        let decoder_transformer = ProjectedTransformer::new(
            mimi_tr_cfg.input_dimension,
            mimi_tr_cfg.output_dimensions.clone(),
            mimi_tr_cfg.d_model,
            mimi_tr_cfg.num_heads,
            mimi_tr_cfg.num_layers,
            mimi_tr_cfg.layer_scale as f32,
            mimi_tr_cfg.context,
            mimi_tr_cfg.max_period as f32,
            mimi_tr_cfg.dim_feedforward,
            "mimi.decoder_transformer",
            vb.pp("mimi.decoder_transformer"),
        )?;

        // Calculate encoder frame rate from SEANet ratios
        let hop_length: usize = seanet_cfg.ratios.iter().product();
        let encoder_frame_rate = config.mimi.sample_rate as f64 / hop_length as f64;

        let mimi = MimiModel::new(
            encoder,
            decoder,
            encoder_transformer,
            decoder_transformer,
            config.mimi.frame_rate,
            encoder_frame_rate,
            config.mimi.sample_rate,
            config.mimi.channels,
            config.mimi.quantizer.dimension,
            config.mimi.quantizer.output_dimension,
            config.mimi.seanet.dimension,
            config.mimi.inner_dim,
            config.mimi.outer_dim,
            "mimi",
            vb.pp("mimi"),
        )?;

        // Speaker projection maps the encoder latent (inner_dim, or the SEANet
        // dimension for the pre-multilingual checkpoints) to the FlowLM model
        // dimension. Tolerate its absence: the without-voice-cloning
        // checkpoints omit it, and the prompt-file path never uses it.
        let speaker_proj_dim = config
            .mimi
            .inner_dim
            .unwrap_or(config.mimi.seanet.dimension);
        let speaker_proj_weight = match vb
            .get((dim, speaker_proj_dim), "flow_lm.speaker_proj_weight")
        {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!(error = %e, "speaker_proj_weight missing; voice cloning from WAV disabled");
                Tensor::zeros((dim, speaker_proj_dim), DType::F32, &device)?
            }
        };

        Ok(Self {
            flow_lm,
            mimi,
            conditioner,
            speaker_proj_weight,
            temp: temp.unwrap_or(config.default_temperature),
            lsd_decode_steps,
            eos_threshold,
            noise_clamp,
            voice_prompt_chunk_frames: None,
            sample_rate: config.mimi.sample_rate,
            frame_rate: config.mimi.frame_rate,
            dim,
            ldim,
            device,
            pad_with_spaces_for_short_inputs: config.pad_with_spaces_for_short_inputs,
            remove_semicolons: config.remove_semicolons,
            model_recommended_frames_after_eos: config.model_recommended_frames_after_eos,
            origin: None,
            has_voice_cloning: true,
        })
    }

    /// Create voice state from audio prompt bytes for voice cloning
    pub fn get_voice_state_from_bytes(&self, bytes: &[u8]) -> Result<ModelState> {
        let (audio, sample_rate) = crate::audio::read_wav_from_bytes(bytes)?;

        // Resample to model sample rate if needed
        let audio = if sample_rate != self.sample_rate as u32 {
            crate::audio::resample(&audio, sample_rate, self.sample_rate as u32)?
        } else {
            audio
        };

        // Add batch dimension: [C, T] -> [B, C, T]
        let audio = audio.unsqueeze(0)?;

        self.get_voice_state_from_tensor(&audio)
    }

    /// Create voice state from audio prompt for voice cloning
    ///
    /// Encodes the audio through Mimi and projects to flow model space.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_voice_state<P: AsRef<std::path::Path>>(&self, audio_path: P) -> Result<ModelState> {
        let (audio, sample_rate) = crate::audio::read_audio(audio_path)?;

        // Resample to model sample rate if needed
        let audio = if sample_rate != self.sample_rate as u32 {
            crate::audio::resample(&audio, sample_rate, self.sample_rate as u32)?
        } else {
            audio
        };

        // Add batch dimension: [C, T] -> [B, C, T]
        let audio = audio.unsqueeze(0)?;

        self.get_voice_state_from_tensor(&audio)
    }

    /// Create voice state from a pre-calculated latent prompt file (.safetensors)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_voice_state_from_prompt_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<ModelState> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        let prompt = tensors
            .get("audio_prompt")
            .ok_or_else(|| anyhow::anyhow!("'audio_prompt' not found in safetensors file"))?;

        self.get_voice_state_from_prompt_tensor(prompt)
    }

    /// Create voice state from a Kyutai predefined-voice embedding
    /// (`.safetensors`), e.g. `kyutai/pocket-tts-without-voice-cloning`
    /// `.../languages/<lang>/embeddings/<name>.safetensors`.
    ///
    /// These files are a per-FlowLM-layer KV-cache exported by upstream:
    /// keys `transformer.layers.{i}.self_attn/cache` plus an offset stored
    /// either as `.../offset` (post-#85, a `long[1]`) or `.../current_end`
    /// (pre-#85, where the length is encoded in the tensor's first dim).
    /// The cache is `[2, B, T, H, D]`; this bridges it into the crate's
    /// `[B, H, T, D]` per-layer state so generation needs neither the Mimi
    /// encoder nor `speaker_proj_weight` (works token-free with the
    /// without-voice-cloning checkpoint).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_voice_state_from_kyutai_embedding<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<ModelState> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        let mut flow_state = init_states(1, 1000);

        let mut layer = 0usize;
        loop {
            let cache_name = format!("transformer.layers.{layer}.self_attn/cache");
            let Some(cache) = tensors.get(&cache_name) else {
                break;
            };

            let offset_name = format!("transformer.layers.{layer}.self_attn/offset");
            let current_end_name = format!("transformer.layers.{layer}.self_attn/current_end");
            let prompt_len: usize = if let Some(off) = tensors.get(&offset_name) {
                // post-#85: a long[1] holding the absolute offset.
                off.flatten_all()?
                    .to_dtype(DType::I64)?
                    .to_vec1::<i64>()?
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("empty offset for layer {layer}"))?
                    .max(0) as usize
            } else if let Some(cur) = tensors.get(&current_end_name) {
                // pre-#85: the length is encoded in the first dimension.
                cur.dim(0)?
            } else {
                anyhow::bail!(
                    "layer {layer}: neither '{offset_name}' nor '{current_end_name}' present"
                );
            };

            let (k_buf, v_buf) = unpack_kv_cache(cache)?;

            let mut layer_state = HashMap::new();
            layer_state.insert(ATTN_K_BUF_KEY.to_string(), k_buf);
            layer_state.insert(ATTN_V_BUF_KEY.to_string(), v_buf);
            layer_state.insert(
                ATTN_POS_KEY.to_string(),
                Tensor::new(prompt_len as u32, &self.device)?,
            );
            layer_state.insert(
                ATTN_LEN_KEY.to_string(),
                Tensor::new(prompt_len as i64, &self.device)?,
            );

            flow_state.insert(
                format!("flow_lm.transformer.layers.{layer}.self_attn"),
                layer_state,
            );
            layer += 1;
        }

        if layer == 0 {
            anyhow::bail!(
                "no 'transformer.layers.*.self_attn/cache' tensors found; \
                 not a Kyutai predefined-voice embedding"
            );
        }

        Ok(flow_state)
    }

    /// Create voice state from pre-calculated latent prompt bytes (.safetensors)
    pub fn get_voice_state_from_prompt_bytes(&self, bytes: &[u8]) -> Result<ModelState> {
        let tensors = candle_core::safetensors::load_buffer(bytes, &self.device)?;
        let prompt = tensors
            .get("audio_prompt")
            .ok_or_else(|| anyhow::anyhow!("'audio_prompt' not found in safetensors bytes"))?;

        self.get_voice_state_from_prompt_tensor(prompt)
    }

    /// Create voice state from a pre-calculated latent prompt tensor
    pub fn get_voice_state_from_prompt_tensor(&self, prompt: &Tensor) -> Result<ModelState> {
        // Ensure prompt tensor is on the same device as the model (fixes Metal device mismatch)
        let prompt = if prompt.device().same_device(&self.device) {
            prompt.clone()
        } else {
            prompt.to_device(&self.device)?
        };

        let mut flow_state = init_states(1, 1000);
        self.run_flow_lm_prompt(&prompt, &mut flow_state)?;
        Ok(flow_state)
    }

    /// Create voice state from audio tensor
    /// Encode a reference-audio tensor into FlowLM conditioning
    /// (`[B, T, dim]`). Shared by [`Self::get_voice_state_from_tensor`].
    pub fn get_conditioning(&self, audio: &Tensor) -> Result<Tensor> {
        // Upstream VOICE_CLONING_UNSUPPORTED: without the gated weights the
        // encoder is zeroed and "cloning" would silently produce silence.
        if !self.has_voice_cloning {
            anyhow::bail!(
                "Voice cloning is unavailable: the without-voice-cloning checkpoint ships a \
                 zeroed Mimi encoder, so an audio prompt would produce silence. Use a predefined \
                 voice (e.g. alba, marius, estelle, giovanni, lola, juergen, rafael) or a \
                 .safetensors embedding instead. For voice cloning, accept the terms at \
                 https://huggingface.co/kyutai/pocket-tts and authenticate (HF_TOKEN)."
            );
        }
        let mut model_state = init_states(1, 1000);

        // Ensure audio tensor is on the same device as the model (fixes Metal device mismatch)
        let audio = if audio.device().same_device(&self.device) {
            audio.clone()
        } else {
            audio.to_device(&self.device)?
        };

        // Pad audio to a multiple of frame size for streaming conv stride alignment
        let frame_size = self.mimi.frame_size();
        let (b, c, t) = audio.dims3()?;
        let pad_len = if t % frame_size != 0 {
            frame_size - (t % frame_size)
        } else {
            0
        };
        let audio = if pad_len > 0 {
            // Create padding on the same device as audio (which is now on self.device)
            let pad = Tensor::zeros((b, c, pad_len), audio.dtype(), &self.device)?;
            Tensor::cat(&[&audio, &pad], 2)?
        } else {
            audio
        };

        // Encode audio through Mimi in chunks to avoid OOM in SEANet Conv1d layers.
        // Chunk size is adapted to prompt length unless explicitly overridden.
        let mut encoded_chunks = Vec::new();
        let (_b, _c, total_samples) = audio.dims3()?;
        let chunk_frames = self.adaptive_voice_prompt_chunk_frames(total_samples, frame_size);
        let chunk_size = frame_size * chunk_frames;

        for start in (0..total_samples).step_by(chunk_size) {
            let end = std::cmp::min(start + chunk_size, total_samples);
            let chunk = audio.narrow(2, start, end - start)?;
            let code = self.mimi.encode_to_latent(&chunk, &mut model_state, 0)?;
            encoded_chunks.push(code);
        }
        let encoded = Tensor::cat(&encoded_chunks, 2)?;

        // Transpose from [B, D, T] to [B, T, D]
        let latents = encoded.transpose(1, 2)?.to_dtype(DType::F32)?;

        // Project to flow model space: [B, T, ldim] @ [dim, ldim].T -> [B, T, dim]
        // Candle needs 2D @ 2D for matmul, so reshape
        let (b, t, d) = latents.dims3()?;
        let latents_2d = latents.reshape((b * t, d))?;
        let conditioning_2d = latents_2d.matmul(&self.speaker_proj_weight.t()?)?;
        let conditioning = conditioning_2d.reshape((b, t, self.dim))?;
        Ok(conditioning)
    }

    /// Create voice state from a reference-audio tensor (voice cloning).
    pub fn get_voice_state_from_tensor(&self, audio: &Tensor) -> Result<ModelState> {
        let conditioning = self.get_conditioning(audio)?;

        // Run flow_lm with audio conditioning to update state
        let mut flow_state = init_states(1, 1000);
        self.run_flow_lm_prompt(&conditioning, &mut flow_state)?;

        Ok(flow_state)
    }

    fn adaptive_voice_prompt_chunk_frames(&self, total_samples: usize, frame_size: usize) -> usize {
        if let Some(override_frames) = self.voice_prompt_chunk_frames {
            return override_frames.max(1);
        }

        let total_frames = total_samples.div_ceil(frame_size);
        if total_frames <= 120 {
            total_frames.max(1)
        } else if total_frames <= 600 {
            120
        } else if total_frames <= 1800 {
            180
        } else {
            240
        }
    }

    /// Run flow LM with audio conditioning (used during prompting)
    fn run_flow_lm_prompt(&self, conditioning: &Tensor, state: &mut ModelState) -> Result<()> {
        // #155: the multilingual checkpoints were trained with a learnt BOS
        // embedding prepended to the voice conditioning. Getting this wrong
        // (or omitting it) produces garbled audio with the French model.
        let conditioning = match &self.flow_lm.bos_before_voice {
            Some(bos) => Tensor::cat(&[bos, conditioning], 1)?,
            None => conditioning.clone(),
        };

        // Empty text tokens and backbone input
        let empty_text = Tensor::zeros((1, 0), DType::I64, &self.device)?;
        let text_embeddings = self.conditioner.forward(&empty_text)?;

        // Concatenate text embeddings and audio conditioning
        // Match Python/reference order: audio conditioning comes before text embeddings.
        let input = Tensor::cat(&[&conditioning, &text_embeddings], 1)?;

        // Run through transformer (no generation, just prompting)
        // With custom SDPA, this is now memory efficient
        let _ = self.flow_lm.transformer.forward(&input, state, 0)?;

        // Increment FlowLM state after prompting (critical for RoPE positioning).
        // Includes the prepended BOS frame when present.
        let increment_by = conditioning.dims()[1];
        increment_steps(state, "offset", increment_by);

        Ok(())
    }

    /// Split text into optimal chunks for generation, matching Python's logic exactly.
    /// Uses actual tokenization to ensure chunks never exceed the default token
    /// budget (50). This prevents O(N²) attention complexity for long texts.
    pub fn split_into_best_sentences(&self, text: &str) -> Vec<String> {
        self.split_into_best_sentences_with(text, defaults::MAX_TOKEN_PER_CHUNK)
    }

    /// Split with an explicit per-chunk token budget (upstream `max_tokens`).
    ///
    /// Port of upstream's token-boundary splitter: sentences end at the
    /// tokens of ".!...?", oversized sentences are sub-split at the tokens
    /// of ",;:" (#143), and segments are greedily packed under the budget.
    /// One deviation kept from the port: a segment that still exceeds the
    /// budget is word-batched instead of merely warned about, so a chunk
    /// can never overflow the token budget.
    pub fn split_into_best_sentences_with(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let prepared_text = prepare_text_prompt(
            text,
            self.pad_with_spaces_for_short_inputs,
            self.remove_semicolons,
        );
        let prepared_text = prepared_text.trim().to_string();

        let Ok(tokens) = self.conditioner.encode_ids(&prepared_text) else {
            return vec![prepared_text];
        };

        // `_, *end_of_sentence_tokens = tokenizer(".!...?")`: drop the
        // leading metaspace token, keep the punctuation token ids.
        let sentence_marks = match self.conditioner.encode_ids(".!...?") {
            Ok(ids) if ids.len() > 1 => ids[1..].to_vec(),
            _ => return vec![prepared_text],
        };

        let boundaries = find_boundary_indices(&tokens, &sentence_marks);
        let mut segments = match self.segments_from_boundaries(&tokens, &boundaries) {
            Ok(s) => s,
            Err(_) => return vec![prepared_text],
        };

        // Sub-split oversized sentences on commas, semicolons and colons to
        // prevent skipped words (#143).
        let fallback_marks = self
            .conditioner
            .encode_ids(",;:")
            .ok()
            .filter(|ids| ids.len() > 1)
            .map(|ids| ids[1..].to_vec());
        if let Some(fallback_marks) = fallback_marks {
            let mut refined = Vec::with_capacity(segments.len());
            for (nb_tokens, segment_text) in segments {
                if nb_tokens <= max_tokens {
                    refined.push((nb_tokens, segment_text));
                    continue;
                }
                let sub = self
                    .conditioner
                    .encode_ids(segment_text.trim())
                    .ok()
                    .and_then(|sub_tokens| {
                        let sub_boundaries =
                            find_boundary_indices(&sub_tokens, &fallback_marks);
                        self.segments_from_boundaries(&sub_tokens, &sub_boundaries)
                            .ok()
                    });
                match sub {
                    Some(sub_segments) if sub_segments.len() > 1 => refined.extend(sub_segments),
                    _ => refined.push((nb_tokens, segment_text)),
                }
            }
            segments = refined;
        }

        // Greedy packing under the budget.
        let mut chunks: Vec<String> = Vec::new();
        let mut current_chunk = String::new();
        let mut current_token_count = 0usize;
        for (nb_tokens, sentence) in segments {
            if current_chunk.is_empty() {
                current_chunk = sentence;
                current_token_count = nb_tokens;
            } else if current_token_count + nb_tokens > max_tokens {
                chunks.push(current_chunk);
                current_chunk = sentence;
                current_token_count = nb_tokens;
            } else {
                current_chunk.push(' ');
                current_chunk.push_str(&sentence);
                current_token_count += nb_tokens;
            }
        }
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        // Deviation from upstream (which only logs a warning): a chunk still
        // over budget is word-batched so a chunk can never overflow.
        let mut bounded = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let chunk = chunk.trim().to_string();
            if chunk.is_empty() {
                continue;
            }
            let count = self.conditioner.count_tokens(&chunk).unwrap_or(0);
            if count > max_tokens {
                tracing::warn!(
                    tokens = count,
                    max = max_tokens,
                    "chunk exceeds the token budget; word-batching it"
                );
                bounded.extend(self.word_batch_chunks(&chunk, max_tokens));
            } else {
                bounded.push(chunk);
            }
        }
        if bounded.is_empty() {
            return vec![prepared_text];
        }
        bounded
    }

    /// Decode token segments between boundary indices into
    /// (token_count, text) pairs, mirroring upstream `_segments_from_boundaries`.
    fn segments_from_boundaries(
        &self,
        tokens: &[u32],
        boundaries: &[usize],
    ) -> Result<Vec<(usize, String)>> {
        let mut segments = Vec::with_capacity(boundaries.len().saturating_sub(1));
        for pair in boundaries.windows(2) {
            let (start, end) = (pair[0], pair[1]);
            if end <= start {
                continue;
            }
            let text = self.conditioner.decode_ids(&tokens[start..end])?;
            let text = text.trim().to_string();
            if !text.is_empty() {
                segments.push((end - start, text));
            }
        }
        Ok(segments)
    }

    /// Last-resort split for a clause/sentence with no usable comma:
    /// fixed-size word batches kept under `max` tokens (a batch still over
    /// budget is halved once). Mirrors the pre-#143 fallback.
    fn word_batch_chunks(&self, sentence: &str, max: usize) -> Vec<String> {
        const WORDS_PER_BATCH: usize = 35; // ~45 tokens, safe margin under 50
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut out = Vec::new();
        for word_batch in words.chunks(WORDS_PER_BATCH) {
            let chunk_str = word_batch.join(" ");
            let actual = self.conditioner.count_tokens(&chunk_str).unwrap_or(max);
            if actual <= max {
                out.push(chunk_str);
            } else {
                let mid = word_batch.len() / 2;
                out.push(word_batch[..mid].join(" "));
                out.push(word_batch[mid..].join(" "));
            }
        }
        out
    }

    /// Generate audio from text with voice state
    pub fn generate(&self, text: &str, voice_state: &ModelState) -> Result<Tensor> {
        let mut audio_chunks = Vec::new();

        for chunk in self.generate_stream(text, voice_state) {
            audio_chunks.push(chunk?);
        }

        // Concatenate all audio chunks
        if audio_chunks.is_empty() {
            anyhow::bail!("No audio generated");
        }
        let audio = Tensor::cat(&audio_chunks, 2)?;
        // Remove batch dimension
        let audio = audio.squeeze(0)?;

        Ok(audio)
    }

    // =========================================================================
    // EXPERIMENTAL: Parallel FlowLM + Mimi decoding
    // =========================================================================
    //
    // This method was an experiment to decode Mimi audio in a separate thread
    // while FlowLM generates the next latent. However, benchmarks showed it's
    // ~21% SLOWER than sequential due to:
    // - Thread spawning and channel synchronization overhead
    // - CPU contention (MKL already parallelizes internally)
    // - Bounded channel backpressure when Mimi can't keep up
    //
    // Keeping this commented out for future exploration with GPU acceleration
    // where FlowLM and Mimi could run on different hardware.
    //
    // To re-enable: uncomment the method below and test with:
    //   model.generate_parallel(text, &voice_state)
    //
    /*
    /// Generate audio with parallel FlowLM + Mimi decoding
    ///
    /// Uses std::thread to decode Mimi audio in parallel with FlowLM generation.
    /// While Mimi decodes frame N, FlowLM generates frame N+1.
    ///
    /// NOTE: Currently slower than sequential due to thread overhead on CPU.
    /// May be useful for GPU acceleration in the future.
    pub fn generate_parallel(&self, text: &str, voice_state: &ModelState) -> Result<Tensor> {
        use std::sync::mpsc;
        use std::thread;

        // Channel for sending latents from FlowLM to Mimi decoder
        let (latent_tx, latent_rx) = mpsc::sync_channel::<Option<(Tensor, usize)>>(4);
        // Channel for receiving decoded audio from Mimi
        let (audio_tx, audio_rx) = mpsc::channel::<Result<Tensor>>();

        // Clone what the decoder thread needs
        let mimi = self.mimi.clone();
        let emb_mean = self.flow_lm.emb_mean.clone();
        let emb_std = self.flow_lm.emb_std.clone();

        // Spawn Mimi decoder thread
        let decoder_handle = thread::spawn(move || {
            let mut mimi_state = init_states(1, 1000);

            while let Ok(Some((next_latent, step))) = latent_rx.recv() {
                let result = (|| -> Result<Tensor> {
                    let next_latent_denorm = next_latent
                        .broadcast_mul(&emb_std)?
                        .broadcast_add(&emb_mean)?;

                    let mimi_input = next_latent_denorm.unsqueeze(1)?.transpose(1, 2)?;
                    let quantized = mimi.quantize(&mimi_input)?;
                    let audio = mimi
                        .decode_from_latent(&quantized, &mut mimi_state, step)
                        .map_err(|e| anyhow::anyhow!(e))?;

                    Ok(audio)
                })();

                if audio_tx.send(result).is_err() {
                    break; // Receiver dropped
                }
            }
        });

        // Main thread: generate latents with FlowLM
        let chunks = self.split_into_best_sentences(text);
        let mut expected_frames = 0;

        for chunk_text in chunks {
            let mut state = voice_state.clone();
            let prepared_text = prepare_text_prompt(&chunk_text);

            let tokens = self.conditioner.prepare(&prepared_text, &self.device)?;
            let text_embeddings = self.conditioner.forward(&tokens)?;

            // Initial text prompt
            self.flow_lm
                .transformer
                .forward(&text_embeddings, &mut state, 0)?;

            let max_gen_len = (prepared_text.split_whitespace().count() + 2) * 13;
            let frames_after_eos = estimate_frames_after_eos(&chunk_text);

            let mut backbone_input = self.flow_lm.bos_emb.clone().reshape((1, 1, self.ldim))?;
            let mut eos_step: Option<usize> = None;

            let time_embeddings = self.flow_lm.flow_net.compute_time_embeddings(
                self.lsd_decode_steps,
                &self.device,
                DType::F32,
            )?;

            let empty_text_embeddings = Tensor::zeros((1, 0, self.dim), DType::F32, &self.device)?;

            for step in 0..max_gen_len {
                let (next_latent, is_eos) = self.flow_lm.forward(
                    &backbone_input,
                    &empty_text_embeddings,
                    &mut state,
                    &time_embeddings,
                    self.temp,
                    self.eos_threshold,
                    step,
                )?;

                // Send latent to decoder thread (non-blocking with bounded channel)
                if latent_tx.send(Some((next_latent.clone(), step))).is_err() {
                    break;
                }
                expected_frames += 1;

                if is_eos && eos_step.is_none() {
                    eos_step = Some(step);
                }

                if let Some(e_step) = eos_step {
                    if step >= e_step + frames_after_eos {
                        break;
                    }
                }

                backbone_input = next_latent.unsqueeze(1)?;
            }
        }

        // Signal decoder thread to finish
        let _ = latent_tx.send(None);

        // Collect all decoded audio frames
        let mut audio_chunks = Vec::with_capacity(expected_frames);
        for _ in 0..expected_frames {
            match audio_rx.recv() {
                Ok(Ok(frame)) => audio_chunks.push(frame),
                Ok(Err(e)) => return Err(e),
                Err(_) => break,
            }
        }

        // Wait for decoder thread
        let _ = decoder_handle.join();

        if audio_chunks.is_empty() {
            anyhow::bail!("No audio generated");
        }

        let audio = Tensor::cat(&audio_chunks, 2)?;
        let audio = audio.squeeze(0)?;
        Ok(audio)
    }
    */

    /// Generate audio from text with pause handling
    ///
    /// This method parses pause markers in the text and inserts silence
    /// at appropriate positions. Supports:
    /// - Explicit pauses: `[pause:500ms]` or `[pause:1s]`
    /// - Natural pauses from punctuation are handled during generation
    ///
    /// # Example
    /// ```ignore
    /// let audio = model.generate_with_pauses("Hello... [pause:500ms] world", &voice_state)?;
    /// ```
    pub fn generate_with_pauses(&self, text: &str, voice_state: &ModelState) -> Result<Tensor> {
        let mut audio_chunks = Vec::new();

        for chunk in self.generate_stream_long(text, voice_state) {
            audio_chunks.push(chunk?);
        }

        // Concatenate all audio chunks
        if audio_chunks.is_empty() {
            anyhow::bail!("No audio generated");
        }
        let audio = Tensor::cat(&audio_chunks, 2)?;
        // Remove batch dimension
        let audio = audio.squeeze(0)?;

        Ok(audio)
    }

    /// Generate audio stream from text with voice state
    ///
    /// Returns an iterator that yields audio chunks (one per Mimi frame).
    /// Generate audio stream from text with voice state
    ///
    /// Returns an iterator that yields audio chunks (one per Mimi frame).
    ///
    /// This method splits the text into optimal sentences and generates each independently,
    /// matching Python's behavior to maintain O(N) complexity for long texts.
    pub fn generate_stream<'a, 'b, 'c>(
        &'a self,
        text: &'b str,
        voice_state: &'c ModelState,
    ) -> Box<dyn Iterator<Item = Result<Tensor>> + 'a> {
        self.generate_stream_opts(text, voice_state, GenerateOptions::default())
    }

    /// Like [`Self::generate_stream`] with explicit per-call options
    /// (upstream `max_tokens` / `frames_after_eos`).
    pub fn generate_stream_opts<'a, 'b, 'c>(
        &'a self,
        text: &'b str,
        voice_state: &'c ModelState,
        options: GenerateOptions,
    ) -> Box<dyn Iterator<Item = Result<Tensor>> + 'a> {
        // Split text into chunks to avoid quadratic complexity scaling
        let chunks = self.split_into_best_sentences_with(text, options.max_tokens);

        // Clone voice state so the iterator owns a copy, untied from lifetime 'c
        let voice_state_owned = voice_state.clone();

        // Create an iterator that processes each chunk sequentially
        let iterator = chunks.into_iter().flat_map(move |chunk_text| {
            // We need to return an iterator for each chunk.
            // We pass a reference to the owned voice state captured by the closure.
            self.generate_stream_segment(chunk_text, &voice_state_owned, options.frames_after_eos)
        });

        Box::new(iterator)
    }

    /// Generate audio stream from text with voice state, returning an owned iterator.
    ///
    /// This is useful for WASM bindings where the iterator must be 'static.
    pub fn generate_stream_owned(
        &self,
        text: &str,
        voice_state: &ModelState,
    ) -> Box<dyn Iterator<Item = Result<Tensor>> + 'static> {
        let model = self.clone();
        let voice_state_owned = voice_state.clone();
        let chunks = model.split_into_best_sentences(text);

        let iterator = chunks.into_iter().flat_map(move |chunk_text| {
            model.generate_stream_segment(chunk_text, &voice_state_owned, None)
        });

        Box::new(iterator)
    }

    /// Internal helper to generate a single segment (short text) matching Python's _generate
    fn generate_stream_segment(
        &self,
        text: String,
        voice_state: &ModelState,
        frames_after_eos_override: Option<usize>,
    ) -> Box<dyn Iterator<Item = Result<Tensor>>> {
        let mut state = voice_state.clone();
        let mut mimi_state = init_states(1, 1000);

        // Prepare text
        let prepared_text = prepare_text_prompt(
            &text,
            self.pad_with_spaces_for_short_inputs,
            self.remove_semicolons,
        );

        // Error handling for preparation failures inside the iterator
        let tokens = match self.conditioner.prepare(&prepared_text, &self.device) {
            Ok(t) => t,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };

        let text_embeddings = match self.conditioner.forward(&tokens) {
            Ok(e) => e,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };

        // Initial text prompt
        if let Err(e) = self
            .flow_lm
            .transformer
            .forward(&text_embeddings, &mut state, 0)
        {
            return Box::new(std::iter::once(Err(anyhow::Error::from(e))));
        }

        // Removed redundant increment_steps("offset") - handled internally by RoPE/Attention with current_end_len

        // Upstream `_estimate_max_gen_len`: token-based length budget.
        let token_count = tokens.dims().last().copied().unwrap_or(0);
        let max_gen_len = self.estimate_max_gen_len(token_count);
        // Explicit caller override first (upstream frames_after_eos), then
        // #155: the model-recommended EOS tail when the config sets it
        // (french_24l = 8), otherwise the length heuristic.
        let frames_after_eos = frames_after_eos_override
            .or(self.model_recommended_frames_after_eos)
            .unwrap_or_else(|| estimate_frames_after_eos(&text));

        let mut backbone_input = match self.flow_lm.bos_emb.clone().reshape((1, 1, self.ldim)) {
            Ok(t) => t,
            Err(e) => return Box::new(std::iter::once(Err(anyhow::Error::from(e)))),
        };

        let mut eos_step: Option<usize> = None;
        let mut finished = false;

        // We need to move 'self' (reference) and owned data into the closure
        // But 'self' is in `generate_stream` lifetime?
        // We clone needed cheap things or use references.
        // `flow_lm`, `mimi` are part of self.
        // The closure will borrow `self`.

        // To make the iterator valid 'static or bound to self, we use move.
        // But we need access to self inside.
        // We can clone `self` if cheap? No, TTSModel is large (holds models).
        // But TTSModel derives Clone! And models are wrappers around Arcs (Candle tensors/vars).
        // So cloning TTSModel is CHEAP (shallow copy of Arc pointers).
        let model = self.clone();

        // Pre-compute time embeddings for the entire segment to avoid re-computing every frame
        // Now returns a single batched Tensor [num_steps, channels]
        let time_embeddings = match model.flow_lm.flow_net.compute_time_embeddings(
            model.lsd_decode_steps,
            &model.device,
            DType::F32,
        ) {
            Ok(te) => te,
            Err(e) => return Box::new(std::iter::once(Err(anyhow::Error::from(e)))),
        };

        let empty_text_embeddings =
            Tensor::zeros((1, 0, model.dim), DType::F32, &model.device).unwrap();

        Box::new((0..max_gen_len).map_while(move |step| {
            if finished {
                return None;
            }

            // Text embeddings are already processed into state during initialization (line 752-757),
            // so we always pass empty text embeddings during autoregressive generation.
            // Passing text_embeddings again would cause duplicate/repeated speech.
            let text_tokens_to_pass = &empty_text_embeddings;

            let (next_latent, is_eos) = match tracing::info_span!("flow_lm.forward", step = step)
                .in_scope(|| {
                    model.flow_lm.forward(
                        &backbone_input,
                        text_tokens_to_pass,
                        &mut state,
                        &time_embeddings,
                        model.temp,
                        model.eos_threshold,
                        step,
                    )
                }) {
                Ok(res) => res,
                Err(e) => return Some(Err(anyhow::anyhow!(e))),
            };

            let audio_frame = match (|| -> Result<Tensor> {
                let next_latent_denorm = next_latent
                    .broadcast_mul(&model.flow_lm.emb_std)?
                    .broadcast_add(&model.flow_lm.emb_mean)?;

                let mimi_input = next_latent_denorm.unsqueeze(1)?.transpose(1, 2)?;
                let quantized = model.mimi.quantize(&mimi_input)?;
                let audio = tracing::info_span!("mimi.decode_from_latent", step = step)
                    .in_scope(|| {
                        model
                            .mimi
                            .decode_from_latent(&quantized, &mut mimi_state, step)
                    })
                    .map_err(|e| anyhow::anyhow!(e))?;

                // Removed redundant increment_steps("offset") for mimi

                Ok(audio)
            })() {
                Ok(frame) => frame,
                Err(e) => return Some(Err(e)),
            };

            if is_eos && eos_step.is_none() {
                eos_step = Some(step);
            }

            if let Some(e_step) = eos_step
                && step >= e_step + frames_after_eos
            {
                finished = true;
            }

            backbone_input = next_latent.unsqueeze(1).unwrap();

            // Removed redundant increment_steps("offset") for FlowLM - handled by attention state

            Some(Ok(audio_frame))
        }))
    }

    /// Generate audio stream from long text by segmenting it
    pub fn generate_stream_long<'a>(
        &'a self,
        text: &str,
        voice_state: &'a ModelState,
    ) -> impl Iterator<Item = Result<Tensor>> + 'a {
        self.generate_stream_long_opts(text, voice_state, GenerateOptions::default())
    }

    /// Like [`Self::generate_stream_long`] with explicit per-call options
    /// (upstream `max_tokens` / `frames_after_eos`).
    pub fn generate_stream_long_opts<'a>(
        &'a self,
        text: &str,
        voice_state: &'a ModelState,
        options: GenerateOptions,
    ) -> impl Iterator<Item = Result<Tensor>> + 'a {
        use crate::pause::{parse_text_with_pauses, silence_samples};

        let parsed = parse_text_with_pauses(text);
        let mut segments = Vec::new();

        // Interleave text chunks and pauses
        let mut last_pos = 0;
        for pause in &parsed.pauses {
            if pause.position > last_pos {
                let text_seg = &parsed.clean_text[last_pos..pause.position];
                if !text_seg.trim().is_empty() {
                    segments.push(Segment::Text(text_seg.to_string()));
                }
            }
            segments.push(Segment::Pause(pause.duration_ms));

            // Explicit pauses were replaced by a single space in clean_text
            // Natural pauses (commas, ellipses) are still in clean_text
            if pause.original.starts_with("[pause:") {
                last_pos = pause.position + 1;
            } else {
                last_pos = pause.position + pause.original.len();
            }
        }
        if last_pos < parsed.clean_text.len() {
            let text_seg = &parsed.clean_text[last_pos..];
            if !text_seg.trim().is_empty() {
                segments.push(Segment::Text(text_seg.to_string()));
            }
        }

        let model = self;
        segments.into_iter().flat_map(move |seg| match seg {
            Segment::Text(s) => {
                let iter = model.generate_stream_opts(&s, voice_state, options);
                Box::new(iter) as Box<dyn Iterator<Item = Result<Tensor>>>
            }
            Segment::Pause(ms) => {
                let n_samples = silence_samples(ms, model.sample_rate as u32);
                let silence_res = Tensor::zeros(
                    (1, model.mimi.channels, n_samples),
                    DType::F32,
                    &model.device,
                );
                Box::new(std::iter::once(silence_res.map_err(anyhow::Error::from)))
                    as Box<dyn Iterator<Item = Result<Tensor>>>
            }
        })
    }
    pub fn estimate_generation_steps(&self, text: &str) -> usize {
        let prepared = prepare_text_prompt(
            text,
            self.pad_with_spaces_for_short_inputs,
            self.remove_semicolons,
        );
        let token_count = self
            .conditioner
            .count_tokens(&prepared)
            .unwrap_or_else(|_| prepared.split_whitespace().count() * 2);
        self.estimate_max_gen_len(token_count)
    }

    /// Upstream `_estimate_max_gen_len`: how many latent frames to budget for
    /// a prompt of `token_count` tokens (tokens/3 seconds plus 2 s padding).
    fn estimate_max_gen_len(&self, token_count: usize) -> usize {
        const TOKENS_PER_SECOND_ESTIMATE: f64 = 3.0;
        const GEN_SECONDS_PADDING: f64 = 2.0;
        let gen_len_sec = token_count as f64 / TOKENS_PER_SECOND_ESTIMATE + GEN_SECONDS_PADDING;
        (gen_len_sec * self.frame_rate).ceil() as usize
    }
}

/// Internal segment type for interleaving text and pauses
enum Segment {
    Text(String),
    Pause(u32),
}

/// Bridge an upstream packed KV cache `[2, B, T, H, D]` into this crate's
/// per-layer buffers `(k_buf, v_buf)` of shape `[B, H, T, D]`.
/// Find token indices where text should be split based on boundary tokens,
/// mirroring upstream `_find_boundary_indices`: each consecutive pair of
/// returned indices delimits one segment; the first element is always 0 and
/// the last is always `tokens.len()`.
fn find_boundary_indices(tokens: &[u32], boundary_tokens: &[u32]) -> Vec<usize> {
    let mut indices = vec![0];
    let mut previous_was_boundary = false;
    for (idx, token) in tokens.iter().enumerate() {
        if boundary_tokens.contains(token) {
            previous_was_boundary = true;
        } else {
            if previous_was_boundary {
                indices.push(idx);
            }
            previous_was_boundary = false;
        }
    }
    indices.push(tokens.len());
    indices
}

fn unpack_kv_cache(packed: &Tensor) -> Result<(Tensor, Tensor)> {
    // [2, B, T, H, D] -> [2, B, H, T, D]
    let transposed = packed.transpose(2, 3)?;
    let keys = transposed.get(0)?.contiguous()?;
    let values = transposed.get(1)?.contiguous()?;
    Ok((keys, values))
}

/// Export a voice/model state to a `.safetensors` file in upstream's
/// `export_model_state` format: flat `module/key` names (module without the
/// crate's `flow_lm.` prefix), the K/V cache packed as `[2, B, T, H, D]`
/// truncated to the valid length, and `offset` as a `long[1]`. Files written
/// here load in the Python implementation and vice versa (the reader also
/// accepts the pre-#85 `current_end` form).
#[cfg(not(target_arch = "wasm32"))]
pub fn export_model_state<P: AsRef<std::path::Path>>(state: &ModelState, dest: P) -> Result<()> {
    let mut flat: HashMap<String, Tensor> = HashMap::new();

    for (module_name, module_state) in state {
        let (Some(k_buf), Some(v_buf)) = (
            module_state.get(ATTN_K_BUF_KEY),
            module_state.get(ATTN_V_BUF_KEY),
        ) else {
            continue;
        };
        let cursor = crate::voice_state::read_attention_cursor(module_state);
        if cursor.len == 0 {
            continue;
        }

        // [B, H, T, D] -> valid slice -> [B, T, H, D]
        let k = k_buf.narrow(2, 0, cursor.len)?.transpose(1, 2)?;
        let v = v_buf.narrow(2, 0, cursor.len)?.transpose(1, 2)?;
        let cache = Tensor::stack(&[&k, &v], 0)?.contiguous()?;

        let upstream_name = module_name.strip_prefix("flow_lm.").unwrap_or(module_name);
        flat.insert(format!("{upstream_name}/cache"), cache);
        flat.insert(
            format!("{upstream_name}/offset"),
            Tensor::new(&[cursor.pos as i64], k_buf.device())?,
        );
    }

    if flat.is_empty() {
        anyhow::bail!("model state holds no attention caches to export");
    }

    candle_core::safetensors::save(&flat, dest)?;
    Ok(())
}

/// Find the config file path for a variant
fn find_config_path(variant: &str) -> Result<std::path::PathBuf> {
    // A path to a YAML file is used directly (upstream's `config=` argument).
    if variant.ends_with(".yaml") || variant.ends_with(".yml") {
        let path = std::path::PathBuf::from(variant);
        if path.exists() {
            return Ok(path);
        }
        anyhow::bail!("Config file not found: {}. Did you make a typo?", variant);
    }

    let filename = format!("{}.yaml", variant);

    // 1. Try relative to Rust crate (crates/pocket-tts/config)
    let crate_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let crate_config = crate_path.join("config").join(&filename);
    if crate_config.exists() {
        return Ok(crate_config);
    }

    // 2. Try relative to workspace root (for tests/cli)
    // Go up 2 levels if in crates/pocket-tts or crates/pocket-tts-cli
    let mut current = crate_path.as_path();
    for _ in 0..3 {
        let python_config = current
            .join("python-reference")
            .join("pocket_tts")
            .join("config")
            .join(&filename);
        if python_config.exists() {
            return Ok(python_config);
        }

        // Also try new crates structure if running from cli
        let crates_config = current
            .join("crates")
            .join("pocket-tts")
            .join("config")
            .join(&filename);
        if crates_config.exists() {
            return Ok(crates_config);
        }

        if let Some(parent) = current.parent() {
            current = parent;
        } else {
            break;
        }
    }

    // 3. Try current directory
    let local_path = std::path::PathBuf::from("config").join(&filename);
    if local_path.exists() {
        return Ok(local_path);
    }

    // Upstream lists the available languages when the name is unknown.
    let crate_config_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config");
    let mut available: Vec<String> = std::fs::read_dir(&crate_config_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let p = e.path();
                    (p.extension().and_then(|x| x.to_str()) == Some("yaml"))
                        .then(|| p.file_stem().unwrap_or_default().to_string_lossy().into_owned())
                })
                .collect()
        })
        .unwrap_or_default();
    available.sort();
    anyhow::bail!(
        "Config file {} not found. Did you make a typo? Available languages: {:?}",
        filename,
        available
    )
}

/// Prepare text for generation, stripping pause markers for TTS processing
fn prepare_text_prompt(
    text: &str,
    pad_with_spaces_for_short_inputs: bool,
    remove_semicolons: bool,
) -> String {
    // First strip any explicit pause markers
    let text = crate::pause::strip_pause_markers(text);

    let mut text = text.trim().to_string();
    if text.is_empty() {
        return ".".to_string(); // Or handle error
    }

    text = text.replace(['\n', '\r'], " ").replace("  ", " ");

    // #155: the multilingual checkpoints replace ';' with ',' before
    // tokenisation (the French SentencePiece handles ',' pauses better).
    if remove_semicolons {
        text = text.replace(';', ",");
    }

    let word_count = text.split_whitespace().count();

    // Ensure first character is uppercase
    if let Some(first) = text.chars().next()
        && !first.is_uppercase()
    {
        text = format!("{}{}", first.to_uppercase(), &text[first.len_utf8()..]);
    }

    // Ensure ends with punctuation
    if let Some(last) = text.chars().last()
        && last.is_alphanumeric()
    {
        text.push('.');
    }

    // Python logic: prepend spaces if too short. #155 gates this behind a
    // per-checkpoint flag — the English models need it, the multilingual
    // ones must not pad.
    if pad_with_spaces_for_short_inputs && word_count < 5 {
        text = format!("{}{}", " ".repeat(8), text);
    }

    text
}

/// Estimate frames after EOS based on text length
pub fn estimate_frames_after_eos(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    if word_count <= 4 {
        3 + 2 // prepare_text_prompt guess + 2
    } else {
        1 + 2 // prepare_text_prompt guess + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_text_prompt() {
        // Short texts (<5 words) get 8 spaces prepended (padding enabled)
        assert_eq!(
            prepare_text_prompt("hello world", true, false),
            "        Hello world."
        );
        assert_eq!(
            prepare_text_prompt("Hello world.", true, false),
            "        Hello world."
        );
        assert_eq!(
            prepare_text_prompt("  hello  ", true, false),
            "        Hello."
        );
        // Long texts don't get spaces
        assert_eq!(
            prepare_text_prompt("one two three four five", true, false),
            "One two three four five."
        );
        // Padding disabled: short text is not padded
        assert_eq!(
            prepare_text_prompt("hello world", false, false),
            "Hello world."
        );
        // remove_semicolons: ';' becomes ','
        assert_eq!(
            prepare_text_prompt("one two; three four five", false, true),
            "One two, three four five."
        );
    }

    #[test]
    fn test_find_config_path() {
        // This MUST pass now that we've moved the config into the crate
        let result = find_config_path("b6369a24");
        assert!(result.is_ok(), "Config file should be found");
        let path = result.unwrap();
        assert!(path.exists(), "Config file path should exist");
    }

    #[test]
    fn test_prepare_text_prompt_strips_pause_markers() {
        // Pause markers should be stripped from text
        let result = prepare_text_prompt("Hello [pause:500ms] world", true, false);
        // The pause marker should be gone, replaced with space
        assert!(!result.contains("[pause:"));
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn test_prepare_text_prompt_handles_multiple_pauses() {
        let result = prepare_text_prompt("One [pause:100ms] two [pause:1s] three", true, false);
        assert!(!result.contains("[pause:"));
        assert!(result.contains("One"));
        assert!(result.contains("two"));
        assert!(result.contains("three"));
    }

    #[test]
    fn test_estimate_frames_after_eos() {
        // Short text (<= 4 words)
        assert_eq!(estimate_frames_after_eos("Hello world"), 5);
        // Longer text (> 4 words)
        assert_eq!(estimate_frames_after_eos("One two three four five"), 3);
    }

    #[test]
    #[cfg(feature = "quantized")]
    fn test_load_quantized_requires_feature() {
        // This test only runs with --features quantized
        // It verifies the load_quantized method exists and compiles
        // Actual model loading requires HF_TOKEN
    }
}
