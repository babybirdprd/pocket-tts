//! Configuration types for pocket-tts, matching Python's utils/config.py

use serde::Deserialize;
use std::path::Path;

/// Flow network configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowConfig {
    pub dim: usize,
    pub depth: usize,
}

/// Transformer configuration for FlowLM
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLMTransformerConfig {
    pub hidden_scale: usize,
    pub max_period: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

/// Lookup table (text conditioner) configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupTableConfig {
    pub dim: usize,
    pub n_bins: usize,
    pub tokenizer: String,
    pub tokenizer_path: String,
}

/// FlowLM model configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLMConfig {
    pub dtype: String,
    pub flow: FlowConfig,
    pub transformer: FlowLMTransformerConfig,
    pub lookup_table: LookupTableConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
    /// If true, prepend a BOS embedding before the voice conditioning step.
    /// Mirrors Python's `FlowLMConfig.insert_bos_before_voice` (default False).
    #[serde(default)]
    pub insert_bos_before_voice: bool,
}

/// SEANet encoder/decoder configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SEANetConfig {
    pub dimension: usize,
    pub channels: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_base: usize,
    pub pad_mode: String,
    pub compress: usize,
}

/// Transformer configuration for Mimi
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimiTransformerConfig {
    pub d_model: usize,
    pub input_dimension: usize,
    pub output_dimensions: Vec<usize>,
    pub num_heads: usize,
    pub num_layers: usize,
    pub layer_scale: f64,
    pub context: usize,
    #[serde(default = "default_max_period")]
    pub max_period: f64,
    pub dim_feedforward: usize,
}

fn default_max_period() -> f64 {
    10000.0
}

/// Quantizer configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizerConfig {
    pub dimension: usize,
    pub output_dimension: usize,
}

/// Mimi model configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MimiConfig {
    pub dtype: String,
    pub sample_rate: usize,
    pub channels: usize,
    pub frame_rate: f64,
    pub seanet: SEANetConfig,
    pub transformer: MimiTransformerConfig,
    pub quantizer: QuantizerConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
    /// Inner latent dimension (post-quantizer) used by newer model packagings.
    /// Mirrors Python's `MimiConfig.inner_dim` (defaults to None / unused).
    #[serde(default)]
    pub inner_dim: Option<usize>,
    /// Outer latent dimension. Mirrors Python's `MimiConfig.outer_dim`.
    #[serde(default)]
    pub outer_dim: Option<usize>,
}

/// Root configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub flow_lm: FlowLMConfig,
    pub mimi: MimiConfig,
    #[serde(default)]
    pub weights_path: Option<String>,
    #[serde(default)]
    pub weights_path_without_voice_cloning: Option<String>,
    /// Pad short text inputs with leading spaces before tokenization.
    /// Mirrors Python's `Config.pad_with_spaces_for_short_inputs` (default False).
    #[serde(default)]
    pub pad_with_spaces_for_short_inputs: bool,
    /// Strip semicolons during text preprocessing rather than treating them as pauses.
    /// Mirrors Python's `Config.remove_semicolons` (default False).
    #[serde(default)]
    pub remove_semicolons: bool,
    /// Optional override for how many Mimi frames to generate after EOS detection.
    /// When `Some`, replaces the heuristic in `estimate_frames_after_eos`.
    /// Mirrors Python's `Config.model_recommended_frames_after_eos`.
    #[serde(default)]
    pub model_recommended_frames_after_eos: Option<usize>,
}

/// Load configuration from a YAML file
pub fn load_config<P: AsRef<Path>>(path: P) -> anyhow::Result<Config> {
    let contents = std::fs::read_to_string(path)?;
    let config: Config = serde_yaml::from_str(&contents)?;
    Ok(config)
}

/// Default generation parameters (matching Python's default_parameters.py)
pub mod defaults {
    pub const TEMPERATURE: f32 = 0.7;
    pub const LSD_DECODE_STEPS: usize = 1;
    pub const NOISE_CLAMP: Option<f32> = None;
    pub const EOS_THRESHOLD: f32 = -4.0;
    pub const DEFAULT_VARIANT: &str = "b6369a24";
    /// Default language used when neither `--language` nor `--variant` is set.
    /// Matches Python's `DEFAULT_LANGUAGE = "english"`.
    pub const DEFAULT_LANGUAGE: &str = "english";
    /// Maximum tokens per chunk for sentence batching.
    pub const MAX_TOKEN_PER_CHUNK: usize = 50;
}

/// Per-language default text and voice, mirroring Python's `default_parameters.py`.
pub mod language_defaults {
    /// Recognized language stems (correspond to bundled YAML files).
    /// Includes both base languages and 24-layer variants.
    pub const LANGUAGES: &[&str] = &[
        "english",
        "english_2026-01",
        "english_2026-04",
        "french_24l",
        "german",
        "german_24l",
        "italian",
        "italian_24l",
        "portuguese",
        "portuguese_24l",
        "spanish",
        "spanish_24l",
    ];

    /// Default greeting text for a language. Falls back to English when no
    /// match is found (matches Python's substring match in
    /// `get_default_text_for_language`).
    pub fn default_text(language: Option<&str>) -> &'static str {
        let lang = language.unwrap_or("english");
        if lang.contains("french") {
            "Bonjour le monde. Je suis le TTS de poche de Kyutai. \
             Je suis assez rapide pour fonctionner sur de petits CPU. \
             J'espère que vous m'aimerez."
        } else if lang.contains("german") {
            "Hallo Welt. Ich bin Pocket TTS von Kyutai. \
             Ich bin schnell genug, um auch auf kleinen CPUs zu laufen. \
             Ich hoffe, ich gefalle dir."
        } else if lang.contains("portuguese") {
            "Olá mundo. Eu sou o Pocket TTS da Kyutai. \
             Sou rápido o suficiente para rodar em CPUs pequenas. \
             Espero que você goste de mim."
        } else if lang.contains("italian") {
            "Ciao mondo. Sono il Pocket TTS di Kyutai. \
             Sono abbastanza veloce da funzionare su piccole CPU. \
             Spero che ti piacerò."
        } else if lang.contains("spanish") {
            "Hola mundo. Soy el Pocket TTS de Kyutai. \
             Soy lo suficientemente rápido para funcionar en pequeñas CPU. \
             Espero que te guste."
        } else {
            "Hello world. I am Kyutai's Pocket TTS. \
             I'm fast enough to run on small CPUs. \
             I hope you'll like me."
        }
    }

    /// Default voice name for a language. Falls back to "alba" when no match
    /// is found (matches Python's substring match in
    /// `get_default_voice_for_language`).
    pub fn default_voice(language: Option<&str>) -> &'static str {
        let Some(lang) = language else { return "alba" };
        if lang.contains("italian") {
            "giovanni"
        } else if lang.contains("spanish") {
            "lola"
        } else if lang.contains("german") {
            "juergen"
        } else if lang.contains("portuguese") {
            "rafael"
        } else if lang.contains("french") {
            "estelle"
        } else {
            "alba"
        }
    }

    /// Whether `name` is a recognized language stem. Used by the CLI to
    /// distinguish `--language` values from arbitrary `--variant` strings.
    pub fn is_known_language(name: &str) -> bool {
        LANGUAGES.contains(&name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn crate_config_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config")
    }

    #[test]
    fn test_french_24l_config_parses_new_fields() {
        let path = crate_config_dir().join("french_24l.yaml");
        let config = load_config(&path).expect("Failed to load french_24l config");
        // french_24l.yaml is the only bundled YAML that exercises BOTH
        // remove_semicolons=true and model_recommended_frames_after_eos=8.
        assert!(config.remove_semicolons);
        assert_eq!(config.model_recommended_frames_after_eos, Some(8));
        assert!(config.flow_lm.insert_bos_before_voice);
        assert_eq!(config.mimi.inner_dim, Some(32));
        assert_eq!(config.mimi.outer_dim, Some(512));
    }

    #[test]
    fn test_english_2026_01_config_padding_flag() {
        let path = crate_config_dir().join("english_2026-01.yaml");
        let config = load_config(&path).expect("Failed to load english_2026-01 config");
        // Only YAML that turns padding on.
        assert!(config.pad_with_spaces_for_short_inputs);
        // BOS-before-voice is *off* in this 2026-01 model.
        assert!(!config.flow_lm.insert_bos_before_voice);
    }

    #[test]
    fn test_load_legacy_b6369a24_config() {
        let path = crate_config_dir().join("b6369a24.yaml");
        let config = load_config(&path).expect("Failed to load b6369a24 config");

        // Verify FlowLM config
        assert_eq!(config.flow_lm.transformer.d_model, 1024);
        assert_eq!(config.flow_lm.transformer.num_heads, 16);
        assert_eq!(config.flow_lm.transformer.num_layers, 6);
        assert_eq!(config.flow_lm.flow.dim, 512);
        assert_eq!(config.flow_lm.flow.depth, 6);
        assert_eq!(config.flow_lm.lookup_table.n_bins, 4000);

        // Verify Mimi config
        assert_eq!(config.mimi.sample_rate, 24000);
        assert_eq!(config.mimi.channels, 1);
        assert!((config.mimi.frame_rate - 12.5).abs() < 1e-6);
        assert_eq!(config.mimi.seanet.dimension, 512);
        assert_eq!(config.mimi.seanet.ratios, vec![6, 5, 4]);
        assert_eq!(config.mimi.transformer.num_layers, 2);
        assert_eq!(config.mimi.quantizer.dimension, 32);

        // Legacy config must default new optional fields to false/None
        assert!(!config.flow_lm.insert_bos_before_voice);
        assert!(!config.pad_with_spaces_for_short_inputs);
        assert!(!config.remove_semicolons);
        assert!(config.model_recommended_frames_after_eos.is_none());
        assert!(config.mimi.inner_dim.is_none());
        assert!(config.mimi.outer_dim.is_none());
    }

    #[test]
    fn test_all_bundled_language_configs_load() {
        let dir = crate_config_dir();
        let mut count = 0;
        for entry in std::fs::read_dir(&dir).expect("config dir missing") {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
                continue;
            }
            count += 1;
            load_config(&path)
                .unwrap_or_else(|e| panic!("Failed to load {}: {}", path.display(), e));
        }
        // We expect b6369a24 + 12 upstream language YAMLs
        assert!(count >= 13, "Expected ≥13 YAMLs in {}, found {}", dir.display(), count);
    }

    #[test]
    fn test_language_defaults_voice_mapping() {
        use language_defaults::default_voice;
        assert_eq!(default_voice(Some("italian")), "giovanni");
        assert_eq!(default_voice(Some("italian_24l")), "giovanni");
        assert_eq!(default_voice(Some("spanish")), "lola");
        assert_eq!(default_voice(Some("german")), "juergen");
        assert_eq!(default_voice(Some("german_24l")), "juergen");
        assert_eq!(default_voice(Some("portuguese")), "rafael");
        assert_eq!(default_voice(Some("french_24l")), "estelle");
        assert_eq!(default_voice(Some("english")), "alba");
        assert_eq!(default_voice(None), "alba");
    }

    #[test]
    fn test_language_defaults_text_mapping() {
        use language_defaults::default_text;
        assert!(default_text(Some("italian")).starts_with("Ciao mondo"));
        assert!(default_text(Some("italian_24l")).starts_with("Ciao mondo"));
        assert!(default_text(Some("french_24l")).starts_with("Bonjour"));
        assert!(default_text(Some("german")).starts_with("Hallo Welt"));
        assert!(default_text(Some("spanish")).starts_with("Hola mundo"));
        assert!(default_text(Some("portuguese")).starts_with("Olá mundo"));
        assert!(default_text(Some("english")).starts_with("Hello world"));
        assert!(default_text(None).starts_with("Hello world"));
    }

    #[test]
    fn test_deny_unknown_fields_at_root() {
        let bad = r#"
flow_lm:
  dtype: float32
  flow: { depth: 6, dim: 512 }
  transformer:
    hidden_scale: 4
    max_period: 10000
    d_model: 1024
    num_heads: 16
    num_layers: 6
  lookup_table:
    dim: 1024
    n_bins: 4000
    tokenizer: sentencepiece
    tokenizer_path: x
mimi:
  dtype: float32
  sample_rate: 24000
  channels: 1
  frame_rate: 12.5
  seanet:
    dimension: 512
    channels: 1
    n_filters: 64
    n_residual_layers: 1
    ratios: [6, 5, 4]
    kernel_size: 7
    residual_kernel_size: 3
    last_kernel_size: 3
    dilation_base: 2
    pad_mode: constant
    compress: 2
  transformer:
    d_model: 512
    num_heads: 8
    num_layers: 2
    layer_scale: 0.01
    context: 250
    dim_feedforward: 2048
    input_dimension: 512
    output_dimensions: [512]
  quantizer: { dimension: 32, output_dimension: 512 }
unknown_field: oops
"#;
        let res: Result<Config, _> = serde_yaml::from_str(bad);
        assert!(res.is_err(), "deny_unknown_fields should reject extra root fields");
    }
}
