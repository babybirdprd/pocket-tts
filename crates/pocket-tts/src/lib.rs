pub mod audio;
pub mod conditioners;
pub mod config;
pub mod models;
pub mod modules;
pub mod pause;
pub mod quantize;
pub mod tts_model;
pub mod voice_state;
pub mod weights;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use pause::{ParsedText, PauseMarker, parse_text_with_pauses};
pub use quantize::{MaybeQuantLinear, QuantizeGroup, RECOMMENDED_CONFIG};
pub use tts_model::{GenerateOptions, TTSModel};
#[cfg(not(target_arch = "wasm32"))]
pub use tts_model::export_model_state;
pub use voice_state::ModelState;
