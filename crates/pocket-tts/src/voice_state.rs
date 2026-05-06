//! Voice state management for streaming generation and voice cloning

use candle_core::{Device, IndexOp, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Model state type for stateful modules
pub type ModelState = HashMap<String, HashMap<String, Tensor>>;

/// Common per-attention state keys.
pub const ATTN_POS_KEY: &str = "pos";
pub const ATTN_LEN_KEY: &str = "l";
pub const ATTN_HEAD_KEY: &str = "head";
pub const ATTN_K_BUF_KEY: &str = "k_buf";
pub const ATTN_V_BUF_KEY: &str = "v_buf";

/// Cursor/scalar metadata for attention cache state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttentionCursor {
    pub pos: usize,
    pub len: usize,
    pub head: usize,
}

/// Initialize empty model state for all stateful modules
///
/// Creates a nested HashMap structure that will be populated
/// as modules run their forward passes.
pub fn init_states(_batch_size: usize, _seq_len: usize) -> ModelState {
    // Start with empty state - modules will populate as needed
    HashMap::new()
}

/// Get or create a module's state entry
pub fn get_or_create_state<'a>(
    state: &'a mut ModelState,
    module_name: &str,
) -> &'a mut HashMap<String, Tensor> {
    state.entry(module_name.to_string()).or_default()
}

fn tensor_to_usize(t: &Tensor) -> Option<usize> {
    if let Ok(v) = t.to_scalar::<i64>() {
        return Some(v.max(0) as usize);
    }
    if let Ok(v) = t.to_scalar::<u32>() {
        return Some(v as usize);
    }
    None
}

/// Read attention cursor values from a module state map.
pub fn read_attention_cursor(module_state: &HashMap<String, Tensor>) -> AttentionCursor {
    AttentionCursor {
        pos: module_state
            .get(ATTN_POS_KEY)
            .and_then(tensor_to_usize)
            .unwrap_or(0),
        len: module_state
            .get(ATTN_LEN_KEY)
            .and_then(tensor_to_usize)
            .unwrap_or(0),
        head: module_state
            .get(ATTN_HEAD_KEY)
            .and_then(tensor_to_usize)
            .unwrap_or(0),
    }
}

/// Write attention cursor values into a module state map.
pub fn write_attention_cursor(
    module_state: &mut HashMap<String, Tensor>,
    cursor: AttentionCursor,
    device: &candle_core::Device,
) -> Result<()> {
    module_state.insert(
        ATTN_POS_KEY.to_string(),
        Tensor::new(cursor.pos as u32, device)?,
    );
    module_state.insert(
        ATTN_LEN_KEY.to_string(),
        Tensor::new(cursor.len as i64, device)?,
    );
    module_state.insert(
        ATTN_HEAD_KEY.to_string(),
        Tensor::new(cursor.head as i64, device)?,
    );
    Ok(())
}

/// Read attention cursor for a module name from a full model state.
pub fn get_attention_cursor(state: &ModelState, module_name: &str) -> AttentionCursor {
    state
        .get(module_name)
        .map(read_attention_cursor)
        .unwrap_or_default()
}

/// Increment step counters in model state for all modules
///
/// This is used after processing tokens to update position information
/// for streaming generation.
pub fn increment_steps(state: &mut ModelState, key: &str, increment: usize) {
    for (_module_name, module_state) in state.iter_mut() {
        if let Some(step_tensor) = module_state.get_mut(key)
            && let Ok(current) = step_tensor.to_scalar::<i64>()
            && let Ok(new_tensor) = Tensor::new(current + increment as i64, step_tensor.device())
        {
            *step_tensor = new_tensor;
        }
    }
}

/// Get the current step/offset for a module
pub fn get_offset(state: &ModelState, module_name: &str) -> usize {
    state
        .get(module_name)
        .and_then(|s| s.get("offset"))
        .and_then(|t| t.to_scalar::<i64>().ok())
        .unwrap_or(0) as usize
}

/// Set the offset for a module
pub fn set_offset(state: &mut ModelState, module_name: &str, offset: usize) -> Result<()> {
    let module_state = get_or_create_state(state, module_name);
    let device = module_state
        .values()
        .next()
        .map(|t| t.device().clone())
        .unwrap_or(candle_core::Device::Cpu);
    module_state.insert("offset".to_string(), Tensor::new(offset as i64, &device)?);
    Ok(())
}

/// Import a pre-exported model state from a safetensors file.
///
/// Mirrors Python's `_import_model_state` in `pocket_tts/models/tts_model.py`,
/// which reads files produced by `export_model_state`. Each safetensors key is
/// of the form `module_name/tensor_key` (note the slash). The HF predefined
/// voice embeddings under
/// `kyutai/pocket-tts-without-voice-cloning/languages/{lang}/embeddings/{name}.safetensors`
/// use this format.
///
/// `module_prefix` is prepended to each module name with a `.` separator. This
/// is needed because the upstream export rooted state dicts at the FlowLM
/// (e.g. `transformer.layers.0.self_attn`), while Rust's `ModelState` uses
/// fully-qualified names (e.g. `flow_lm.transformer.layers.0.self_attn`).
///
/// Tensor key remappings (matching upstream behavior plus Rust's KV-cache
/// representation):
/// - `cache` (shape `[2, B, T, H, D]`) is split along axis 0 into `k_buf` and
///   `v_buf` with axis-1/2 transpose to Rust's `(B, H, T, D)` layout.
/// - `offset` is stored as-is and also seeds the attention cursor
///   (`pos = len = offset_value`, `head = 0`) for unwindowed transformers.
/// - `current_end` (legacy upstream key) is converted to `offset` using
///   `tensor.shape[0]` as the value, mirroring upstream's compatibility shim.
/// - All other keys pass through unchanged.
pub fn import_model_state_from_file<P: AsRef<Path>>(
    path: P,
    device: &Device,
    module_prefix: &str,
) -> anyhow::Result<ModelState> {
    let tensors = candle_core::safetensors::load(path, device)?;
    import_model_state_from_tensors(tensors, device, module_prefix)
}

/// Variant of [`import_model_state_from_file`] that accepts an already-loaded
/// tensor map (e.g. from `safetensors::load_buffer`).
pub fn import_model_state_from_tensors(
    tensors: HashMap<String, Tensor>,
    device: &Device,
    module_prefix: &str,
) -> anyhow::Result<ModelState> {
    let mut result: ModelState = HashMap::new();

    for (key, tensor) in tensors {
        let (module_name, tensor_key) = key.split_once('/').ok_or_else(|| {
            anyhow::anyhow!("expected '<module>/<key>' in safetensors key, got '{}'", key)
        })?;

        let full_module_name = if module_prefix.is_empty() {
            module_name.to_string()
        } else {
            format!("{}.{}", module_prefix, module_name)
        };

        let module_state = result.entry(full_module_name).or_default();

        match tensor_key {
            "cache" => {
                // Upstream packs K and V along axis 0: shape `[2, B, T, H, D]`.
                // Rust's attention buffers are `(B, H, T, D)` separately.
                let dims = tensor.dims();
                if dims.len() != 5 || dims[0] != 2 {
                    anyhow::bail!(
                        "expected cache shape [2, B, T, H, D], got {:?} for key {}",
                        dims,
                        key
                    );
                }
                let k = tensor.i(0)?.transpose(1, 2)?.contiguous()?; // (B, H, T, D)
                let v = tensor.i(1)?.transpose(1, 2)?.contiguous()?;
                module_state.insert(ATTN_K_BUF_KEY.to_string(), k);
                module_state.insert(ATTN_V_BUF_KEY.to_string(), v);
                // The cache's time dimension is the prompt length; record it.
                let prompt_len = dims[2];
                let cursor = AttentionCursor {
                    pos: prompt_len,
                    len: prompt_len,
                    head: 0,
                };
                write_attention_cursor(module_state, cursor, device)?;
            }
            "offset" => {
                // The upstream-exported `offset` is an int64 scalar (or
                // shape `[1]`). Use it as the canonical step counter.
                let off_val = read_offset_scalar(&tensor)?;
                module_state.insert("offset".to_string(), Tensor::new(off_val as i64, device)?);
                // If no cache key was seen yet, also seed the cursor so a
                // module that ships only `offset` (no cache tensor) still has
                // position info.
                if !module_state.contains_key(ATTN_POS_KEY) {
                    write_attention_cursor(
                        module_state,
                        AttentionCursor {
                            pos: off_val,
                            len: off_val,
                            head: 0,
                        },
                        device,
                    )?;
                }
            }
            "current_end" => {
                // Legacy upstream key: shape[0] is the step count.
                let len = tensor.dim(0).unwrap_or(0);
                module_state.insert("offset".to_string(), Tensor::new(len as i64, device)?);
                if !module_state.contains_key(ATTN_POS_KEY) {
                    write_attention_cursor(
                        module_state,
                        AttentionCursor {
                            pos: len,
                            len,
                            head: 0,
                        },
                        device,
                    )?;
                }
            }
            _ => {
                module_state.insert(tensor_key.to_string(), tensor);
            }
        }
    }

    Ok(result)
}

fn read_offset_scalar(t: &Tensor) -> anyhow::Result<usize> {
    if let Ok(v) = t.to_scalar::<i64>() {
        return Ok(v.max(0) as usize);
    }
    if let Ok(v) = t.to_scalar::<u32>() {
        return Ok(v as usize);
    }
    // Tensor of shape [1], int64.
    if let Ok(vec) = t.flatten_all().and_then(|x| x.to_vec1::<i64>()) {
        if let Some(&first) = vec.first() {
            return Ok(first.max(0) as usize);
        }
    }
    anyhow::bail!("unsupported offset tensor dtype/shape: {:?}", t.dims())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_states() {
        let state = init_states(1, 100);
        assert!(state.is_empty());
    }

    #[test]
    fn test_get_or_create_state() {
        let mut state = init_states(1, 100);
        let module_state = get_or_create_state(&mut state, "test_module");
        assert!(module_state.is_empty());
        assert!(state.contains_key("test_module"));
    }

    #[test]
    fn test_offset_operations() -> Result<()> {
        let mut state = init_states(1, 100);

        // Initially offset is 0
        assert_eq!(get_offset(&state, "test"), 0);

        // Set offset
        set_offset(&mut state, "test", 42)?;
        assert_eq!(get_offset(&state, "test"), 42);

        Ok(())
    }
}
