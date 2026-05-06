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

/// Export a `ModelState` to a safetensors file using the upstream Python
/// `export_model_state` layout (keys of the form `{module_name}/{tensor_key}`).
///
/// `module_prefix` is stripped from each module name with a `.` separator,
/// inverse of [`import_model_state_from_file`]'s prepending. Pass `"flow_lm"`
/// to produce files compatible with the predefined-voice format upstream
/// uses for `kyutai/pocket-tts-without-voice-cloning/.../embeddings/*.safetensors`.
///
/// Tensor key remappings:
/// - Rust's split `k_buf` / `v_buf` `(B, H, T, D)` are stacked back into a
///   single `cache` tensor `[2, B, T, H, D]` with the appropriate transpose,
///   so import/export round-trips bit-exactly.
/// - `pos`, `l`, `head` cursor tensors are dropped (they are derived from
///   `cache.shape[2]` and `offset` on import). This keeps the export format
///   identical to upstream's.
/// - All other tensors (e.g. `offset`) pass through unchanged.
pub fn export_model_state_to_file<P: AsRef<Path>>(
    state: &ModelState,
    path: P,
    module_prefix: &str,
) -> anyhow::Result<()> {
    let to_store = build_export_tensors(state, module_prefix)?;
    safetensors_save_to_path(&to_store, path.as_ref())?;
    Ok(())
}

fn build_export_tensors(
    state: &ModelState,
    module_prefix: &str,
) -> anyhow::Result<HashMap<String, Tensor>> {
    let prefix_dot = if module_prefix.is_empty() {
        String::new()
    } else {
        format!("{}.", module_prefix)
    };
    let mut to_store: HashMap<String, Tensor> = HashMap::new();

    for (full_module_name, module_state) in state.iter() {
        // Strip the prefix so on-disk module names are FlowLM-relative.
        let module_name = match full_module_name.strip_prefix(&prefix_dot) {
            Some(rest) => rest.to_string(),
            None if !prefix_dot.is_empty() => continue, // skip modules outside the prefix
            None => full_module_name.clone(),
        };

        // Recombine k_buf + v_buf into a single `cache` tensor when both are present.
        match (
            module_state.get(ATTN_K_BUF_KEY),
            module_state.get(ATTN_V_BUF_KEY),
        ) {
            (Some(k_buf), Some(v_buf)) => {
                // Rust:  (B, H, T, D)  ->  (B, T, H, D)
                let k_btshd = k_buf.transpose(1, 2)?.contiguous()?;
                let v_btshd = v_buf.transpose(1, 2)?.contiguous()?;
                // stack along new axis 0 -> [2, B, T, H, D]
                let cache = Tensor::stack(&[&k_btshd, &v_btshd], 0)?;
                to_store.insert(format!("{}/cache", module_name), cache);
            }
            (Some(_), None) | (None, Some(_)) => {
                anyhow::bail!(
                    "module '{}' has only one of k_buf/v_buf; refusing to export inconsistent state",
                    full_module_name
                );
            }
            (None, None) => {}
        }

        // Pass through other tensors except the cursor scratch keys (which
        // are reconstructible from cache.shape[2] / offset on import).
        for (k, v) in module_state.iter() {
            if matches!(
                k.as_str(),
                ATTN_K_BUF_KEY | ATTN_V_BUF_KEY | ATTN_POS_KEY | ATTN_LEN_KEY | ATTN_HEAD_KEY
            ) {
                continue;
            }
            to_store.insert(format!("{}/{}", module_name, k), v.clone());
        }
    }

    Ok(to_store)
}

#[cfg(not(target_arch = "wasm32"))]
fn safetensors_save_to_path(map: &HashMap<String, Tensor>, path: &Path) -> anyhow::Result<()> {
    candle_core::safetensors::save(map, path)?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn safetensors_save_to_path(_map: &HashMap<String, Tensor>, _path: &Path) -> anyhow::Result<()> {
    anyhow::bail!("safetensors save not supported on wasm32 target")
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

    #[test]
    fn test_export_import_roundtrip() -> anyhow::Result<()> {
        // Build a minimal state matching the upstream voice file shape:
        //   transformer.layers.0.self_attn -> { k_buf, v_buf, offset, cursor }
        //   transformer.layers.1.self_attn -> { k_buf, v_buf, offset, cursor }
        let device = candle_core::Device::Cpu;
        let mut state = init_states(1, 16);

        for layer_idx in 0..2 {
            let k_data: Vec<f32> = (0..(1 * 16 * 8 * 64)).map(|x| x as f32 * 0.001).collect();
            let v_data: Vec<f32> = (0..(1 * 16 * 8 * 64)).map(|x| -(x as f32) * 0.001).collect();
            let k_buf = Tensor::from_vec(k_data, (1, 16, 8, 64), &device)?;
            let v_buf = Tensor::from_vec(v_data, (1, 16, 8, 64), &device)?;
            let module_name =
                format!("flow_lm.transformer.layers.{}.self_attn", layer_idx);
            let m = state.entry(module_name).or_default();
            m.insert(ATTN_K_BUF_KEY.to_string(), k_buf);
            m.insert(ATTN_V_BUF_KEY.to_string(), v_buf);
            m.insert("offset".to_string(), Tensor::new(8_i64, &device)?);
            write_attention_cursor(
                m,
                AttentionCursor {
                    pos: 8,
                    len: 8,
                    head: 0,
                },
                &device,
            )?;
        }

        let tmp = tempfile_path("pocket_tts_voice_state_roundtrip.safetensors");
        export_model_state_to_file(&state, &tmp, "flow_lm")?;

        let imported = import_model_state_from_file(&tmp, &device, "flow_lm")?;

        // Cleanup
        let _ = std::fs::remove_file(&tmp);

        // Confirm both layers came back with k_buf, v_buf, offset, cursor.
        for layer_idx in 0..2 {
            let module_name =
                format!("flow_lm.transformer.layers.{}.self_attn", layer_idx);
            let imported_module = imported
                .get(&module_name)
                .unwrap_or_else(|| panic!("missing module {} after import", module_name));
            assert!(
                imported_module.contains_key(ATTN_K_BUF_KEY),
                "missing k_buf"
            );
            assert!(
                imported_module.contains_key(ATTN_V_BUF_KEY),
                "missing v_buf"
            );
            assert!(imported_module.contains_key("offset"), "missing offset");
            assert!(imported_module.contains_key(ATTN_POS_KEY), "missing pos");

            // Dimensions must match Rust's layout.
            let k = imported_module.get(ATTN_K_BUF_KEY).unwrap();
            assert_eq!(k.dims(), &[1, 16, 8, 64]);
            let v = imported_module.get(ATTN_V_BUF_KEY).unwrap();
            assert_eq!(v.dims(), &[1, 16, 8, 64]);

            // Numerical equality with the originals.
            let orig_k = state
                .get(&module_name)
                .unwrap()
                .get(ATTN_K_BUF_KEY)
                .unwrap();
            let max_diff = (k - orig_k)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(max_diff < 1e-6, "k_buf round-trip diff too large: {}", max_diff);
        }
        Ok(())
    }

    fn tempfile_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("pid{}-{}", std::process::id(), name))
    }
}
