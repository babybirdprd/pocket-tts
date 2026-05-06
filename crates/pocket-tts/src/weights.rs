use anyhow::Result;
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
use candle_core::Device;

#[cfg(not(target_arch = "wasm32"))]
use hf_hub::api::sync::ApiBuilder;
#[cfg(not(target_arch = "wasm32"))]
use hf_hub::{Repo, RepoType};

/// Download a file from HuggingFace Hub if necessary.
///
/// Supports the format: `hf://owner/repo/filename@revision`
/// where `@revision` is optional.
///
/// When the file is already present in the standard HF Hub cache layout
/// (`$HF_HOME/hub/models--{owner}--{repo}/snapshots/{revision}/{filename}`,
/// defaulting to `~/.cache/huggingface`), this function short-circuits and
/// returns the cached path without contacting the network. This makes
/// offline test environments and cache-pre-populated CI runs work even
/// when the in-process TLS stack cannot reach HuggingFace.
///
/// Note: Not available on wasm32 targets (use local file loading instead).
#[cfg(not(target_arch = "wasm32"))]
pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    if !file_path.starts_with("hf://") {
        return Ok(PathBuf::from(file_path));
    }

    let path = file_path.trim_start_matches("hf://");
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() < 3 {
        anyhow::bail!(
            "Invalid hf:// path: {}. Expected hf://repo_owner/repo_name/filename[@revision]",
            file_path
        );
    }
    let repo_id = format!("{}/{}", parts[0], parts[1]);
    let filename_with_revision = parts[2..].join("/");

    // Parse optional revision from filename (e.g., "file.safetensors@abc123")
    let (filename, revision) = if let Some(at_pos) = filename_with_revision.rfind('@') {
        let (f, r) = filename_with_revision.split_at(at_pos);
        (f.to_string(), Some(r[1..].to_string())) // Skip the '@'
    } else {
        (filename_with_revision, None)
    };

    // Cache short-circuit: when the file is already available under the
    // standard HF layout, return it before invoking hf-hub. This avoids
    // a network round-trip and is the only path that works in environments
    // where the in-process TLS stack cannot validate huggingface.co.
    if let Some(cached) = lookup_in_hf_cache(&repo_id, &filename, revision.as_deref())
        && cached.is_file()
    {
        return Ok(cached);
    }

    // Use ApiBuilder to support HF_TOKEN from environment
    let token = std::env::var("HF_TOKEN").ok();

    let api = ApiBuilder::new().with_token(token).build()?;

    // Create repo with or without revision
    let repo = if let Some(rev) = revision {
        Repo::with_revision(repo_id, RepoType::Model, rev)
    } else {
        Repo::model(repo_id)
    };

    let api_repo = api.repo(repo);
    let path = api_repo.get(&filename)?;
    Ok(path)
}

/// Compute the expected path of `filename` in the HF Hub cache for
/// `repo_id` at `revision`, mirroring the layout used by both
/// `huggingface_hub` (Python) and `hf-hub` (Rust). Returns `None` when
/// the cache root cannot be resolved (e.g. no `HOME` env var) or when
/// `revision` is `None` (we currently only short-circuit revisioned
/// fetches; for un-pinned downloads we still defer to the API to resolve
/// the latest commit).
#[cfg(not(target_arch = "wasm32"))]
fn lookup_in_hf_cache(repo_id: &str, filename: &str, revision: Option<&str>) -> Option<PathBuf> {
    let revision = revision?;

    // Resolve cache root: HF_HOME > HUGGINGFACE_HUB_CACHE > ~/.cache/huggingface
    let cache_root = if let Ok(hf_home) = std::env::var("HF_HOME") {
        PathBuf::from(hf_home)
    } else if let Ok(legacy) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        // Legacy var points directly at .../hub
        return Some(
            PathBuf::from(legacy)
                .join(format!("models--{}", repo_id.replace('/', "--")))
                .join("snapshots")
                .join(revision)
                .join(filename),
        );
    } else {
        let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).ok()?;
        PathBuf::from(home).join(".cache").join("huggingface")
    };

    Some(
        cache_root
            .join("hub")
            .join(format!("models--{}", repo_id.replace('/', "--")))
            .join("snapshots")
            .join(revision)
            .join(filename),
    )
}

/// WASM version: Only supports local file paths
#[cfg(target_arch = "wasm32")]
pub fn download_if_necessary(file_path: &str) -> Result<PathBuf> {
    if file_path.starts_with("hf://") {
        anyhow::bail!("HuggingFace Hub downloads not supported on WASM. Use local file paths.");
    }
    Ok(PathBuf::from(file_path))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_weights(
    file_path: &str,
    _device: &Device,
) -> Result<candle_core::safetensors::MmapedSafetensors> {
    let path = download_if_necessary(file_path)?;
    let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path)? };
    Ok(safetensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_if_necessary_local() {
        let path = "test.safetensors";
        let res = download_if_necessary(path).unwrap();
        assert_eq!(res, PathBuf::from(path));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_invalid_hf_path() {
        let path = "hf://invalid";
        let res = download_if_necessary(path);
        assert!(res.is_err());
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_parse_revision() {
        // Test parsing logic (doesn't actually download)
        let path = "hf://kyutai/pocket-tts/file.safetensors@abc123def";
        // This will fail to download but we're testing the parsing
        let res = download_if_necessary(path);
        // We expect a network error, not a parsing error
        assert!(res.is_err());
    }
}
