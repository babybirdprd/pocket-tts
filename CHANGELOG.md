# Changelog

All notable changes to the Rust/Candle port of Pocket TTS are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

## [0.7.0] - 2026-05-06

Catch-up release: the Rust port now reaches structural parity with
[`kyutai-labs/pocket-tts`](https://github.com/kyutai-labs/pocket-tts) v2.1.0.
Italian end-to-end generation has been verified against the upstream
HuggingFace weights.

### Added

- **Multi-language support.** Bundles all 12 upstream language YAMLs
  (`english`, `english_2026-01`, `english_2026-04`, `french_24l`, `german`,
  `german_24l`, `italian`, `italian_24l`, `portuguese`, `portuguese_24l`,
  `spanish`, `spanish_24l`) under `crates/pocket-tts/config/`. Legacy
  `b6369a24.yaml` is retained for backward compatibility.
- **`--language LANG` CLI flag** on `generate` and `serve`, defaulting to
  `english`. Mutually exclusive with `--variant`.
- **Per-language defaults**: when `--text` and `--voice` are omitted,
  `generate` falls back to the language's recommended voice
  (`giovanni`/it, `lola`/es, `juergen`/de, `rafael`/pt, `estelle`/fr,
  `alba` otherwise) and a localized greeting, matching Python's
  `default_parameters.py`.
- **Stdin text input** on `generate`: when `--text` is omitted and stdin
  is piped (not a TTY), text is read from stdin.
- **`pocket-tts export-voice`** CLI subcommand. Encodes a `.wav` and
  writes the resulting flow-LM state in upstream Python's
  `export_model_state` layout, so files round-trip with both
  implementations.
- **Voice state import** for upstream `export_model_state` files
  (`{module_name}/{tensor_key}` keys). All HF predefined voice
  embeddings under
  `kyutai/pocket-tts-without-voice-cloning/languages/<lang>/embeddings/`
  now load via `TTSModel::get_voice_state_from_prompt_file`.
- **Voice state export** library API
  (`pocket_tts::voice_state::export_model_state_to_file`), used by the
  new CLI subcommand and verified by an export -> import round-trip
  test.
- **HF cache short-circuit** in `weights::download_if_necessary`. When a
  file is already present in the standard HF Hub cache layout
  (`$HF_HOME/hub/models--{owner}--{repo}/snapshots/{rev}/{file}`), the
  function returns the cached path without contacting the network. This
  also unblocks environments where the in-process TLS stack cannot
  reach huggingface.co.
- **`bos_before_voice` parameter** in `FlowLMModel`, loaded from
  safetensors when the YAML's `flow_lm.insert_bos_before_voice` is true,
  and prepended to voice conditioning in
  `TTSModel::get_voice_state_from_tensor`. Mirrors Python's
  `tts_model.py:893-894`.
- **Mimi `inner_dim` / `outer_dim`** plumbing through
  `MimiModel::new_with_dims`,
  `ConvDownsample1d::new_with_dims`, and
  `ConvTrUpsample1d::new_with_dims`. Required to load the new English
  / Italian / etc. weights, where the downsample reduces 512 -> 32.
- **Predefined voices expanded** from 8 to 26 names in
  `pocket-tts-cli`, including the per-language defaults.
- **`#[serde(deny_unknown_fields)]`** on every config struct, so any
  future schema drift fails loud instead of silently dropping fields.

### Changed

- **Default model is now upstream English.** `pocket-tts generate` with
  no flags now downloads
  `kyutai/pocket-tts/languages/english/model.safetensors` instead of
  `tts_b6369a24.safetensors`. Previous behavior is reachable via
  `--variant b6369a24`.
- **`pad_with_spaces_for_short_inputs` defaults to false** (matching
  Python's Pydantic default). Rust previously always padded short
  inputs unconditionally; this brings it into parity with upstream.
  Only `english_2026-01.yaml` opts back into padding.
- **`speaker_proj_weight` shape** is now
  `(d_model, inner_dim or seanet.dimension)` to match Python's
  `tts_model.py:146`. For legacy `b6369a24` (no `inner_dim`) this
  resolves to the prior `(1024, 512)`; for new languages it shrinks to
  `(1024, 32)`.

### Tests

- 47 lib tests + 11 CLI lib tests + 3 integration tests cover the
  new functionality. The integration tests in
  `crates/pocket-tts/tests/italian_parity.rs` self-skip without
  `HF_TOKEN`.
- Verified end-to-end: `cargo test -p pocket-tts --test italian_parity`
  produces 3.84 s of Italian speech with peak amplitude 0.348
  (non-silent, non-clipping, no NaN/Inf).

[0.7.0]: https://github.com/dusterbloom/pocket-tts/compare/v0.6.2...v0.7.0
