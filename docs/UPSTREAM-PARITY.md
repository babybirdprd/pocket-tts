# Upstream parity — kyutai-labs/pocket-tts @ main (d108410, 2026-07-16)

The vendored python-reference was refreshed from v1.0.1 to upstream main.
This file tracks every upstream behavior change since v1.0.1 and its status
in the Rust port. Reference commit: d108410 ("Default the English model's
temperature to 0.3 (#223)").

## Status legend

- DONE: ported and verified
- TODO: to port
- HAD IT: the Rust port already had the behavior (often ahead of upstream)
- N/A: Python-only mechanics with no Rust counterpart (justified inline)

## Model selection and configs

| Item | Upstream | Status |
| --- | --- | --- |
| 12 language configs (english*, french_24l, german*, italian*, portuguese*, spanish*) | v2.0/v2.1 | DONE: mirrored into crates/pocket-tts/config (b6369a24.yaml kept as legacy alias of english_2026-01) |
| `language=` selection + `config=` local yaml path, mutually exclusive | v2.0 | DONE: --language/--config on generate/serve/export-voice (--variant kept as hidden alias); lib accepts a .yaml path anywhere a variant name goes |
| Default language "english" (= english_2026-04 weights) | v2.0 | DONE: defaults::DEFAULT_VARIANT = "english" |
| "french" refused with pointer to french_24l | v2.0 | DONE (resolve_model_spec) |
| `default_temperature` per config, `temp=None` resolves from it | #223 | DONE (config.rs, load paths take `Option<f32>`) |
| Helpful config-not-found error listing available languages | v2.0 | DONE (find_config_path) |
| `insert_bos_before_voice`, `inner_dim`/`outer_dim`, `pad_with_spaces_for_short_inputs`, `remove_semicolons`, `model_recommended_frames_after_eos` | v2.0 | HAD IT |

## Voices

| Item | Upstream | Status |
| --- | --- | --- |
| `DEFAULT_VOICE_FOR_LANGUAGE` (estelle fr, giovanni it, lola es, juergen de, rafael pt; alba fallback), substring match on origin | v2.1 | DONE (voice.rs `default_voice_for`) |
| `get_predefined_voice`: languages/{origin}/embeddings/{name}.safetensors @e041936c | v2.1 | DONE (voice.rs `stock_voice_url`; b6369a24 keeps legacy root embeddings) |
| `origin` stored on the model (config stem) | v2.1 | DONE (TTSModel.origin) |
| Any bare name resolves as predefined voice (not a fixed list) | v2.1 | DONE (voice.rs `is_bare_voice_name`) |
| `_ORIGINS_OF_PREDEFINED_VOICES` (cloning source WAVs incl. anna, vera, charles, ... and the language voices) | v2.1 | N/A as a list to embed (cloning-by-URL goes through the same resolver) |
| `has_voice_cloning` + VOICE_CLONING_UNSUPPORTED refusal | v1.x/v2.1 | DONE: the without-voice-cloning checkpoint ships a ZEROED Mimi encoder — the port used to silently condition on silence; now the flag is set on the weights fallback and get_conditioning refuses loudly with the upstream-style message. Streaming tests moved to the alba embedding accordingly |
| Default demo text per language | v2.0 | DONE (generate.rs) |

## Generation

| Item | Upstream | Status |
| --- | --- | --- |
| `max_tokens` parameter (CLI + API) threaded to the splitter | v2.0 | DONE: GenerateOptions + --max-tokens |
| Token-boundary splitter (sentence tokens of ".!...?", fallback ",;:") | v2.0 | DONE: split_into_best_sentences_with is the upstream algorithm (encode/decode ids). Caught in testing: the Metaspace-only decoder left byte-fallback tokens as literal "<0x21>" (French typography puts a space before "!"/":"/"?"), so the decode/re-encode round trip made the model spell them out; fixed with the sentencepiece decoder chain (Replace + ByteFallback + Fuse + Strip), regression test in tests/splitter_tests.rs |
| Comma sub-split of oversized sentences (#143) | v2.0 | HAD IT (plus word-batch last resort upstream lacks; keep) |
| Warning when a chunk still exceeds max_tokens | v2.0 | DONE: warns, then word-batches (kept deviation: a chunk can never overflow) |
| `_estimate_max_gen_len`: tokens/3.0 + 2 s padding, ceil * frame_rate | v2.1 | DONE (frame_rate stored on the model) |
| Mimi decoder state sized max_gen_len * steps_per_latent; increment = encoder_frame_rate/frame_rate | v2.1 | N/A: the port's states are lazily allocated ring/growing buffers (init_states is empty); nothing is preallocated to a fixed 1000 |
| Voice state init with sequence_length = prompt length, `_expand_kv_cache` to required_len before generation | v2.1 | N/A: same reason; imported prompt-length caches already grow on demand (proven by Estelle KV embeddings) |
| `frames_after_eos` explicit override wins over guess; model_recommended fallback | v1.0.3/v2.0 | DONE: explicit override > model recommendation > heuristic (the CLI flag was previously dead) |
| stdin input (`--text -`) | v1.1 | DONE (resolve_text) |

## Voice state export/import

| Item | Upstream | Status |
| --- | --- | --- |
| `export_model_state` / `_import_model_state` ("module/key" flat safetensors; legacy current_end -> offset) | v1.1 | DONE: export_model_state writes the upstream format; import was already get_voice_state_from_kyutai_embedding; round-trip test in tests/export_import_tests.rs |
| `export-voice` CLI command (audio or directory -> .safetensors) | v1.1 | DONE (commands/export_voice.rs) |
| `get_state_for_audio_prompt` accepts .safetensors path/URL directly | v1.1 | HAD IT (resolve_file_voice) |

## Quantization

| Item | Upstream | Status |
| --- | --- | --- |
| `quantize=True`: dynamic int8 on FlowLM attention (in/out proj) + FFN (linear1/2); flow net + Mimi stay f32; ~-48% mem, ~+27% x86, WER unchanged | v2.0 | DONE: real q8_0 QMatMul (MaybeQuantLinear); the simulated-f32 placeholder was deleted; flow_net group refuses loudly (upstream does not recommend it) |
| Quantized init_state device/dtype handling | v2.1 | Covered by the QMatMul design (activations stay f32) |
| serve --quantize | v2.0 | DONE (--quantize on generate and serve, --quantized alias; no longer feature-gated) |

## Audio I/O

| Item | Upstream | Status |
| --- | --- | --- |
| Multichannel WAV downmix to mono on read | v2.1 | DONE (read_wav_internal averages channels) |
| Non-WAV formats (mp3/flac/ogg) for cloning input, optional dependency | v2.1 | DONE: read_audio + `audio-formats` feature (symphonia), mirroring upstream's optional soundfile |

## Server

| Item | Upstream | Status |
| --- | --- | --- |
| Default voice per language when request has none | v2.1 | DONE (default resolved at startup) |
| voice_url accepts bare predefined names | v2.1 | DONE (resolve path); request validation follows resolver |
| Upload preserves file extension for format detection | v2.1 | PARTIAL: the port's upload path base64-wraps as WAV; non-WAV uploads need the resolver to learn data: mime routing (tracked) |
| index.html: DEFAULT_TEXT_PROMPT placeholder, full voice list, audio/* accept, relative /tts endpoint, latencyHint | v2.0/2.1 | N/A: the port ships its own React UI; the /tts multipart endpoint parity is what matters and it exists |
| Voice state LRU cache | v1.1 | HAD IT |

## Internals upstream changed where the port was already there or diverges deliberately

- Unified streaming attention (Mimi variant deleted, context window in the
  one implementation): the Rust port already has a single attention (the
  `kind` parameter is vestigial); its SDPA path is custom and memory-efficient.
- weights_loading.py rewrites (weight-norm folding, in_proj_weight renames,
  wavlm skips): the Rust loader reads the published safetensors names
  directly and loads every shipped checkpoint; the Python renames chase its
  own module naming. N/A.
- Device placement fixes in conv/transformer states: candle states are
  allocated on the model device already. N/A.
- `_module_absolute_name` moved out of init_states: Python identity
  bookkeeping. N/A.
- `POCKET_TTS_SAVE_WEIGHTS`, `DEBUG_MIMI` dumps: debug helpers. N/A unless
  needed.
- `get_default_tokenizer`: convenience for Python scripts. N/A.
- x.com plug log line: no.

## Dependency refresh (post-parity, same session)

- candle 0.9.1 -> 0.11.0 (workspace pin; zero source changes needed). The
  old `metal 0.27`/`block 0.1.6` chain is replaced by objc2-metal 0.3; the
  future-incompat warnings are gone.
- intel-mkl-src is now optional and x86_64-only (dep of the `mkl` feature):
  Apple Silicon builds drop the ocipkg/getset/proc-macro-error2 chain.
- New `accelerate` feature (candle Accelerate BLAS) for the macOS CPU path.
- `cargo update` across the tree.
- The two cloning parity tests were failing BEFORE the bump too (proven by
  a worktree A/B on the pre-session commit): root cause is the zeroed
  encoder above, not candle. They now skip with an explicit message when
  has_voice_cloning is false.
- q8_0 quantized generation verified working on Metal with candle 0.11.
