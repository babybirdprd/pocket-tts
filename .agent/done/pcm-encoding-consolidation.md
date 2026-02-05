# Consolidate PCM Encoding in Core Audio Module

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

PLANS.md for this plan lives at `.agent/PLANS.md` from the repository root. This ExecPlan must be maintained in accordance with that file.

## Purpose / Big Picture

After this change, all paths that turn model audio into 16-bit PCM bytes (CLI streaming, HTTP streaming, WAV writing, and the WASM base64 path) will use a single helper in the core audio module. This eliminates drift, keeps the clamp and interleave rules identical everywhere, and makes it safe to change the sample conversion logic in one place. You can see it working by running the unit test that checks the exact PCM byte output and by building the CLI/server without any duplicated conversion loops.

## Progress

- [x] (2026-02-05 01:05Z) Milestone 1: Added PCM conversion helpers and unit test in `crates/pocket-tts/src/audio.rs`. Ran `cargo test -p pocket-tts --release audio::tests::test_pcm_i16_le_bytes_clamp_and_interleave`.
- [x] (2026-02-05 01:40Z) Milestone 2: Migrated CLI streaming, HTTP streaming, WAV writing, and WASM base64 generation to the helper. Ran `cargo check -p pocket-tts-cli --release` and `cargo check -p pocket-tts --release --features wasm`.

## Surprises & Discoveries

- Cargo commands failed without escalation due to `D:\RustBuilds\release\.cargo-lock` permission. Re-running with elevated permissions succeeded.
- The first release test run timed out; re-running with a longer timeout completed successfully.

## Decision Log

- Decision: Centralize float-to-PCM conversion in `crates/pocket-tts/src/audio.rs` and reuse it from CLI, server, and WASM. Rationale: the same clamp and scale logic is duplicated in multiple call sites and is easy to let drift. Date/Author: 2026-02-04, Codex.
- Decision: Combine milestones into a single commit because the user requested “commit everything.” Rationale: explicit user direction overrides the plan’s suggested per-milestone commits. Date/Author: 2026-02-05, Codex.

## Outcomes & Retrospective

Completed consolidation of PCM encoding into a single helper with a unit test that locks in clamp/interleave behavior. CLI streaming, HTTP streaming, WAV writing, and WASM base64 generation now share the same conversion path. Release-mode unit test and both CLI and WASM release build checks completed successfully. No full test suite was run beyond the targeted test and cargo check.

## Context and Orientation

`crates/pocket-tts/src/audio.rs` owns WAV I/O and currently performs the float-to-i16 clamp and scale inside `write_wav_to_writer`. The CLI streaming path in `crates/pocket-tts-cli/src/commands/generate.rs` and the HTTP streaming path in `crates/pocket-tts-cli/src/server/handlers.rs` manually repeat the same clamp and interleave loops to produce raw PCM bytes. The WASM path in `crates/pocket-tts/src/wasm.rs` repeats the clamp and scale again when building base64 WAV data. In this repo, audio tensors are shaped as `[channels, samples]` and PCM is 16-bit signed little-endian samples, interleaved by channel.

## Plan of Work

Add a small set of helpers in `crates/pocket-tts/src/audio.rs` that convert either a `Tensor` or a slice of `f32` samples into interleaved 16-bit PCM bytes with the existing clamp rule. Update `write_wav_to_writer` to use the helper so the WAV and streaming paths share the same conversion. Replace the manual loops in CLI streaming and HTTP streaming with calls to the helper. Update the WASM base64 generator to reuse the same helper for mono sample slices. Add a focused unit test in `audio.rs` that asserts the exact byte sequence for a tiny two-channel tensor so the clamp and interleaving rules are locked in.

## Concrete Steps

Run all commands from `D:\pocket-tts-candle`.

After adding the helper and test, run the unit test in release mode:

    D:\pocket-tts-candle> cargo test -p pocket-tts --release audio::tests::test_pcm_i16_le_bytes_clamp_and_interleave

Expected output includes a single passing test, for example:

    test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

After updating CLI and server callers, verify the CLI crate still builds:

    D:\pocket-tts-candle> cargo check -p pocket-tts-cli --release

Expected output ends with:

    Finished release [optimized] target(s) in ...

If the WASM target is installed, also check the wasm feature build:

    D:\pocket-tts-candle> cargo check -p pocket-tts --release --features wasm

If the target is not installed, skip this step and note it in Progress.

## Validation and Acceptance

Acceptance means there is a single helper in `crates/pocket-tts/src/audio.rs` that performs clamp and interleaving, `write_wav_to_writer` uses it, and the streaming paths in CLI and HTTP server no longer contain their own clamp loops. The new unit test must fail before the helper exists and pass after it is implemented. Build checks for CLI (and WASM if available) must succeed.

Milestone 1 verification workflow:

1. Tests to write: In `crates/pocket-tts/src/audio.rs`, add `test_pcm_i16_le_bytes_clamp_and_interleave` under `#[cfg(test)]`. Build a two-channel tensor with samples `[-1.0, 0.0, 1.0]` in channel 0 and `[0.5, -0.5, 2.0]` in channel 1, call the new helper, and assert the resulting interleaved i16 sequence equals `[-32767, 16383, 0, -16383, 32767, 32767]` when decoded from the bytes.
2. Implementation: Add a public helper `pub fn pcm_i16_le_bytes(audio: &Tensor) -> anyhow::Result<Vec<u8>>` in `crates/pocket-tts/src/audio.rs` plus any private helpers needed for clamping and interleaving. Use the existing clamp rule and scaling factor `32767.0`.
3. Verification: Run `cargo test -p pocket-tts --release audio::tests::test_pcm_i16_le_bytes_clamp_and_interleave` and confirm it passes.
4. Commit: Commit with message `Milestone 1: Add PCM conversion helper and test`.

Milestone 2 verification workflow:

1. Tests to write: No new tests are required beyond the helper unit test; this milestone reuses that test and compile checks because it is a caller migration.
2. Implementation: Update `write_wav_to_writer` in `crates/pocket-tts/src/audio.rs`, `run_streaming` in `crates/pocket-tts-cli/src/commands/generate.rs`, the streaming loop in `crates/pocket-tts-cli/src/server/handlers.rs`, and `generate_wav_base64` in `crates/pocket-tts/src/wasm.rs` to call the helper instead of duplicating clamp logic. For WASM, add a small internal helper in `audio.rs` that accepts a `&[f32]` mono slice and returns PCM bytes, and call it from `generate_wav_base64`.
3. Verification: Re-run the unit test from Milestone 1 and run `cargo check -p pocket-tts-cli --release`. If the WASM target is available, run `cargo check -p pocket-tts --release --features wasm`.
4. Commit: Commit with message `Milestone 2: Use shared PCM helper across CLI/server/WASM`.

## Idempotence and Recovery

These changes are additive and can be re-run safely. If a step fails, fix the error and re-run the same test or build command. If you need to roll back, use version control to revert the last milestone commit and reapply the changes.

## Artifacts and Notes

Example expected PCM byte order for the unit test:

    Interleaved i16 samples: [-32767, 16383, 0, -16383, 32767, 32767]

## Interfaces and Dependencies

Add the following function to `crates/pocket-tts/src/audio.rs`:

    pub fn pcm_i16_le_bytes(audio: &Tensor) -> anyhow::Result<Vec<u8>>

For WASM, add an internal helper in the same file that accepts a mono `&[f32]` and returns PCM bytes; keep it `pub(crate)` so only the crate can use it. The helper must clamp each sample to `[-1.0, 1.0]`, multiply by `32767.0`, cast to `i16`, and emit little-endian bytes. The interleaving order for multi-channel tensors must remain `channel0[0], channel1[0], ..., channelN[0], channel0[1], ...`.

Plan update (2026-02-05): Marked milestones complete, recorded test/build commands and permission/timeouts, and noted the single combined commit per user request.
