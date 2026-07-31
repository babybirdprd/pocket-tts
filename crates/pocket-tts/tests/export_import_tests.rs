//! Round-trip test for upstream-format voice-state export/import.
//!
//! Needs the french_24l checkpoint and the Estelle embedding (downloads
//! them into the HF cache if absent), so it is ignored by default:
//! `cargo test -p pocket-tts --test export_import_tests -- --ignored`

use pocket_tts::{TTSModel, export_model_state};

const ESTELLE: &str = "hf://kyutai/pocket-tts-without-voice-cloning/languages/french_24l/embeddings/estelle.safetensors@e041936c75475d350b405bc870bcf7c22da4e9e6";

#[test]
#[ignore = "downloads model weights"]
fn export_import_roundtrip_matches() {
    let model = TTSModel::load("french_24l").expect("load french_24l");
    let estelle_file =
        pocket_tts::weights::download_if_necessary(ESTELLE).expect("download estelle");
    let state = model
        .get_voice_state_from_kyutai_embedding(&estelle_file)
        .expect("load estelle state");

    let tmp = std::env::temp_dir().join("pocket-tts-export-roundtrip.safetensors");
    export_model_state(&state, &tmp).expect("export");
    let reloaded = model
        .get_voice_state_from_kyutai_embedding(&tmp)
        .expect("reimport exported state");

    assert_eq!(state.len(), reloaded.len(), "module set changed");
    for (name, module) in &state {
        let other = reloaded
            .get(name)
            .unwrap_or_else(|| panic!("missing module {name}"));

        let cursor = pocket_tts::voice_state::read_attention_cursor(module);
        let other_cursor = pocket_tts::voice_state::read_attention_cursor(other);
        assert_eq!(cursor.pos, other_cursor.pos, "{name}: pos changed");
        assert_eq!(cursor.len, other_cursor.len, "{name}: len changed");

        for key in ["k_buf", "v_buf"] {
            let a = module.get(key).unwrap_or_else(|| panic!("{name}: no {key}"));
            let b = other.get(key).unwrap_or_else(|| panic!("{name}: no {key}"));
            // The reloaded buffer holds exactly the valid slice.
            let a = a.narrow(2, 0, cursor.len).expect("narrow");
            let diff = (a - b)
                .and_then(|d| d.abs())
                .and_then(|d| d.max_all())
                .and_then(|d| d.to_scalar::<f32>())
                .expect("diff");
            assert_eq!(diff, 0.0, "{name}/{key}: cache bytes changed");
        }
    }
    std::fs::remove_file(&tmp).ok();
}
