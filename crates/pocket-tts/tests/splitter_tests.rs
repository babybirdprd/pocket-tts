//! Splitter round-trip tests against the real tokenizer.
//!
//! Needs the french_24l checkpoint in the HF cache (downloads it
//! otherwise), so it is ignored by default:
//! `cargo test -p pocket-tts --test splitter_tests -- --ignored`

use pocket_tts::TTSModel;

/// French typography puts a space before "!", ":", "?" — those punctuation
/// marks go through the tokenizer's byte fallback, and a Metaspace-only
/// decoder used to leave them as literal "<0x21>" in the decode/re-encode
/// round trip, which the model then spelled out loud.
#[test]
#[ignore = "downloads model weights"]
fn french_typography_survives_the_split_round_trip() {
    let model = TTSModel::load("french_24l").expect("load french_24l");

    let cases = [
        "Bonjour Leonard ! Le portage upstream est terminé : quantization, \
         voix par langue, et découpe par tokens. J'espère que ma voix te plaît.",
        "Ça va ? Oui ! Très bien : merci.",
        "Bonjour le monde. Je suis le TTS de poche de Kyutai.",
    ];

    for text in cases {
        let chunks = model.split_into_best_sentences(text);
        let rejoined = chunks.join(" ");
        assert_eq!(rejoined, text, "split round trip altered the text");
        for chunk in &chunks {
            assert!(
                !chunk.contains("<0x"),
                "byte-fallback token leaked into chunk: {chunk:?}"
            );
        }
    }
}
