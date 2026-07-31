//! Print what the splitter feeds the model for a given text.
//!
//! cargo run --example split_debug --no-default-features -- french_24l "text"

use pocket_tts::TTSModel;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let variant = args.next().unwrap_or_else(|| "french_24l".to_string());
    let text = args
        .next()
        .unwrap_or_else(|| "Bonjour le monde. Je suis le TTS de poche.".to_string());

    let model = TTSModel::load(&variant)?;
    println!("input: {text:?}");
    for (i, chunk) in model.split_into_best_sentences(&text).iter().enumerate() {
        let tokens = model.conditioner.count_tokens(chunk).unwrap_or(0);
        println!("chunk {i} ({tokens} tokens): {chunk:?}");
    }
    Ok(())
}
