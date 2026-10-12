//! Print what the binding actually produces, for checking against Python.
//!
//! Assembly's Python side is the reference implementation, and the surest way to
//! find a disagreement is to print both. This example exists so a binding detail
//! -- a vector's width, a prompt's exact rendering, an embedding's magnitude --
//! can be compared by eye without writing a test that hard-codes a number nobody
//! has verified.

use std::env;
use std::path::PathBuf;

use assembly_core::engine::{Engine, EngineOptions, LoadOptions, Messages};
use llama_engine::LlamaEngine;

fn main() {
    let Some(embed) = var("ASSEMBLY_EMBED_MODEL") else {
        eprintln!("set ASSEMBLY_EMBED_MODEL to an embedding model");
        std::process::exit(1);
    };

    let engine = LlamaEngine::new();

    if let Some(chat) = var("ASSEMBLY_CHAT_MODEL") {
        println!("== chat ==");
        let model = engine
            .load(&chat, &LoadOptions::generate().n_ctx(2048))
            .expect("load chat model");

        let messages = Messages::from_prompt("Reply with the single word: ready");

        println!("-- streamed --");
        for chunk in engine
            .chat_stream(
                model.as_ref(),
                &messages,
                &EngineOptions::new().max_tokens(32).seed(1),
            )
            .expect("stream")
        {
            print!("{}", chunk.text);
            if chunk.done {
                println!(" [done]");
            }
        }

        println!("-- tokens --");
        let tokens = engine
            .tokenize(model.as_ref(), "hello there", false)
            .expect("tokenize");
        println!("{:?}", tokens);
    }

    println!("== embed ==");
    let model = engine
        .load(&embed, &LoadOptions::embed())
        .expect("load embedding model");
    for phrase in ["writing rust", "baking bread"] {
        let vector = engine.embed(model.as_ref(), phrase).expect("embed");
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!(
            "{phrase:?}: dim={} norm={:.4} first={:?}",
            vector.len(),
            norm,
            &vector[..vector.len().min(4)]
        );
    }
}

fn var(name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(env::var_os(name)?);
    assert!(path.is_file(), "{name} is not a file: {}", path.display());
    Some(path)
}
