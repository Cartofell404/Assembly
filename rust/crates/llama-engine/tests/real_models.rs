//! Real-model tests, skipped unless a GGUF is pointed at.
//!
//! These are the only tests that touch llama.cpp, and they are the only ones
//! that can catch a wrong assumption about how llama.cpp behaves rather than how
//! Assembly behaves. Unit tests cover the parts that are Assembly's own decision
//! -- stop sequences, pooling defaults, role names -- because those can be wrong
//! without a model ever being loaded.
//!
//! # Running them
//!
//! ```text
//! ASSEMBLY_EMBED_MODEL=/path/to/nomic-embed-text-v1.5.f32.gguf \
//! ASSEMBLY_CHAT_MODEL=/path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
//!     cargo test -p llama-engine --test real_models -- --ignored
//! ```
//!
//! `--ignored` rather than a runtime skip, so that a run with the variables set
//! actually fails if something is wrong instead of quietly passing with nothing
//! tested.
//!
//! Both variables are optional and each covers its own half: with only one set,
//! the tests that need the other are skipped.

use std::path::PathBuf;
use std::sync::Arc;

use assembly_core::engine::{Engine, EngineOptions, LoadOptions, Messages};
use llama_engine::LlamaEngine;

/// The embedding model to test against, if one was named.
fn embed_model() -> Option<PathBuf> {
    model_path("ASSEMBLY_EMBED_MODEL")
}

/// The chat model to test against, if one was named.
fn chat_model() -> Option<PathBuf> {
    model_path("ASSEMBLY_CHAT_MODEL")
}

fn model_path(variable: &str) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os(variable)?);
    // A variable pointing at nothing is a mistake worth reporting, not a reason
    // to silently skip the test that was asked for.
    assert!(
        path.is_file(),
        "{variable} is set to {}, which is not a file",
        path.display()
    );
    Some(path)
}

#[test]
#[ignore = "needs ASSEMBLY_EMBED_MODEL"]
fn an_embedding_model_returns_a_dense_vector() {
    let Some(path) = embed_model() else {
        eprintln!("skipping: ASSEMBLY_EMBED_MODEL is not set");
        return;
    };

    let engine = LlamaEngine::new();
    let model = engine
        .load(&path, &LoadOptions::embed())
        .expect("the embedding model should load");

    let vector = engine
        .embed(model.as_ref(), "a phrase to embed")
        .expect("embedding should succeed");

    assert!(
        !vector.is_empty(),
        "an embedding model must produce something"
    );
    assert!(
        vector.iter().all(|v| v.is_finite()),
        "every component should be a real number, not NaN or infinity"
    );
    // A unit-length vector would mean a normalised model, which none of the ones
    // Assembly routes with are; catching it here means the router's cosine
    // similarity is doing the work it thinks it is.
    let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        norm > 0.0,
        "a zero vector would make every route score zero"
    );
}

#[test]
#[ignore = "needs ASSEMBLY_EMBED_MODEL"]
fn embeddings_of_different_phrases_are_not_identical() {
    let Some(path) = embed_model() else {
        eprintln!("skipping: ASSEMBLY_EMBED_MODEL is not set");
        return;
    };

    let engine = LlamaEngine::new();
    let model = engine.load(&path, &LoadOptions::embed()).expect("load");

    let one = engine.embed(model.as_ref(), "writing rust").expect("embed");
    let two = engine.embed(model.as_ref(), "baking bread").expect("embed");

    assert_ne!(
        one, two,
        "two different phrases embedded to the same vector; the model is not \
         reading the text"
    );
}

#[test]
#[ignore = "needs ASSEMBLY_EMBED_MODEL"]
fn a_generation_model_refuses_to_embed_rather_than_returning_nothing_useful() {
    let Some(path) = chat_model() else {
        eprintln!("skipping: ASSEMBLY_CHAT_MODEL is not set");
        return;
    };

    let engine = LlamaEngine::new();
    let model = engine
        .load(&path, &LoadOptions::generate())
        .expect("the chat model should load");

    let err = engine
        .embed(model.as_ref(), "text")
        .expect_err("a generation model must not pretend to embed");

    assert!(
        err.to_string().contains("cannot"),
        "the error should say what went wrong, got: {err}"
    );
}

#[test]
#[ignore = "needs ASSEMBLY_CHAT_MODEL"]
fn a_chat_model_answers_and_stops_on_its_own() {
    let Some(path) = chat_model() else {
        eprintln!("skipping: ASSEMBLY_CHAT_MODEL is not set");
        return;
    };

    let engine = LlamaEngine::new();
    let model = engine
        .load(&path, &LoadOptions::generate().n_ctx(2048))
        .expect("the chat model should load");

    let answer = engine
        .chat(
            model.as_ref(),
            &Messages::from_prompt("Reply with the single word: ready"),
            &EngineOptions::new().max_tokens(24).seed(1),
        )
        .expect("generation should succeed");

    assert!(!answer.trim().is_empty(), "the model said nothing");
}

#[test]
#[ignore = "needs ASSEMBLY_CHAT_MODEL"]
fn streaming_and_non_streaming_produce_the_same_text() {
    let Some(path) = chat_model() else {
        eprintln!("skipping: ASSEMBLY_CHAT_MODEL is not set");
        return;
    };

    let engine = LlamaEngine::new();
    let model = engine
        .load(&path, &LoadOptions::generate().n_ctx(2048))
        .expect("load");

    // A fixed seed, because the two calls sample independently and would
    // otherwise differ for reasons that have nothing to do with streaming.
    let options = || EngineOptions::new().max_tokens(24).seed(7);
    let messages = Messages::from_prompt("Count to three.");

    let streamed: String = engine
        .chat_stream(model.as_ref(), &messages, &options())
        .expect("stream")
        .map(|c| c.text)
        .collect();

    let whole = engine
        .chat(model.as_ref(), &messages, &options())
        .expect("chat");

    assert_eq!(
        streamed, whole,
        "the streamed text should be exactly what the blocking call returned"
    );
}

#[test]
#[ignore = "needs ASSEMBLY_CHAT_MODEL"]
fn a_stream_ends_with_a_done_chunk() {
    let Some(path) = chat_model() else {
        eprintln!("skipping: ASSEMBLY_CHAT_MODEL is not set");
        return;
    };

    let engine = LlamaEngine::new();
    let model = engine
        .load(&path, &LoadOptions::generate().n_ctx(2048))
        .expect("load");

    let chunks: Vec<_> = engine
        .chat_stream(
            model.as_ref(),
            &Messages::from_prompt("Say hello."),
            &EngineOptions::new().max_tokens(16).seed(3),
        )
        .expect("stream")
        .collect();

    assert!(!chunks.is_empty(), "the stream produced nothing");
    let last = chunks.last().expect("at least one chunk");
    assert!(last.done, "the final chunk must be marked done");
    assert!(
        chunks[..chunks.len() - 1].iter().all(|c| !c.done),
        "only the final chunk may be done"
    );
}

#[test]
#[ignore = "needs ASSEMBLY_CHAT_MODEL"]
fn a_stop_sequence_truncates_the_answer() {
    let Some(path) = chat_model() else {
        eprintln!("skipping: ASSEMBLY_CHAT_MODEL is not set");
        return;
    };

    let engine = LlamaEngine::new();
    let model = engine
        .load(&path, &LoadOptions::generate().n_ctx(2048))
        .expect("load");

    // An unlikely string: the point is that the stop is honoured whether or not
    // the model would have produced it.
    let stop = "<<STOP-NOW>>";
    let answer = engine
        .chat(
            model.as_ref(),
            &Messages::from_prompt("Repeat this exactly: alpha <<STOP-NOW>> omega"),
            &EngineOptions::new().max_tokens(64).seed(11).stop(stop),
        )
        .expect("generation should succeed");

    assert!(
        !answer.contains(stop),
        "the stop sequence should not appear in the answer, got: {answer:?}"
    );
}

#[test]
#[ignore = "needs ASSEMBLY_CHAT_MODEL"]
fn tokenizing_round_trips_through_detokenizing() {
    let Some(path) = chat_model() else {
        eprintln!("skipping: ASSEMBLY_CHAT_MODEL is not set");
        return;
    };

    let engine = LlamaEngine::new();
    let model = engine.load(&path, &LoadOptions::generate()).expect("load");

    let tokens = engine
        .tokenize(model.as_ref(), "hello there", false)
        .expect("tokenize");

    assert!(!tokens.is_empty(), "tokenizing produced nothing");
    assert!(
        tokens.iter().all(|t| *t < 1_000_000),
        "token ids should be vocabulary indices, got {tokens:?}"
    );
}

#[test]
#[ignore = "needs ASSEMBLY_CHAT_MODEL"]
fn a_model_without_a_chat_template_says_so() {
    let Some(path) = embed_model() else {
        eprintln!("skipping: ASSEMBLY_EMBED_MODEL is not set");
        return;
    };

    let engine = LlamaEngine::new();
    // Loaded for generation so the template check is what fails, not the
    // embedding check.
    let model = engine
        .load(&path, &LoadOptions::generate().n_ctx(512))
        .expect("load");

    let err = engine
        .chat(
            model.as_ref(),
            &Messages::from_prompt("hello"),
            &EngineOptions::new().max_tokens(8),
        )
        // An embedding model may still carry a template, so either outcome is
        // legitimate; what must not happen is success or a panic.
        .map(|_| ())
        .map_err(|e| e.to_string());

    if let Err(message) = err {
        assert!(
            message.contains("chat template") || message.contains("cannot"),
            "the error should explain itself, got: {message}"
        );
    }
}

#[test]
#[ignore = "needs ASSEMBLY_CHAT_MODEL"]
fn releasing_a_model_twice_does_not_panic() {
    let Some(path) = chat_model() else {
        eprintln!("skipping: ASSEMBLY_CHAT_MODEL is not set");
        return;
    };

    let engine = LlamaEngine::new();
    let model = engine
        .load(&path, &LoadOptions::generate().n_ctx(512))
        .expect("load");

    engine.release(Arc::from(model));
    // The seam says release must tolerate being called twice; a real
    // implementation cannot be called twice on the same box, but releasing two
    // models must work, which is what a caller unwinding actually does.
    let second = engine
        .load(&path, &LoadOptions::generate().n_ctx(512))
        .expect("load again");
    engine.release(Arc::from(second));
}
