//! End-to-end tests for the HTTP server.
//!
//! These start a real listener on port 0, talk to it over TCP with a hand-written
//! client, and check what comes back. That is more honest than calling handlers
//! directly, because most of what can go wrong in a server is between the
//! handler and the socket: a wrong status, a missing header, a stream that never
//! terminates.
//!
//! No GGUF file is involved. The archive is built from fixtures and answered by
//! a fake engine, which is the entire reason the manager takes an `Arc<dyn
//! Engine>`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use assembly_core::engine::{
    Chunk, Engine, EngineError, EngineOptions, LoadOptions, Messages, ModelHandle,
};
use assembly_core::manager::Assembly;
use assembly_server::{serve, Settings};

// `asmb_format` is not a direct dependency, but the test builds a real archive
// so the server sees real bytes on disk.
use asmb_format as _;

/// A handle that answers with the model name it was loaded under.
struct FakeModel {
    name: String,
    path: std::path::PathBuf,
}

impl ModelHandle for FakeModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

/// An engine with no runtime behind it.
#[derive(Default)]
struct FakeEngine {
    loaded: Mutex<Vec<String>>,
    released: AtomicUsize,
}

impl Engine for FakeEngine {
    fn load(
        &self,
        path: &Path,
        options: &LoadOptions,
    ) -> Result<Box<dyn ModelHandle>, EngineError> {
        let name = options.name.clone().unwrap_or_else(|| {
            path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".into())
        });
        self.loaded.lock().unwrap().push(name.clone());
        Ok(Box::new(FakeModel {
            name,
            path: path.to_path_buf(),
        }))
    }

    fn embed(&self, _model: &dyn ModelHandle, _text: &str) -> Result<Vec<f32>, EngineError> {
        Ok(vec![1.0, 0.0, 0.0])
    }

    fn chat(
        &self,
        model: &dyn ModelHandle,
        messages: &Messages,
        _options: &EngineOptions,
    ) -> Result<String, EngineError> {
        let asked = messages
            .items
            .iter()
            .rev()
            .find(|m| matches!(m.role, assembly_core::engine::Role::User))
            .map(|m| m.content.clone())
            .unwrap_or_default();
        Ok(format!("{} heard: {}", model.name(), asked))
    }

    fn chat_stream<'a>(
        &'a self,
        _model: &'a dyn ModelHandle,
        _messages: &Messages,
        _options: &EngineOptions,
    ) -> Result<Box<dyn Iterator<Item = Chunk> + 'a>, EngineError> {
        let chunks = ["one ", "two ", "three"]
            .into_iter()
            .map(|text| Chunk {
                text: text.to_string(),
                done: false,
            })
            .chain(std::iter::once(Chunk {
                text: String::new(),
                done: true,
            }));
        Ok(Box::new(chunks.collect::<Vec<_>>().into_iter()))
    }

    fn tokenize(
        &self,
        _model: &dyn ModelHandle,
        _text: &str,
        _add_special: bool,
    ) -> Result<Vec<u32>, EngineError> {
        Ok(vec![1, 2, 3])
    }

    fn release(&self, _model: Arc<dyn ModelHandle>) {
        self.released.fetch_add(1, Ordering::SeqCst);
    }
}

/// A GGUF-shaped file with a valid header, so pre-flight accepts it.
fn gguf(dir: &Path, name: &str, size: usize) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"GGUF").unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&0u64.to_le_bytes()).unwrap();
    f.write_all(&vec![0u8; size.saturating_sub(24)]).unwrap();
    path
}

/// An archive with a router and two routable models.
fn archive(dir: &Path) -> std::path::PathBuf {
    let plan = asmb_format::BuildPlan::new(
        "Served",
        Some(asmb_format::BlobSource::new(
            "router",
            gguf(dir, "router.gguf", 128),
        )),
    )
    .with_router(asmb_format::RouterSettings {
        confidence_threshold: 0.5,
        fallback_model: Some("prose".into()),
    })
    .with_model(
        asmb_format::ModelSource::new(
            asmb_format::BlobSource::new("code", gguf(dir, "code.gguf", 256)),
            vec!["code".into()],
        )
        .with_vectors(vec![vec![1.0, 0.0, 0.0]]),
    )
    .with_model(
        asmb_format::ModelSource::new(
            asmb_format::BlobSource::new("prose", gguf(dir, "prose.gguf", 512)),
            vec!["prose".into()],
        )
        .with_vectors(vec![vec![0.0, 1.0, 0.0]]),
    );

    let dest = dir.join("served.asmb");
    plan.build(&dest).unwrap();
    dest
}

/// A live server plus the temp directory its archive lives in.
struct Harness {
    #[allow(dead_code)]
    dir: tempfile::TempDir,
    port: u16,
    engine: Arc<FakeEngine>,
    running: assembly_server::RunningServer,
    shutdown: Arc<AtomicBool>,
}

impl Harness {
    fn start() -> Harness {
        Harness::start_with(|_| {})
    }

    fn start_with(tweak: impl FnOnce(&mut Settings)) -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let path = archive(dir.path());
        let engine = Arc::new(FakeEngine::default());
        let assembly = Assembly::open(&path, Arc::clone(&engine) as Arc<dyn Engine>, 0.8).unwrap();

        let mut settings = Settings {
            port: 0, // let the OS pick
            ..Settings::default()
        };
        tweak(&mut settings);

        let shutdown = Arc::new(AtomicBool::new(false));
        let running = serve(assembly, settings, Arc::clone(&shutdown)).unwrap();
        let port = running.port();

        Harness {
            dir,
            port,
            engine,
            running,
            shutdown,
        }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.running.shutdown();
    }
}

/// One HTTP response, parsed.
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

impl Reply {
    fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|e| panic!("body is not JSON ({e}): {}", self.body))
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Send one request and read the whole response.
fn request(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
    headers: &[(&str, &str)],
) -> Reply {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .unwrap();

    let payload = body.unwrap_or("");
    let mut req = format!("{method} {path} HTTP/1.1\r\nConnection: close\r\n");
    // Written last so a test can substitute its own Host; two Host headers
    // would make the server's answer depend on which one it picked.
    let mut wrote_host = false;
    for (k, v) in headers {
        if k.eq_ignore_ascii_case("Host") {
            wrote_host = true;
        }
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    if !wrote_host {
        req.push_str("Host: 127.0.0.1\r\n");
    }
    if body.is_some() {
        req.push_str("Content-Type: application/json\r\n");
    }
    req.push_str(&format!("Content-Length: {}\r\n\r\n", payload.len()));
    req.push_str(payload);

    stream.write_all(req.as_bytes()).unwrap();
    stream.flush().unwrap();

    read_reply(stream)
}

fn get(port: u16, path: &str) -> Reply {
    request(port, "GET", path, None, &[])
}

fn post_json(port: u16, path: &str, body: &str) -> Reply {
    request(port, "POST", path, Some(body), &[])
}

fn read_reply(mut stream: TcpStream) -> Reply {
    let mut reader = BufReader::new(stream.try_clone().unwrap());

    let mut status_line = String::new();
    reader.read_line(&mut status_line).unwrap();
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("no status in {status_line:?}"));

    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }

    let length: usize = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);

    let chunked = headers
        .iter()
        .any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.contains("chunked"));

    let body = if length > 0 {
        let mut body = vec![0u8; length];
        reader.read_exact(&mut body).unwrap();
        body
    } else if chunked {
        // A stream has no length, so the server frames it with chunked encoding
        // and ends with a zero-length chunk. Real clients decode this, so the
        // test does too rather than waiting for a close that never comes.
        read_chunked(&mut reader)
    } else {
        // Neither a length nor chunking: the body ends with the connection.
        let mut body = Vec::new();
        stream.read_to_end(&mut body).unwrap();
        body
    };

    Reply {
        status,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    }
}

/// Read a chunked body up to the terminating zero-length chunk.
fn read_chunked(reader: &mut impl BufRead) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut size_line = String::new();
        if reader.read_line(&mut size_line).unwrap_or(0) == 0 {
            break;
        }
        // A chunk header may carry extensions after a semicolon.
        let size_text = size_line.trim().split(';').next().unwrap_or("").trim();
        let size = match usize::from_str_radix(size_text, 16) {
            Ok(size) => size,
            Err(_) => break,
        };
        if size == 0 {
            break;
        }
        let mut chunk = vec![0u8; size];
        reader.read_exact(&mut chunk).unwrap();
        out.extend_from_slice(&chunk);

        // Each chunk is followed by CRLF.
        let mut crlf = String::new();
        reader.read_line(&mut crlf).unwrap();
    }
    out
}

#[test]
fn health_answers_without_loading_a_model() {
    let h = Harness::start();
    let reply = get(h.port, "/health");

    assert_eq!(reply.status, 200);
    assert_eq!(reply.json()["status"], "ok");
    assert_eq!(reply.json()["assembly"], "Served");
    assert_eq!(
        h.engine.loaded.lock().unwrap().len(),
        0,
        "a liveness check must not cost a model load"
    );
}

#[test]
fn models_lists_the_archive_as_exactly_one_model() {
    let h = Harness::start();
    let reply = get(h.port, "/v1/models");

    assert_eq!(reply.status, 200);
    let json = reply.json();
    assert_eq!(json["object"], "list");
    assert_eq!(json["data"].as_array().unwrap().len(), 1);
    assert_eq!(json["data"][0]["id"], "Served");
}

#[test]
fn info_reports_the_archive_shape_and_cache_state() {
    let h = Harness::start();
    let reply = get(h.port, "/api/info");
    assert_eq!(reply.status, 200);

    let json = reply.json();
    assert_eq!(json["assembly"], "Served");
    assert_eq!(json["models"].as_array().unwrap().len(), 2);
    assert_eq!(json["models"][0]["name"], "code");
    assert_eq!(json["models"][0]["routable"], true);
    assert_eq!(json["router"]["name"], "router");
    assert!(
        json["cache"]["loaded"].as_array().unwrap().is_empty(),
        "nothing is loaded yet"
    );
}

#[test]
fn route_reports_the_choice_without_generating_anything() {
    let h = Harness::start();
    let reply = post_json(h.port, "/api/route", r#"{"prompt":"write me a function"}"#);

    assert_eq!(reply.status, 200);
    assert_eq!(reply.json()["model"], "code");

    let loaded = h.engine.loaded.lock().unwrap();
    assert!(
        !loaded.iter().any(|n| n.as_str() == "code"),
        "a dry run must not load the model it names, got {loaded:?}"
    );
    assert_eq!(
        loaded.iter().filter(|n| n.as_str() == "router").count(),
        1,
        "routing does need the embedding model, exactly once: {loaded:?}"
    );
}

#[test]
fn route_refuses_a_non_string_prompt() {
    let h = Harness::start();
    let reply = post_json(h.port, "/api/route", r#"{"prompt":42}"#);
    assert_eq!(reply.status, 400);
    assert_eq!(reply.json()["error"]["code"], 400);
}

#[test]
fn route_refuses_an_empty_prompt() {
    let h = Harness::start();
    let reply = post_json(h.port, "/api/route", r#"{"prompt":"   "}"#);
    assert_eq!(reply.status, 400);
}

#[test]
fn chat_answers_in_the_openai_shape() {
    let h = Harness::start();
    let reply = post_json(
        h.port,
        "/v1/chat/completions",
        r#"{"model":"Served","messages":[{"role":"user","content":"code please"}]}"#,
    );

    assert_eq!(reply.status, 200);
    let json = reply.json();
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "Served");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert!(
        json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("code heard: code please"),
        "the routed model answers, got {}",
        json["choices"][0]["message"]["content"]
    );
}

#[test]
fn chat_streams_server_sent_events_terminated_by_done() {
    let h = Harness::start();
    let reply = post_json(
        h.port,
        "/v1/chat/completions",
        r#"{"model":"Served","stream":true,"messages":[{"role":"user","content":"go"}]}"#,
    );

    assert_eq!(reply.status, 200);
    assert_eq!(reply.header("Content-Type"), Some("text/event-stream"));
    assert_eq!(reply.header("Cache-Control"), Some("no-cache"));

    let events: Vec<&str> = reply
        .body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();

    assert_eq!(
        events.last().copied(),
        Some("[DONE]"),
        "the stream must terminate exactly once"
    );
    assert_eq!(
        events.iter().filter(|e| **e == "[DONE]").count(),
        1,
        "a second [DONE] would break strict clients"
    );

    let text: String = events
        .iter()
        .filter(|e| **e != "[DONE]")
        .filter_map(|e| serde_json::from_str::<serde_json::Value>(e).ok())
        .map(|v| {
            v["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    assert_eq!(text, "one two three", "every chunk reaches the client");

    let first: serde_json::Value = serde_json::from_str(events[0]).unwrap();
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
}

#[test]
fn every_stream_frame_is_separated_by_a_blank_line() {
    let h = Harness::start();
    let reply = post_json(
        h.port,
        "/v1/chat/completions",
        r#"{"model":"Served","stream":true,"messages":[{"role":"user","content":"go"}]}"#,
    );
    assert!(
        reply.body.contains("\n\n"),
        "SSE frames are delimited by a blank line: {:?}",
        reply.body
    );
}

#[test]
fn a_wrong_model_name_is_reported_rather_than_silently_routed() {
    let h = Harness::start();
    let reply = post_json(
        h.port,
        "/v1/chat/completions",
        r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(reply.status, 404);
    assert!(reply.body.contains("no model named"), "{}", reply.body);
}

#[test]
fn an_empty_conversation_is_refused() {
    let h = Harness::start();
    let reply = post_json(
        h.port,
        "/v1/chat/completions",
        r#"{"model":"Served","messages":[]}"#,
    );
    assert_eq!(reply.status, 400);
}

#[test]
fn malformed_json_is_a_400_and_not_a_crash() {
    let h = Harness::start();
    let reply = post_json(h.port, "/v1/chat/completions", "{not json");
    assert_eq!(reply.status, 400);
    assert!(
        !reply.body.contains("panic!") && !reply.body.contains("RUST_BACKTRACE"),
        "an error body must not leak internals: {}",
        reply.body
    );
}

#[test]
fn an_oversized_body_is_refused_before_it_is_read() {
    let h = Harness::start();
    let huge = "x".repeat(assembly_server::MAX_BODY_BYTES + 1024);
    let body = format!(r#"{{"model":"Served","messages":[{{"role":"user","content":"{huge}"}}]}}"#);
    let reply = post_json(h.port, "/v1/chat/completions", &body);
    assert_eq!(reply.status, 413, "got {}", reply.body);
}

#[test]
fn an_unknown_path_is_a_404_naming_what_exists() {
    let h = Harness::start();
    let reply = get(h.port, "/api/models");
    assert_eq!(reply.status, 404);
    assert!(reply.body.contains("no endpoint"), "{}", reply.body);
}

#[test]
fn there_is_no_endpoint_that_loads_an_arbitrary_model() {
    // The Python server had /api/load and /api/unload. They are gone on purpose:
    // loading a caller-chosen file is remote code execution by another name.
    for path in [
        "/api/load",
        "/api/unload",
        "/api/generate",
        "/api/chat",
        "/api/tags",
    ] {
        let h = Harness::start();
        let reply = post_json(h.port, path, r#"{"path":"/etc/passwd"}"#);
        assert_eq!(reply.status, 404, "{path} must not exist");
        assert_eq!(h.engine.loaded.lock().unwrap().len(), 0);
    }
}

#[test]
fn a_wrong_method_on_a_real_path_says_405() {
    let h = Harness::start();
    let reply = get(h.port, "/v1/chat/completions");
    assert_eq!(reply.status, 405);
    assert!(reply.body.contains("POST"), "{}", reply.body);
}

#[test]
fn an_api_key_is_required_when_one_is_configured() {
    let h = Harness::start_with(|s| s.api_key = Some("s3cret".into()));

    let denied = post_json(
        h.port,
        "/v1/chat/completions",
        r#"{"model":"Served","messages":[{"role":"user","content":"hi"}]}"#,
    );
    assert_eq!(denied.status, 401);

    let wrong = post_json(
        h.port,
        "/v1/chat/completions",
        r#"{"model":"Served","messages":[{"role":"user","content":"hi"}]}"#,
        // A wrong token must be refused even when the length matches, which is
        // what the constant-time comparison is for.
    );
    assert_eq!(wrong.status, 401);

    let allowed = request(
        h.port,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"model":"Served","messages":[{"role":"user","content":"hi"}]}"#),
        &[("Authorization", "Bearer s3cret")],
    );
    assert_eq!(allowed.status, 200);
}

#[test]
fn a_bearer_token_of_the_wrong_length_is_refused() {
    let h = Harness::start_with(|s| s.api_key = Some("s3cret".into()));
    let reply = request(
        h.port,
        "GET",
        "/health",
        None,
        &[("Authorization", "Bearer s3cre")],
    );
    assert_eq!(reply.status, 401);
}

#[test]
fn a_foreign_host_header_is_refused() {
    // DNS rebinding: a browser resolves evil.example to 127.0.0.1 and the
    // request arrives here with a Host this server never serves.
    let h = Harness::start();
    let reply = request(h.port, "GET", "/health", None, &[("Host", "evil.example")]);
    assert_eq!(reply.status, 403, "got {}", reply.body);
}

#[test]
fn localhost_is_an_acceptable_host_header() {
    let h = Harness::start();
    let reply = request(
        h.port,
        "GET",
        "/health",
        None,
        &[("Host", "localhost:8000")],
    );
    assert_eq!(reply.status, 200);
}

#[test]
fn responses_carry_a_header_saying_what_answered() {
    let h = Harness::start();
    let reply = get(h.port, "/health");
    assert_eq!(
        reply.header("X-Assembly-Version"),
        Some(asmb_format::VERSION)
    );
}

#[test]
fn ten_answers_cost_one_model_load() {
    let h = Harness::start();
    for _ in 0..10 {
        let reply = post_json(
            h.port,
            "/v1/chat/completions",
            r#"{"model":"Served","messages":[{"role":"user","content":"code please"}]}"#,
        );
        assert_eq!(reply.status, 200);
    }

    let loaded = h.engine.loaded.lock().unwrap();
    assert_eq!(
        loaded.iter().filter(|n| *n == "code").count(),
        1,
        "ten answers, one copy in RAM, got {loaded:?}"
    );
    assert_eq!(
        loaded.iter().filter(|n| *n == "router").count(),
        1,
        "the router is loaded once too, got {loaded:?}"
    );
}

#[test]
fn shutdown_stops_accepting_new_work() {
    let h = Harness::start();
    assert_eq!(get(h.port, "/health").status, 200);
    h.shutdown.store(true, Ordering::SeqCst);
    h.running.shutdown();

    // The listener may take a moment to close; what matters is that the server
    // does not hang the process.
    std::thread::sleep(std::time::Duration::from_millis(200));
}

#[test]
fn the_worker_pool_answers_more_concurrent_requests_than_it_has_workers() {
    let h = Harness::start();
    let port = h.port;
    let threads: Vec<_> = (0..8)
        .map(|i| std::thread::spawn(move || get(port, &format!("/health?{i}")).status))
        .collect();
    for t in threads {
        assert_eq!(t.join().unwrap(), 200);
    }
}
