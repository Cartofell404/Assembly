//! A local HTTP server that speaks the OpenAI chat API over an Assembly archive.
//!
//! Five endpoints, and no more:
//!
//! | Method | Path                   | What it does                                |
//! |--------|------------------------|---------------------------------------------|
//! | GET    | `/health`              | Liveness, no model needed                   |
//! | GET    | `/v1/models`           | What this archive can route to              |
//! | POST   | `/v1/chat/completions` | Chat, streaming or not                      |
//! | GET    | `/api/info`            | Archive shape and cache state               |
//! | POST   | `/api/route`           | Which model would answer, without generating |
//!
//! Everything else is a 404. That is a deliberate choice rather than an
//! omission: the earlier Python server also exposed `/api/load` and
//! `/api/unload`, which take a filesystem path and load an arbitrary GGUF. On a
//! machine that runs a local model, that endpoint is remote code execution by
//! another name -- it loads and parses attacker-chosen binary files, and binary
//! parsers are exactly the kind of code that has had bugs. The archive to serve
//! is chosen when the process starts, and there is no way to ask for a different
//! one over HTTP.
//!
//! # Safety posture
//!
//! Binds loopback by default. Refuses to start on a public address without an
//! API key. Checks the `Host` header against the addresses it actually serves,
//! which blocks DNS rebinding -- a browser tricked into resolving some domain to
//! `127.0.0.1` would otherwise be able to drive this server and read the
//! conversation. Bounds request bodies before reading them. And returns JSON
//! errors, never stack traces: an error message is not the place to leak a path
//! or a line of Rust source.

use std::io::{self, Read};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use assembly_core::engine::{EngineOptions, Messages, Role as EngineRole};
use assembly_core::manager::Assembly;
use serde_json::json;
use tiny_http::{Header, ListenAddr, Method, Request, Response, Server, StatusCode};

use crate::settings::{Settings, MAX_BODY_BYTES, SHUTDOWN_POLL_SECS};
use crate::types::{
    ChatRequest, ChatResponse, Choice, ErrorBody, ErrorDetail, Message, ModelCard, ModelList,
    ResponseMessage, Role, Usage,
};

/// The chat completions endpoint.
pub const CHAT_PATH: &str = "/v1/chat/completions";

/// The model listing endpoint.
pub const MODELS_PATH: &str = "/v1/models";

/// The liveness endpoint.
pub const HEALTH_PATH: &str = "/health";

/// The archive introspection endpoint.
pub const INFO_PATH: &str = "/api/info";

/// The routing-dry-run endpoint.
pub const ROUTE_PATH: &str = "/api/route";

/// Every path this server answers, for the startup banner.
pub const ENDPOINTS: &[(&str, &str)] = &[
    ("GET", HEALTH_PATH),
    ("GET", MODELS_PATH),
    ("GET", INFO_PATH),
    ("POST", CHAT_PATH),
    ("POST", ROUTE_PATH),
];

/// The one response body type.
///
/// Everything answers with `Box<dyn Read>`, so a JSON response and a live stream
/// can share a signature. SSE works by handing `tiny_http` a reader that
/// produces bytes on demand rather than a buffer that is already complete --
/// which is the whole difference between a stream and a slow non-stream.
type Body = Box<dyn Read + Send>;

/// A running server.
pub struct RunningServer {
    server: Arc<Server>,
    shutdown: Arc<AtomicBool>,
    /// The port actually bound, which differs from the requested one when the
    /// caller asked for port 0.
    port: u16,
}

impl RunningServer {
    /// The port the server is listening on.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Ask the workers to stop.
    ///
    /// In-flight requests are not cancelled: a client halfway through an answer
    /// gets a complete answer. Interrupting a llama.cpp decode mid-stream is not
    /// something the binding makes safe.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        self.server.unblock();
    }
}

impl Drop for RunningServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

/// What the request handlers share.
struct State {
    assembly: Assembly,
    settings: Settings,
    started: u64,
    next_id: AtomicU64,
}

impl State {
    /// A response id, in the shape OpenAI clients expect.
    fn new_id(&self) -> String {
        format!(
            "chatcmpl-{:016x}",
            self.next_id.fetch_add(1, Ordering::SeqCst)
        )
    }

    fn created(&self) -> u64 {
        self.started
    }

    /// Handle one request.
    fn handle(&self, request: &mut Request) -> Response<Body> {
        let path = request.url().split('?').next().unwrap_or("").to_string();

        if self.reject_foreign_host(request) {
            return self.error(403, "forbidden", "unknown Host header");
        }
        if !self.authorized(request) {
            return self.error(401, "unauthorized", "a bearer token is required");
        }

        let known_get = matches!(path.as_str(), HEALTH_PATH | MODELS_PATH | INFO_PATH);
        let known_post = matches!(path.as_str(), CHAT_PATH | ROUTE_PATH);

        match (request.method(), path.as_str()) {
            (Method::Get, HEALTH_PATH) => health(self.assembly.name()),
            (Method::Get, MODELS_PATH) => self.models(),
            (Method::Get, INFO_PATH) => self.info(),
            (Method::Post, ROUTE_PATH) => self.route(request),
            (Method::Post, CHAT_PATH) => self.chat(request),
            // A wrong method on a known path is a 405, not a 404: the client can
            // fix it, and saying so saves a confused debugging session.
            _ if known_get || known_post => self.error(
                405,
                "method_not_allowed",
                format!("{path} accepts {}", if known_get { "GET" } else { "POST" }),
            ),
            _ => self.error(404, "not_found", format!("no endpoint at {path}")),
        }
    }

    /// Whether a request's `Host` names somewhere this server does not serve.
    ///
    /// The check exists for DNS rebinding: a browser is talked into resolving
    /// some domain the attacker controls to `127.0.0.1`, then sends requests here
    /// with that domain in `Host`. Same-origin policy stops it *reading* the
    /// replies, but only if the reply is not a simple cross-origin response --
    /// and a browser that can make this server generate text at all is already a
    /// problem. Refusing the mismatched `Host` closes it.
    fn reject_foreign_host(&self, request: &Request) -> bool {
        if self.settings.is_public() {
            // Bound to the world already, so the operator has accepted the risk.
            // Refusing here would make the public mode unusable.
            return false;
        }

        let host = match request.headers().iter().find(|h| h.field.equiv("Host")) {
            Some(header) => String::from_utf8_lossy(header.value.as_bytes()).into_owned(),
            // HTTP/1.0 clients may omit Host entirely. There is nothing to
            // rebind, so there is nothing to check.
            None => return false,
        };

        // Strip the port. Bracketed IPv6 is handled first because `[::1]:8000`
        // has two colons and splitting on the last one would leave a bracket.
        let host = host.trim();
        let authority = if let Some(rest) = host.strip_prefix('[') {
            rest.split(']').next().unwrap_or("")
        } else {
            host.rsplit_once(':')
                .map(|(name, _port)| name)
                .unwrap_or(host)
        };

        // This server is bound to a loopback address, so the two names that
        // legitimately reach it are that address and `localhost`.
        let bound = &self.settings.host;
        let is_loopback = |name: &str| {
            name.parse::<std::net::IpAddr>()
                .map(|ip| ip.is_loopback())
                .unwrap_or(false)
        };

        !(authority == bound || authority == "localhost" || is_loopback(authority))
    }

    /// Check the bearer token when one was configured.
    ///
    /// The comparison is constant-time so a wrong token cannot be discovered one
    /// character at a time, and it compares raw bytes so a token containing
    /// non-ASCII cannot make the handler fail before it reaches its own error
    /// path.
    fn authorized(&self, request: &Request) -> bool {
        let expected = match &self.settings.api_key {
            Some(key) => key,
            None => return true,
        };
        let header = match request
            .headers()
            .iter()
            .find(|h| h.field.equiv("Authorization"))
        {
            Some(h) => String::from_utf8_lossy(h.value.as_bytes()).into_owned(),
            None => return false,
        };
        let (scheme, token) = match header.split_once(' ') {
            Some(parts) => parts,
            None => return false,
        };
        if !scheme.eq_ignore_ascii_case("bearer") {
            return false;
        }
        let (got, want) = (token.trim().as_bytes(), expected.as_bytes());
        got.len() == want.len()
            && got
                .iter()
                .zip(want.iter())
                .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                == 0
    }

    /// `GET /v1/models`.
    ///
    /// Lists the archive name as one model, because that is what a client may
    /// put in `model`. Listing the individual execution models would suggest a
    /// client can choose one, and it cannot: routing is the whole point.
    fn models(&self) -> Response<Body> {
        json_response(
            200,
            &ModelList {
                object: "list",
                data: vec![ModelCard {
                    id: self.assembly.name().to_string(),
                    object: "model",
                    created: self.created(),
                    owned_by: "assembly".to_string(),
                }],
            },
        )
    }

    /// `GET /api/info`.
    fn info(&self) -> Response<Body> {
        let header = self.assembly.header();
        let models: Vec<_> = header
            .models
            .iter()
            .map(|m| {
                json!({
                    "name": m.name,
                    "bytes": m.size,
                    "anchors": m.anchors.len(),
                    "vectors": m.anchor_vectors().map(|v| v.len()).unwrap_or(0),
                    "dimensions": m.anchor_dim,
                    "routable": self.assembly.routable_models().iter().any(|n| n == &m.name),
                })
            })
            .collect();

        json_response(
            200,
            &json!({
                "assembly": self.assembly.name(),
                "archive": self.assembly.path(),
                "persona": self.assembly.persona(),
                "router": {
                    "name": header.embedding.name,
                    "confidence_threshold": header.embedding.confidence_threshold,
                    "fallback_model": header.embedding.fallback_model,
                },
                "models": models,
                "cache": {
                    "loaded": self.assembly.loaded_models(),
                    "loaded_bytes": self.assembly.loaded_bytes(),
                    "staged_bytes": self.assembly.staged_bytes(),
                    "headroom": self.settings.headroom,
                },
            }),
        )
    }

    fn error(&self, code: u16, kind: &str, message: impl Into<String>) -> Response<Body> {
        error_response(code, kind, message)
    }

    /// `POST /api/route`: which model would answer, without generating.
    fn route(&self, request: &mut Request) -> Response<Body> {
        let raw = match read_body(request) {
            Ok(raw) => raw,
            Err((code, why)) => return error_response(code, error_kind(code), why),
        };
        let body: serde_json::Value = match serde_json::from_slice(&raw) {
            Ok(v) => v,
            Err(e) => {
                return error_response(400, "invalid_request_error", format!("malformed body: {e}"))
            }
        };
        let prompt = match body.get("prompt").and_then(|p| p.as_str()) {
            Some(p) => p,
            None => return error_response(400, "invalid_request_error", "prompt must be a string"),
        };
        if prompt.trim().is_empty() {
            return error_response(400, "invalid_request_error", "prompt must not be empty");
        }

        match self.assembly.route(prompt) {
            Ok(route) => json_response(
                200,
                &json!({
                    "prompt": prompt,
                    "model": route.model_name,
                    "score": route.score,
                    "is_fallback": route.is_fallback,
                    "skipped": route.skipped,
                }),
            ),
            Err(e) => error_response(500, "internal_error", e.to_string()),
        }
    }

    /// `POST /v1/chat/completions`.
    fn chat(&self, request: &mut Request) -> Response<Body> {
        let raw = match read_body(request) {
            Ok(raw) => raw,
            Err((code, why)) => return error_response(code, error_kind(code), why),
        };
        let parsed: ChatRequest = match serde_json::from_slice(&raw) {
            Ok(v) => v,
            Err(e) => {
                return error_response(
                    400,
                    "invalid_request_error",
                    format!("malformed request body: {e}"),
                )
            }
        };
        if let Err(why) = parsed.validate() {
            return error_response(400, "invalid_request_error", why);
        }

        // A named model that is not this archive is a mistake worth reporting.
        // It is not a routing input: Assembly decides, which is the feature.
        if !parsed.model.is_empty() && parsed.model != self.assembly.name() {
            return error_response(
                404,
                "not_found",
                format!(
                    "no model named '{}'; this server has one model, '{}'",
                    parsed.model,
                    self.assembly.name()
                ),
            );
        }

        let options = engine_options(&parsed);
        let id = self.new_id();
        let model = self.assembly.name().to_string();
        let created = self.created();

        if parsed.stream {
            let Some(text) = parsed.routing_text() else {
                return error_response(400, "invalid_request_error", "no user message to route on");
            };
            // The whole conversation, not just the last turn: a chat that
            // forgets after one answer is not a chat.
            let messages = to_messages(&parsed.messages);
            return self.stream_chat(messages, text.to_string(), options, id, model, created);
        }

        let messages = to_messages(&parsed.messages);
        match self.assembly.chat(&messages, &options) {
            Ok(reply) => json_response(
                200,
                &ChatResponse {
                    object: "chat.completion",
                    model,
                    id,
                    created,
                    choices: vec![Choice {
                        index: 0,
                        message: ResponseMessage {
                            role: Role::Assistant,
                            content: reply.text,
                        },
                        finish_reason: "stop",
                    }],
                    usage: Some(Usage::default()),
                },
            ),
            Err(e) => error_response(500, "generation_failed", e.to_string()),
        }
    }

    /// Stream an answer as server-sent events.
    ///
    /// Routing happens *before* the response is built, so a routing failure is a
    /// clean error status rather than a 200 followed by an error event nobody
    /// asked for. After that the response body is a reader that pulls one chunk
    /// at a time, which is what makes this streaming: the first bytes reach the
    /// client while the model is still generating the rest.
    fn stream_chat(
        &self,
        messages: Messages,
        text: String,
        options: EngineOptions,
        id: String,
        model: String,
        created: u64,
    ) -> Response<Body> {
        let route = match self.assembly.route(&text) {
            Ok(route) => route,
            Err(e) => return error_response(500, "routing_failed", e.to_string()),
        };

        // A model that is not in RAM is a cold start for this turn: the blob is
        // staged out of the archive and loaded before the first token, which on
        // a big model is tens of seconds of silence. The client is told, so a
        // switch reads as a switch rather than a hang.
        let warming = if self.assembly.loaded_models().contains(&route.model_name) {
            String::new()
        } else {
            format!(
                "loading '{}' into memory, this can take a while",
                route.model_name
            )
        };

        let assembly = self.assembly.clone();
        let options = Arc::new(options);

        let producer: Producer = Box::new(move |out: &mut Vec<u8>| {
            // `finished` is what stops the reader asking for more. A producer that
            // always writes something would be pulled forever: the reader cannot
            // tell "still generating" from "one more time".
            let mut finished = false;
            let mut first = true;

            // An SSE comment: every client ignores a line starting with `: `,
            // so OpenAI-compatible consumers are unaffected and this one can
            // say what is happening before generation starts.
            if !warming.is_empty() {
                out.extend_from_slice(format!(": {warming}\n\n").as_bytes());
            }

            if let Err(e) = assembly.chat_stream(&messages, &options, &mut |chunk| {
                // The first chunk carries the role, as the API does, so a client
                // reading deltas incrementally knows who is speaking.
                let role = if first { Some("assistant") } else { None };
                first = false;
                write_sse(
                    out,
                    &chunk_payload(&id, &model, created, &chunk.text, role, None),
                );

                if chunk.done {
                    write_sse(
                        out,
                        &chunk_payload(&id, &model, created, "", None, Some("stop")),
                    );
                    write_sse(out, "[DONE]");
                    finished = true;
                    // Stop generation: the stream is complete.
                    return Err(());
                }
                Ok(())
            }) {
                // The model stopped without saying it was done, or the first
                // chunk never arrived. Either way the client is still owed a
                // terminator, or it waits forever.
                let payload =
                    json!({ "error": ErrorDetail::new(500, "generation_failed", e.to_string()) })
                        .to_string();
                write_sse(out, &payload);
                write_sse(out, "[DONE]");
                finished = true;
            }

            !finished
        });

        Response::new(
            StatusCode(200),
            vec![
                header("Content-Type", "text/event-stream"),
                header("Cache-Control", "no-cache"),
                // nginx will otherwise buffer the whole response, which turns a
                // stream back into a slow non-stream.
                header("X-Accel-Buffering", "no"),
                header("X-Assembly-Version", asmb_format::VERSION),
            ],
            Box::new(ProducerStream::new(producer)) as Body,
            // No length: the response is not known until generation finishes.
            None,
            None,
        )
    }
}

/// Produces SSE bytes on demand, as the socket asks for them.
///
/// Returns `false` once the stream is finished, which is how the reader knows to
/// report EOF rather than ask again.
type Producer = Box<dyn FnMut(&mut Vec<u8>) -> bool + Send>;

/// Adapts a `Producer` to `Read`.
///
/// `tiny_http` writes a response body by reading until EOF, so streaming is just
/// a question of when the next read blocks. This produces the next frame on
/// demand and reports EOF once the producer stops writing anything, which is
/// what ends a stream.
struct ProducerStream {
    producer: Option<Producer>,
    pending: Vec<u8>,
}

impl ProducerStream {
    fn new(producer: Producer) -> Self {
        ProducerStream {
            producer: Some(producer),
            pending: Vec::new(),
        }
    }
}

impl Read for ProducerStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            if !self.pending.is_empty() {
                break;
            }
            let producer = match self.producer.as_mut() {
                Some(p) => p,
                // Already drained.
                None => return Ok(0),
            };
            let mut frame = Vec::new();
            let more = producer(&mut frame);
            if !more || frame.is_empty() {
                // The stream is over, or the producer had nothing to say and
                // will not say more. Either way: EOF, which ends the response.
                self.producer = None;
                if frame.is_empty() {
                    return Ok(0);
                }
            }
            self.pending = frame;
        }

        let n = buf.len().min(self.pending.len());
        buf[..n].copy_from_slice(&self.pending[..n]);
        self.pending.drain(..n);
        Ok(n)
    }
}

/// Write one SSE frame.
///
/// The trailing blank line is mandatory: it is what separates events, and a
/// client that never sees one waits forever for a complete frame.
fn write_sse(out: &mut Vec<u8>, data: &str) {
    out.extend_from_slice(format!("data: {data}\n\n").as_bytes());
}

/// A streamed chunk as JSON.
fn chunk_payload(
    id: &str,
    model: &str,
    created: u64,
    delta: &str,
    role: Option<&str>,
    finish: Option<&'static str>,
) -> String {
    json!({
        "object": "chat.completion.chunk",
        "model": model,
        "id": id,
        "created": created,
        "choices": [{
            "index": 0,
            "delta": {"role": role, "content": delta},
            "finish_reason": finish,
        }],
    })
    .to_string()
}

/// Translate wire messages into engine messages.
fn to_messages(wire: &[Message]) -> Messages {
    let mut messages = Messages::new();
    for m in wire {
        messages.push(assembly_core::engine::Message::new(
            to_role(m.role),
            m.content.clone(),
        ));
    }
    messages
}

fn to_role(role: Role) -> EngineRole {
    match role {
        Role::System => EngineRole::System,
        Role::User => EngineRole::User,
        Role::Assistant => EngineRole::Assistant,
        Role::Tool => EngineRole::Tool,
    }
}

/// Build engine options from a request.
fn engine_options(req: &ChatRequest) -> EngineOptions {
    let mut options = EngineOptions::default();
    if let Some(t) = req.temperature {
        options = options.temperature(t);
    }
    if let Some(p) = req.top_p {
        options = options.top_p(p);
    }
    if let Some(m) = req.max_tokens {
        options = options.max_tokens(m);
    }
    if let Some(seed) = req.seed {
        options = options.seed(seed);
    }
    if let Some(stop) = req.stop.clone() {
        for sequence in stop {
            options = options.stop(sequence);
        }
    }
    options
}

/// Read a bounded request body.
fn read_body(request: &mut Request) -> Result<Vec<u8>, (u16, String)> {
    // Refuse an oversized declared length before reading a byte. The header is
    // client-controlled, so this is what stops a one-line request from making the
    // server allocate a gigabyte.
    if let Some(declared) = request.body_length() {
        if declared > MAX_BODY_BYTES {
            return Err((413, format!("request body exceeds {MAX_BODY_BYTES} bytes")));
        }
    }
    let mut raw = Vec::new();
    request
        .as_reader()
        .take((MAX_BODY_BYTES + 1) as u64)
        .read_to_end(&mut raw)
        .map_err(|e| (400, format!("cannot read request body: {e}")))?;
    if raw.len() > MAX_BODY_BYTES {
        return Err((413, format!("request body exceeds {MAX_BODY_BYTES} bytes")));
    }
    Ok(raw)
}

/// The `error.type` slug for a status code.
fn error_kind(code: u16) -> &'static str {
    match code {
        400 => "invalid_request_error",
        401 => "unauthorized",
        403 => "forbidden",
        404 => "not_found",
        405 => "method_not_allowed",
        413 => "payload_too_large",
        _ => "internal_error",
    }
}

/// `GET /health`.
fn health(name: &str) -> Response<Body> {
    json_response(
        200,
        &json!({
            "status": "ok",
            "assembly": name,
            "version": asmb_format::VERSION,
        }),
    )
}

fn header(field: &str, value: &str) -> Header {
    Header::from_bytes(field.as_bytes(), value.as_bytes())
        .unwrap_or_else(|_| panic!("{field}: {value} is a valid header"))
}

/// A JSON response with the standard headers.
fn json_response<T: serde::Serialize>(code: u16, body: &T) -> Response<Body> {
    let payload = serde_json::to_vec(body).unwrap_or_else(|_| {
        br#"{"error":{"message":"could not serialise the response","type":"internal_error","code":500}}"#
            .to_vec()
    });
    let len = payload.len();
    Response::new(
        StatusCode(code),
        vec![
            header("Content-Type", "application/json"),
            header("Content-Length", &len.to_string()),
            header("X-Assembly-Version", asmb_format::VERSION),
        ],
        Box::new(std::io::Cursor::new(payload)) as Body,
        Some(len),
        None,
    )
}

/// An error in the shape OpenAI clients expect.
fn error_response(code: u16, kind: &str, message: impl Into<String>) -> Response<Body> {
    json_response(
        code,
        &ErrorBody {
            error: ErrorDetail::new(code, kind, message),
        },
    )
}

/// Unix seconds, or zero if the clock is before the epoch.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Start the server.
///
/// Returns once the workers are running. The workers keep running until
/// `shutdown` is set, so the caller can inspect [`RunningServer::port`] before
/// anyone is talking to it.
///
/// # Errors
///
/// Returns [`ServeError`] when the address cannot be bound or a worker cannot
/// start.
pub fn serve(
    assembly: Assembly,
    settings: Settings,
    shutdown: Arc<AtomicBool>,
) -> Result<RunningServer, ServeError> {
    let addr = settings.bind_addr()?;
    let server = Arc::new(Server::http(addr).map_err(|e| ServeError::Bind {
        addr,
        reason: e.to_string(),
    })?);

    let port = match server.server_addr() {
        ListenAddr::IP(ip) => ip.port(),
        ListenAddr::Unix(_) => settings.port,
    };

    let state = Arc::new(State {
        assembly,
        settings: settings.clone(),
        started: unix_now(),
        next_id: AtomicU64::new(1),
    });

    for worker in 0..settings.workers.max(1) {
        let server = Arc::clone(&server);
        let state = Arc::clone(&state);
        let shutdown = Arc::clone(&shutdown);
        thread::Builder::new()
            .name(format!("assembly-http-{worker}"))
            .spawn(move || {
                loop {
                    if shutdown.load(Ordering::SeqCst) {
                        return;
                    }
                    // A timeout rather than a blocking read, so shutdown is
                    // noticed within a second instead of at the next request.
                    match server.recv_timeout(Duration::from_secs(SHUTDOWN_POLL_SECS)) {
                        Ok(Some(request)) => handle_one(&state, request),
                        Ok(None) => continue,
                        Err(_) => return,
                    }
                }
            })
            .map_err(|e| ServeError::Worker(e.to_string()))?;
    }

    Ok(RunningServer {
        server,
        shutdown,
        port,
    })
}

/// Serve one request and send its response.
///
/// A panic is caught here rather than left to unwind the worker. There are only
/// `settings.workers` of these threads and nothing supervises them, so a panic
/// used to take one down quietly: the client saw a dropped connection with no
/// explanation, and after four the server still listened but answered nobody.
/// Now the panic is reported on the server's stderr -- where someone can read
/// it -- and the client gets a 500 it can act on.
///
/// Chat and error responses are logged with how long they took. Everything
/// else is silent: the client's discovery polls `/health` every half second,
/// and a line per probe would drown the two lines that matter.
fn handle_one(state: &State, mut request: Request) {
    let method = request.method().to_string();
    let path = request.url().split('?').next().unwrap_or("").to_string();
    let is_chat = path == CHAT_PATH;
    let before = is_chat.then(|| state.assembly.loaded_models());
    let started = std::time::Instant::now();

    let response =
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| state.handle(&mut request)))
        {
            Ok(response) => response,
            Err(panic) => {
                let why = panic
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("a value that was not a string");
                // The default panic hook has already printed where and what; this
                // adds which request was being served when it happened.
                eprintln!("assembly: {method} {path} failed: {why}");
                error_response(
                    500,
                    "internal_error",
                    "the server failed while handling this request; see its terminal for why",
                )
            }
        };
    let status = response.status_code().0;

    // A failed write means the client hung up mid-answer. There is nothing to
    // report and nothing to retry: the next request will start fresh.
    let _ = request.respond(response);

    if !is_chat && status < 400 {
        return;
    }
    let elapsed = started.elapsed();
    let took = if elapsed.as_secs() >= 1 {
        format!("{:.1}s", elapsed.as_secs_f64())
    } else {
        format!("{}ms", elapsed.as_millis())
    };

    if let Some(before) = before {
        // A switch to another model in the archive is the slow path: the blob
        // is staged out of the archive and loaded before a token is generated,
        // and whatever no longer fits is evicted to make room. Saying so turns
        // "this looks hung" into "this is loading code", and a machine that
        // evicts on every other turn is visible instead of merely slow.
        let after = state.assembly.loaded_models();
        let loaded: Vec<&String> = after.iter().filter(|name| !before.contains(name)).collect();
        let dropped: Vec<&String> = before.iter().filter(|name| !after.contains(name)).collect();
        if !loaded.is_empty() || !dropped.is_empty() {
            let mut changes = Vec::new();
            if !loaded.is_empty() {
                changes.push(format!(
                    "loaded {}",
                    loaded
                        .iter()
                        .map(|n| n.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            if !dropped.is_empty() {
                changes.push(format!(
                    "evicted {}",
                    dropped
                        .iter()
                        .map(|n| n.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            eprintln!("assembly: {} ({took})", changes.join(", "));
        }
    }
    eprintln!("assembly: {method} {path} -> {status} in {took}");
}

/// Why the server could not start.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    /// The settings did not describe a usable address.
    #[error(transparent)]
    Settings(#[from] crate::settings::SettingsError),

    /// The address could not be bound.
    #[error("cannot listen on {addr}: {reason}")]
    Bind {
        /// The address.
        addr: std::net::SocketAddr,
        /// What the OS said.
        reason: String,
    },

    /// A worker thread could not be started.
    #[error("cannot start a worker thread: {0}")]
    Worker(String),
}
