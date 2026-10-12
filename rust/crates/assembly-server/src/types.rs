//! The OpenAI-compatible shapes this server speaks.
//!
//! Compatibility here means *shape*, not behaviour: an OpenAI client sends
//! `model` and `messages`, gets back `choices[0].message.content`, and works.
//! Fields the API has and Assembly cannot honour -- `logprobs`, `n > 1`,
//! `logit_bias`, tool calls -- are accepted where ignoring them is harmless and
//! rejected where ignoring them would silently change the answer.
//!
//! `model` deserves a note. An OpenAI client requires the field, and sending a
//! model name that is not in `/v1/models` is a normal thing to do with other
//! providers, so it is checked but not treated as routing input. Assembly routes
//! on the prompt; letting a client name the model would hand the router's job
//! to whoever wrote the request.

use serde::{Deserialize, Serialize};

/// A chat completion request.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    /// Model name. Required by the API; checked against the archive, not used
    /// for routing.
    #[serde(default)]
    pub model: String,

    /// The conversation.
    pub messages: Vec<Message>,

    /// Whether to stream the answer as server-sent events.
    #[serde(default)]
    pub stream: bool,

    /// Sampling temperature. Clamped to what the runtime accepts.
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Nucleus sampling cut-off.
    #[serde(default)]
    pub top_p: Option<f32>,

    /// How many completions to return. Only 1 is supported.
    #[serde(default)]
    pub n: Option<u32>,

    /// Seed for reproducible sampling.
    #[serde(default)]
    pub seed: Option<u64>,

    /// Stop sequences.
    #[serde(default)]
    pub stop: Option<Vec<String>>,

    /// Maximum tokens to generate.
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// Tokens of context the client wants room for. Ignored with a note in
    /// `/api/info` rather than rejected, because clients send it reflexively.
    #[serde(default)]
    pub user: Option<String>,
}

impl ChatRequest {
    /// What is wrong with a request, if anything.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.messages.is_empty() {
            return Err("messages must not be empty");
        }
        if let Some(n) = self.n {
            if n != 1 {
                return Err("n must be 1; this server returns one completion");
            }
        }
        if let Some(stop) = &self.stop {
            if stop.len() > 4 {
                return Err("at most 4 stop sequences are supported");
            }
            if stop.iter().any(|s| s.is_empty()) {
                return Err("stop sequences must not be empty");
            }
        }
        Ok(())
    }

    /// The last user message, which is what Assembly routes on.
    pub fn routing_text(&self) -> Option<&str> {
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
    }
}

/// One message in a conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: Role,
    /// The text.
    pub content: String,
}

/// A message role.
///
/// Only the four the chat API defines are accepted. An unknown role is refused
/// rather than passed through, because a runtime that receives a role it does not
/// understand will usually answer it as if it were the user's turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions that frame the conversation.
    System,
    /// The person asking.
    User,
    /// The model, replying.
    Assistant,
    /// The result of a tool call.
    Tool,
}

/// A non-streaming chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    /// Always `chat.completion`, for clients that switch on it.
    pub object: &'static str,
    /// The archive name.
    pub model: String,
    /// Server-assigned identifier.
    pub id: String,
    /// Unix seconds.
    pub created: u64,
    /// The choices. Always exactly one.
    pub choices: Vec<Choice>,
    /// How the request was served.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One completion.
#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    /// Always zero: only one completion is returned.
    pub index: u32,
    /// The message.
    pub message: ResponseMessage,
    /// Why generation stopped.
    pub finish_reason: &'static str,
}

/// The assistant's reply.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    /// Always `assistant`.
    pub role: Role,
    /// The text.
    pub content: String,
}

/// Token accounting.
///
/// Counts are the runtime's own when it reports them and zero when it does not.
/// Reporting a fabricated count would be worse than reporting none: a client
/// that bills on `usage` would bill on fiction.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Tokens generated.
    pub completion_tokens: u32,
    /// Sum of the two.
    pub total_tokens: u32,
}

/// One streamed chunk.
#[derive(Debug, Clone, Serialize)]
pub struct Chunk {
    /// Always `chat.completion.chunk`.
    pub object: &'static str,
    /// The archive name.
    pub model: String,
    /// Matches the non-streaming response id.
    pub id: String,
    /// Unix seconds.
    pub created: u64,
    /// The choices.
    pub choices: Vec<ChunkChoice>,
}

/// One streamed choice.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    /// Always zero.
    pub index: u32,
    /// The delta. Empty text is still sent: it is how a client learns the
    /// generation ended.
    pub delta: ResponseMessage,
    /// `stop` on the final chunk, absent before that.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

/// The `GET /v1/models` listing.
#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    /// Always `list`.
    pub object: &'static str,
    /// The models.
    pub data: Vec<ModelCard>,
}

/// One entry in the model listing.
#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    /// The model name.
    pub id: String,
    /// Always `model`, as the API requires.
    pub object: &'static str,
    /// When the server started, in Unix seconds. The API has no field for
    /// "archive build time", and this is the closest honest answer.
    pub created: u64,
    /// Who owns it. Always `assembly`.
    pub owned_by: String,
}

/// An error, in the shape OpenAI clients expect.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    /// The error.
    pub error: ErrorDetail,
}

/// The error itself.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    /// Human-readable explanation. Never a stack trace.
    pub message: String,
    /// The HTTP status as a string, or a short slug like `not_found`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The HTTP status code.
    pub code: u16,
    /// The server's parameter name when one applies.
    #[serde(rename = "param", skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

impl ErrorDetail {
    /// An error with a status and a message.
    pub fn new(code: u16, kind: &str, message: impl Into<String>) -> Self {
        ErrorDetail {
            message: message.into(),
            kind: kind.to_string(),
            code,
            param: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(json: &str) -> ChatRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn parses_a_minimal_request() {
        let r = request(r#"{"model":"a","messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(r.model, "a");
        assert!(!r.stream);
        assert_eq!(r.routing_text(), Some("hi"));
        r.validate().unwrap();
    }

    #[test]
    fn a_missing_model_field_is_accepted_because_it_is_ignored() {
        // Clients differ on whether they send it. Since Assembly routes on the
        // prompt, refusing here would break clients over a field we never read.
        let r = request(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(r.model, "");
        r.validate().unwrap();
    }

    #[test]
    fn an_empty_conversation_is_refused() {
        let r = request(r#"{"model":"a","messages":[]}"#);
        assert!(r.validate().is_err());
    }

    #[test]
    fn asking_for_more_than_one_completion_is_refused_rather_than_ignored() {
        let r = request(r#"{"model":"a","n":3,"messages":[{"role":"user","content":"x"}]}"#);
        assert!(r.validate().is_err());
    }

    #[test]
    fn routing_text_is_the_last_user_turn_not_the_first() {
        let r = request(
            r#"{"model":"a","messages":[
                {"role":"user","content":"first"},
                {"role":"assistant","content":"reply"},
                {"role":"user","content":"second"}
            ]}"#,
        );
        assert_eq!(
            r.routing_text(),
            Some("second"),
            "the latest question is the one that decides the model"
        );
    }

    #[test]
    fn an_unknown_role_is_refused_at_the_wire() {
        let err = serde_json::from_str::<ChatRequest>(
            r#"{"model":"a","messages":[{"role":"wizard","content":"x"}]}"#,
        );
        assert!(
            err.is_err(),
            "a runtime would not know what to do with this"
        );
    }

    #[test]
    fn an_empty_stop_sequence_is_refused() {
        let r = request(r#"{"model":"a","stop":[""],"messages":[{"role":"user","content":"x"}]}"#);
        assert!(r.validate().is_err(), "an empty stop matches everything");
    }

    #[test]
    fn usage_is_omitted_when_absent_rather_than_serialised_as_null() {
        let body = serde_json::to_string(&Usage::default()).unwrap();
        assert!(
            body.contains("total_tokens"),
            "zero is a real count: {body}"
        );
    }
}
