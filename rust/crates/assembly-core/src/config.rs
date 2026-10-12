//! Build configuration: parsing and pre-flight validation.
//!
//! This mirrors `assembly/config.py`, deliberately rule for rule. A config that
//! builds under Python has to build under Rust, or the port has two answers to
//! "is this configuration valid" and neither is more correct than the other.
//!
//! # Why validate before building
//!
//! A build copies gigabytes. Learning afterwards that a path was wrong, that a
//! download was an HTML error page, or that two models share a name means
//! waiting out the whole copy first. Pre-flight answers all of it in
//! milliseconds, and reports everything it found rather than the first thing.
//!
//! # Never panics
//!
//! Every entry point returns a [`ValidationReport`] instead of failing on the
//! first bad field. A malformed config should produce a list of what is wrong,
//! not a stack trace.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use asmb_format::gguf;

/// Severity of a single finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// The build cannot proceed.
    Error,
    /// The build can proceed, but something looks wrong.
    Warning,
}

/// One thing that is wrong with a config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Issue {
    /// Error or warning.
    pub severity: Severity,
    /// Which field the problem is with, e.g. `models[1].path`.
    pub field: String,
    /// What is wrong, in one sentence.
    pub message: String,
}

impl Issue {
    fn error(field: impl Into<String>, message: impl Into<String>) -> Issue {
        Issue {
            severity: Severity::Error,
            field: field.into(),
            message: message.into(),
        }
    }

    fn warning(field: impl Into<String>, message: impl Into<String>) -> Issue {
        Issue {
            severity: Severity::Warning,
            field: field.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The field path is the useful part and the message repeats context on
        // its own, so this prints as `[models[0].path] model for 'x' is
        // shard 1 of 2 ...` rather than doubling every message with its own
        // field name.
        write!(f, "[{}] {}", self.field, self.message)
    }
}

/// The router model as the config names it.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingConfig {
    /// Name recorded in the archive header.
    pub name: String,
    /// Absolute path to the GGUF file.
    pub path: PathBuf,
    /// Cosine similarity below which the router stops trusting its pick.
    pub confidence_threshold: f32,
    /// Model to use when nothing clears the threshold.
    pub fallback_model: Option<String>,
}

/// One execution model as the config names it.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Name recorded in the archive header.
    pub name: String,
    /// Absolute path to the GGUF file.
    pub path: PathBuf,
    /// Phrases this model is meant to answer.
    pub anchors: Vec<String>,
}

/// A validated config, ready to build from.
#[derive(Debug, Clone, PartialEq)]
pub struct AssemblyConfig {
    /// Archive name.
    pub assembly_name: String,
    /// The router model.
    pub embedding: EmbeddingConfig,
    /// Execution models, in build order.
    pub models: Vec<ModelConfig>,
    /// System prompt prepended to every conversation.
    pub persona: Option<String>,
}

/// One row of the pre-flight summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    /// `router` or `execution`.
    pub role: &'static str,
    /// Model name from the config.
    pub name: String,
    /// Path it was found at.
    pub path: PathBuf,
    /// Size in bytes.
    pub size_bytes: u64,
}

/// What pre-flight found out.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// True when there are no errors. Warnings do not clear it.
    pub is_valid: bool,
    /// Everything found, errors and warnings.
    pub issues: Vec<Issue>,
    /// Combined size of every GGUF the config points at.
    pub total_size_bytes: u64,
    /// Free space on the volume holding the output, when it was checked.
    pub free_disk_bytes: u64,
    /// One line per model, for the summary table.
    pub models_info: Vec<ModelInfo>,
    /// The parsed config. `None` when validation failed too badly to build one.
    pub config: Option<AssemblyConfig>,
}

impl ValidationReport {
    /// Just the errors.
    pub fn errors(&self) -> impl Iterator<Item = &Issue> {
        self.issues.iter().filter(|i| i.severity == Severity::Error)
    }

    /// Just the warnings.
    pub fn warnings(&self) -> impl Iterator<Item = &Issue> {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
    }

    /// Errors and warnings, rendered for a terminal.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for issue in self.errors() {
            out.push_str(&format!("  x {issue}\n"));
        }
        for issue in self.warnings() {
            out.push_str(&format!("  ! {issue}\n"));
        }
        out
    }

    fn invalid(issues: Vec<Issue>) -> ValidationReport {
        ValidationReport {
            is_valid: false,
            issues,
            total_size_bytes: 0,
            free_disk_bytes: 0,
            models_info: Vec::new(),
            config: None,
        }
    }
}

/// Check a config file without building anything.
///
/// `output_path` enables the overwrite check and the disk-space check. Pass
/// `None` to validate a config whose destination is not yet decided.
///
/// `check_gguf_magic` reads each model's magic bytes and split metadata. It
/// costs one short read per file, and it is what catches a download that saved
/// an error page under a `.gguf` name, so it is on by default.
///
/// Use [`validate_with`] to allow replacing an existing archive.
pub fn validate(
    config_path: &Path,
    output_path: Option<&Path>,
    check_gguf_magic: bool,
) -> anyhow::Result<ValidationReport> {
    validate_with(config_path, output_path, check_gguf_magic, false)
}

/// [`validate`], with control over whether an existing output may be replaced.
///
/// The overwrite check is a separate knob rather than a special case inside
/// `validate` so `--force` cannot quietly skip the rest of the checks.
pub fn validate_with(
    config_path: &Path,
    output_path: Option<&Path>,
    check_gguf_magic: bool,
    allow_overwrite: bool,
) -> anyhow::Result<ValidationReport> {
    let text = match std::fs::read_to_string(config_path) {
        Ok(t) => t,
        Err(e) => {
            return Ok(ValidationReport::invalid(vec![Issue::error(
                "config_path",
                format!(
                    "config file does not exist or cannot be read: {}: {e}",
                    config_path.display()
                ),
            )]))
        }
    };

    let raw: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            return Ok(ValidationReport::invalid(vec![Issue::error(
                "json_syntax",
                format!("malformed JSON in {}: {e}", config_path.display()),
            )]))
        }
    };

    let mut issues = Vec::new();
    let mut total_size_bytes = 0u64;
    let mut models_info = Vec::new();

    let assembly_name = match raw.get("assembly_name").and_then(json_str) {
        Some(n) if !n.trim().is_empty() => n.to_string(),
        _ => {
            issues.push(Issue::error(
                "assembly_name",
                "missing or empty 'assembly_name'",
            ));
            String::new()
        }
    };

    let persona = match raw.get("persona") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(_) => {
            issues.push(Issue::error(
                "persona",
                "'persona' must be a string if defined",
            ));
            None
        }
    };

    let embedding = parse_embedding(raw.get("embedding"), &mut issues);
    if let Some(emb) = &embedding {
        if let Some(size) = check_model_file(
            &emb.path,
            "embedding.path",
            "",
            check_gguf_magic,
            &mut issues,
        ) {
            total_size_bytes += size;
            models_info.push(ModelInfo {
                role: "router",
                name: emb.name.clone(),
                path: emb.path.clone(),
                size_bytes: size,
            });
        }
    }

    let models = parse_models(raw.get("models"), &mut issues);
    for (idx, model) in models.iter().enumerate() {
        if let Some(size) = check_model_file(
            &model.path,
            &format!("models[{idx}].path"),
            &model.name,
            check_gguf_magic,
            &mut issues,
        ) {
            total_size_bytes += size;
            models_info.push(ModelInfo {
                role: "execution",
                name: model.name.clone(),
                path: model.path.clone(),
                size_bytes: size,
            });
        }
    }

    if let Some(emb) = &embedding {
        if let Some(fallback) = &emb.fallback_model {
            let known: Vec<&str> = models.iter().map(|m| m.name.as_str()).collect();
            if !known.contains(&fallback.as_str()) {
                issues.push(Issue::error(
                    "embedding.fallback_model",
                    format!("fallback_model '{fallback}' is not among defined models: {known:?}"),
                ));
            }
        }
    }

    let mut free_disk_bytes = 0u64;
    if let Some(out) = output_path {
        if out.exists() && !allow_overwrite {
            issues.push(Issue::error(
                "output",
                format!("{} already exists; refusing to overwrite it", out.display()),
            ));
        }

        // The archive holds a copy of every blob, so the sources' total is a
        // floor rather than an estimate. Padding and the header only add.
        free_disk_bytes = free_space(out.parent().unwrap_or_else(|| Path::new(".")));
        if free_disk_bytes > 0 && free_disk_bytes < total_size_bytes {
            issues.push(Issue::warning(
                "disk",
                format!(
                    "not enough free space: need about {} bytes, {} available",
                    total_size_bytes, free_disk_bytes
                ),
            ));
        }
    }

    let is_valid = !issues.iter().any(|i| i.severity == Severity::Error);
    let config = if is_valid {
        embedding.map(|embedding| AssemblyConfig {
            assembly_name,
            embedding,
            models,
            persona,
        })
    } else {
        None
    };

    Ok(ValidationReport {
        is_valid,
        issues,
        total_size_bytes,
        free_disk_bytes,
        models_info,
        config,
    })
}

fn json_str(value: &serde_json::Value) -> Option<&str> {
    value.as_str()
}

fn parse_embedding(
    value: Option<&serde_json::Value>,
    issues: &mut Vec<Issue>,
) -> Option<EmbeddingConfig> {
    let Some(obj) = value.and_then(|v| v.as_object()) else {
        issues.push(Issue::error(
            "embedding",
            "missing 'embedding' object in config",
        ));
        return None;
    };

    let name = match obj.get("name").and_then(json_str) {
        Some(n) if !n.is_empty() => n.to_string(),
        _ => {
            issues.push(Issue::error("embedding.name", "missing 'embedding.name'"));
            String::new()
        }
    };

    let Some(path) = resolve_path(obj.get("path")) else {
        path_problem(obj.get("path"), "embedding.path", "", issues);
        return None;
    };

    let confidence_threshold = match obj.get("confidence_threshold") {
        None => 0.75,
        Some(n) => match n.as_f64() {
            Some(v) if (0.0..=1.0).contains(&v) => v as f32,
            other => {
                issues.push(Issue::error(
                    "embedding.confidence_threshold",
                    format!(
                        "'confidence_threshold' must be between 0.0 and 1.0 (got {})",
                        other.map_or_else(|| n.to_string(), |v| v.to_string())
                    ),
                ));
                0.75
            }
        },
    };

    // `null` means "no fallback", which is different from "not a usable value".
    // Falling through to `None` on a number would quietly drop a typo and build an
    // archive whose routing has no fallback at all, so the bad type is reported.
    let fallback_model = match obj.get("fallback_model") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => match json_str(value) {
            Some(s) => Some(s.to_string()),
            None => {
                issues.push(Issue::error(
                    "embedding.fallback_model",
                    format!(
                        "fallback_model must be a string or null, got {}",
                        type_name(value)
                    ),
                ));
                None
            }
        },
    };

    Some(EmbeddingConfig {
        name,
        path,
        confidence_threshold,
        fallback_model,
    })
}

fn parse_models(value: Option<&serde_json::Value>, issues: &mut Vec<Issue>) -> Vec<ModelConfig> {
    let Some(entries) = value.and_then(|v| v.as_array()) else {
        issues.push(Issue::error(
            "models",
            "'models' must be a list of at least 2 models",
        ));
        return Vec::new();
    };

    if entries.len() < 2 {
        issues.push(Issue::error(
            "models",
            format!(
                "at least 2 execution models are required (got {})",
                entries.len()
            ),
        ));
        return Vec::new();
    }

    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::with_capacity(entries.len());

    for (idx, entry) in entries.iter().enumerate() {
        let Some(obj) = entry.as_object() else {
            issues.push(Issue::error(
                format!("models[{idx}]"),
                "model entry must be a JSON object",
            ));
            continue;
        };

        let name = match obj.get("name").and_then(json_str) {
            Some(n) if !n.trim().is_empty() => n.to_string(),
            _ => {
                issues.push(Issue::error(
                    format!("models[{idx}].name"),
                    "model name cannot be empty",
                ));
                continue;
            }
        };

        if !seen.insert(name.clone()) {
            issues.push(Issue::error(
                format!("models[{idx}].name"),
                format!("duplicate model name '{name}' (names must be unique)"),
            ));
        }

        let Some(path) = resolve_path(obj.get("path")) else {
            path_problem(
                obj.get("path"),
                &format!("models[{idx}].path"),
                &name,
                issues,
            );
            continue;
        };

        let anchors = parse_anchors(obj.get("anchors"), idx, &name, issues);

        out.push(ModelConfig {
            name,
            path,
            anchors,
        });
    }

    out
}

/// Read a model's anchor phrases, reporting anything malformed.
///
/// Every entry must be a string. Silently dropping the ones that are not would
/// let a typo -- `"anchors": ["hi", 42]` -- produce an archive that looks right
/// and routes on one phrase instead of two, which is the kind of mistake that
/// only shows up as odd behaviour much later.
fn parse_anchors(
    value: Option<&serde_json::Value>,
    idx: usize,
    name: &str,
    issues: &mut Vec<Issue>,
) -> Vec<String> {
    let where_ = format!("models[{idx}].anchors");

    let Some(value) = value else {
        issues.push(Issue::error(
            where_,
            format!("model '{name}' must list at least one anchor"),
        ));
        return Vec::new();
    };

    let Some(list) = value.as_array() else {
        issues.push(Issue::error(
            where_,
            format!(
                "model '{name}': anchors must be a list of phrases, got {}",
                type_name(value)
            ),
        ));
        return Vec::new();
    };

    // Two phrases that differ only in case or padding collapse to the same vector,
    // so the second one adds no routing signal. A warning, not an error: the
    // config still works, it just does less than it looks like it does.
    let mut seen: HashSet<String> = HashSet::new();
    let mut anchors = Vec::with_capacity(list.len());

    for (i, entry) in list.iter().enumerate() {
        match json_str(entry) {
            Some(s) if !s.trim().is_empty() => {
                let folded = s.trim().to_lowercase();
                if !seen.insert(folded) {
                    issues.push(Issue::warning(
                        format!("{where_}[{i}]"),
                        format!("duplicate anchor text '{s}' in model '{name}'"),
                    ));
                }
                anchors.push(s.to_string());
            }
            Some(_) => issues.push(Issue::error(
                format!("{where_}[{i}]"),
                format!("model '{name}': anchor phrase cannot be empty"),
            )),
            None => issues.push(Issue::error(
                format!("{where_}[{i}]"),
                format!(
                    "model '{name}': anchor phrase must be a string, got {}",
                    type_name(entry)
                ),
            )),
        }
    }

    if anchors.is_empty() && list.is_empty() {
        issues.push(Issue::error(
            where_,
            format!("model '{name}' must list at least one anchor"),
        ));
    } else if list.len() == 1 {
        // Counted from the raw list, not from `anchors`. An invalid entry has
        // already produced an error naming itself, and then warning that the
        // model is short an anchor is a second complaint about the same mistake
        // that buries the first one.
        issues.push(Issue::warning(
            where_,
            format!("model '{name}' has only 1 anchor; 3-5 improves routing accuracy"),
        ));
    }

    anchors
}

/// The JSON type of `value`, for an error message.
fn type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "a list",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Turn a JSON path value into an absolute existing file, or `None`.
///
/// Relative paths are rejected rather than resolved: a config is a portable
/// artifact, and resolving against the current directory would make the same
/// file mean different things depending on where the command ran.
fn resolve_path(value: Option<&serde_json::Value>) -> Option<PathBuf> {
    let text = value?.as_str()?.trim();
    if text.is_empty() {
        return None;
    }
    let path = PathBuf::from(text);
    if !path.is_absolute() || !path.is_file() {
        return None;
    }
    Some(path)
}

/// Say what is wrong with a config path, naming the file when there is one.
///
/// A config can be wrong about a path in three ways that need different fixes,
/// and "invalid path" leaves the reader guessing which one applies: a relative
/// path needs a prefix, a missing file needs a download, and a directory needs
/// the file inside it. The field carries the config location either way, so
/// which of several paths is broken is never in question.
fn path_problem(
    value: Option<&serde_json::Value>,
    field: &str,
    label: &str,
    issues: &mut Vec<Issue>,
) {
    let who = if label.is_empty() {
        String::new()
    } else {
        format!(" for '{label}'")
    };

    let Some(text) = value.and_then(json_str) else {
        issues.push(Issue::error(
            field,
            format!("model path{who} is missing or not a string"),
        ));
        return;
    };

    let path = PathBuf::from(text.trim());
    if text.trim().is_empty() {
        issues.push(Issue::error(field, format!("model path{who} is empty")));
    } else if !path.is_absolute() {
        issues.push(Issue::error(
            field,
            format!(
                "model path{who} must be absolute, got '{}'; a config is portable, so relative paths are not resolved against the current directory",
                text.trim()
            ),
        ));
    } else if !path.exists() {
        issues.push(Issue::error(
            field,
            format!("model path{who} does not exist: {}", path.display()),
        ));
    } else if !path.is_file() {
        issues.push(Issue::error(
            field,
            format!("model path{who} is not a file: {}", path.display()),
        ));
    }
}

/// Check one GGUF: non-empty, readable, right magic, not a split shard.
///
/// Returns its size in bytes when it passes.
fn check_model_file(
    path: &Path,
    field: &str,
    label: &str,
    check_magic: bool,
    issues: &mut Vec<Issue>,
) -> Option<u64> {
    let who = if label.is_empty() {
        String::new()
    } else {
        format!(" for '{label}'")
    };

    let size = match std::fs::metadata(path) {
        Ok(m) => m.len(),
        Err(e) => {
            issues.push(Issue::error(
                field,
                format!("could not read model{who} at {}: {e}", path.display()),
            ));
            return None;
        }
    };

    if size == 0 {
        issues.push(Issue::error(
            field,
            format!("model{who} file is empty (0 bytes): {}", path.display()),
        ));
        return None;
    }

    if check_magic {
        // One prefix read answers both questions: is this a GGUF at all, and is
        // it one shard of a split?
        let Some(meta) = gguf::metadata_from_file(path) else {
            issues.push(Issue::error(
                field,
                format!(
                    "model{who} at {} is not a GGUF (no GGUF header found; the download may be corrupt)",
                    path.display()
                ),
            ));
            return None;
        };

        if let Some((count, index)) = meta.split_info().filter(|(count, _)| *count > 1) {
            issues.push(Issue::error(
                field,
                format!(
                    "model{who} is shard {} of {count} from a split GGUF. llama.cpp cannot load a shard on its own, so this model would fail at inference time even though the archive builds cleanly. Merge the shards into one .gguf file and point the config at that.",
                    index + 1
                ),
            ));
        }
    }

    Some(size)
}

/// Free bytes on the volume holding `path`, or 0 when it cannot be determined.
fn free_space(path: &Path) -> u64 {
    // std has no statvfs, and a build that discovers a full disk an hour in is
    // the failure this check exists to prevent. `df` is the portable answer
    // without pulling in a platform-specific dependency for one number.
    let probe = if path.exists() { path } else { Path::new(".") };
    let Ok(output) = std::process::Command::new("df")
        .arg("-k")
        .arg(probe)
        .output()
    else {
        return 0;
    };
    if !output.status.success() {
        return 0;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let Some(line) = text.lines().nth(1) else {
        return 0;
    };
    // Filesystem, blocks, used, available, ...
    line.split_whitespace()
        .nth(3)
        .and_then(|kb| kb.parse::<u64>().ok())
        .map(|kb| kb.saturating_mul(1024))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write as _;

    /// A GGUF file with a real header: magic, version, no tensors, no metadata.
    fn gguf(dir: &Path, name: &str, size: usize) -> PathBuf {
        let path = dir.join(name);
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(&gguf::GGUF_MAGIC).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        // The header above is 24 bytes; pad the rest so the file is exactly
        // `size`, which the size assertions below depend on.
        f.write_all(&vec![0u8; size.saturating_sub(24)]).unwrap();
        path
    }

    /// One shard of a split GGUF, declared in its metadata the way llama.cpp
    /// requires.
    fn sharded_gguf(dir: &Path, name: &str, count: u64, index: u64, size: usize) -> PathBuf {
        let path = dir.join(name);
        let mut head = Vec::new();
        head.extend_from_slice(&gguf::GGUF_MAGIC);
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&0u64.to_le_bytes());
        head.extend_from_slice(&2u64.to_le_bytes());
        head.extend_from_slice(&(11u64).to_le_bytes());
        head.extend_from_slice(b"split.count");
        head.extend_from_slice(&4u32.to_le_bytes());
        head.extend_from_slice(&(count as u32).to_le_bytes());
        head.extend_from_slice(&(8u64).to_le_bytes());
        head.extend_from_slice(b"split.no");
        head.extend_from_slice(&5u32.to_le_bytes());
        head.extend_from_slice(&(index as u32).to_le_bytes());

        let mut f = fs::File::create(&path).unwrap();
        f.write_all(&head).unwrap();
        f.write_all(&vec![0u8; size.saturating_sub(head.len())])
            .unwrap();
        path
    }

    /// A file that is not a GGUF, as a corrupt download would be.
    fn not_gguf(dir: &Path, name: &str, size: usize) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, vec![b'<'; size]).unwrap();
        path
    }

    fn config(dir: &Path, models: &[(&str, &str)], embedding: &str) -> PathBuf {
        let models_json: Vec<String> = models
            .iter()
            .map(|(name, path)| {
                format!(r#"{{"name":"{name}","path":"{path}","anchors":["a phrase","another"]}}"#)
            })
            .collect();
        let json = format!(
            r#"{{
                "assembly_name": "Test",
                "embedding": {{"name":"route","path":"{embedding}","confidence_threshold":0.7}},
                "models": [{}]
            }}"#,
            models_json.join(",")
        );
        let path = dir.join("config.json");
        fs::write(&path, json).unwrap();
        path
    }

    #[test]
    fn accepts_a_good_config() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 300);
        let cfg = config(
            dir.path(),
            &[
                ("general", a.to_str().unwrap()),
                ("code", b.to_str().unwrap()),
            ],
            route.to_str().unwrap(),
        );

        let report = validate(&cfg, None, true).unwrap();
        assert!(report.is_valid, "{}", report.render());
        assert!(report.issues.is_empty());

        let parsed = report.config.expect("a valid config is returned");
        assert_eq!(parsed.assembly_name, "Test");
        assert_eq!(parsed.embedding.name, "route");
        assert_eq!(parsed.embedding.confidence_threshold, 0.7);
        assert_eq!(parsed.models.len(), 2);
        assert_eq!(parsed.models[0].name, "general");
        assert_eq!(
            parsed.models[0].anchors,
            vec!["a phrase".to_string(), "another".to_string()]
        );
        assert_eq!(report.total_size_bytes, 600);
        assert_eq!(report.models_info.len(), 3, "router plus two models");
        assert_eq!(report.models_info[0].role, "router");
        assert_eq!(report.models_info[0].name, "route");
    }

    #[test]
    fn reports_a_missing_config_file() {
        let report = validate(Path::new("/nope/config.json"), None, true).unwrap();
        assert!(!report.is_valid);
        assert!(report.config.is_none());
        assert!(
            report.render().contains("config_path"),
            "{}",
            report.render()
        );
    }

    #[test]
    fn reports_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("bad.json");
        fs::write(&cfg, "{not json").unwrap();
        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert_eq!(report.errors().next().unwrap().field, "json_syntax");
    }

    #[test]
    fn requires_two_models() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let cfg = config(
            dir.path(),
            &[("solo", a.to_str().unwrap())],
            route.to_str().unwrap(),
        );

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(
            report.render().contains("at least 2"),
            "{}",
            report.render()
        );
    }

    #[test]
    fn rejects_duplicate_model_names() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let cfg = config(
            dir.path(),
            &[("same", a.to_str().unwrap()), ("same", a.to_str().unwrap())],
            route.to_str().unwrap(),
        );

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(report.render().contains("duplicate"), "{}", report.render());
    }

    #[test]
    fn catches_a_corrupt_download() {
        let dir = tempfile::tempdir().unwrap();
        let route = not_gguf(dir.path(), "route.gguf", 500);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = config(
            dir.path(),
            &[("a", a.to_str().unwrap()), ("b", b.to_str().unwrap())],
            route.to_str().unwrap(),
        );

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(
            report.render().contains("not a GGUF"),
            "an HTML error page saved as .gguf: {}",
            report.render()
        );
    }

    #[test]
    fn magic_check_can_be_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let route = not_gguf(dir.path(), "route.gguf", 500);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = config(
            dir.path(),
            &[("a", a.to_str().unwrap()), ("b", b.to_str().unwrap())],
            route.to_str().unwrap(),
        );

        assert!(!validate(&cfg, None, true).unwrap().is_valid);
        assert!(
            validate(&cfg, None, false).unwrap().is_valid,
            "the magic check was skipped, so a non-GGUF is not caught"
        );
    }

    #[test]
    fn rejects_a_relative_path() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"relative/route.gguf"}},
                    "models":[{{"name":"a","path":"{}","anchors":["x"]}},
                              {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
                route.display(),
                route.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(report.render().contains("absolute"), "{}", report.render());
    }

    #[test]
    fn rejects_a_threshold_outside_zero_to_one() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}","confidence_threshold":1.5}},
                    "models":[{{"name":"a","path":"{}","anchors":["x"]}},
                              {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(report.render().contains("between 0.0 and 1.0"));
    }

    #[test]
    fn rejects_an_unknown_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}","fallback_model":"ghost"}},
                    "models":[{{"name":"a","path":"{}","anchors":["x"]}},
                              {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(report.render().contains("ghost"), "{}", report.render());
    }

    #[test]
    fn rejects_an_empty_model_file() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let empty = dir.path().join("empty.gguf");
        fs::write(&empty, b"").unwrap();
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = config(
            dir.path(),
            &[
                ("empty", empty.to_str().unwrap()),
                ("b", b.to_str().unwrap()),
            ],
            route.to_str().unwrap(),
        );

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(
            report.render().contains("empty (0 bytes)"),
            "{}",
            report.render()
        );
    }

    #[test]
    fn rejects_a_model_with_no_anchors() {
        // Python's ModelConfig raises here, and a model that can never win a
        // routing decision is not something to build on purpose.
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}"}},
                    "models":[{{"name":"a","path":"{}","anchors":[]}},
                              {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        let rendered = report.render();
        assert!(rendered.contains("at least one anchor"), "{rendered}");
        assert!(
            rendered.contains("models[0].anchors"),
            "the issue must point at the offending field: {rendered}"
        );
    }

    #[test]
    fn rejects_a_model_with_no_anchor_field_at_all() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}"}},
                    "models":[{{"name":"a","path":"{}"}},
                              {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(report.render().contains("at least one anchor"));
    }

    #[test]
    fn rejects_an_anchor_that_is_not_a_string() {
        // Dropping the bad entry silently would leave an archive that routes on
        // fewer phrases than the config lists.
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}"}},
                    "models":[{{"name":"a","path":"{}","anchors":["hi",42]}},
                              {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        let rendered = report.render();
        assert!(rendered.contains("models[0].anchors[1]"), "{rendered}");
        assert!(rendered.contains("must be a string"), "{rendered}");
        assert!(
            rendered.contains("a number"),
            "the message should say what it found: {rendered}"
        );
    }

    #[test]
    fn rejects_anchors_that_are_not_a_list() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}"}},
                    "models":[{{"name":"a","path":"{}","anchors":"hi"}},
                              {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        let rendered = report.render();
        assert!(rendered.contains("must be a list of phrases"), "{rendered}");
    }

    #[test]
    fn warns_about_duplicate_anchor_text() {
        // Two phrases differing only in case embed to the same vector, so the
        // second adds no routing signal.
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}"}},
                    "models":[{{"name":"a","path":"{}","anchors":["How are you","  how are you  "]}},
                              {{"name":"b","path":"{}","anchors":["x","y"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(report.is_valid, "a repeated phrase is not fatal");
        let rendered = report.render();
        assert!(rendered.contains("duplicate anchor text"), "{rendered}");
        assert!(rendered.contains("models[0].anchors[1]"), "{rendered}");
    }

    #[test]
    fn rejects_a_fallback_that_is_not_a_string() {
        // Falling through to `None` here would drop the typo and build an archive
        // whose routing has no fallback at all, with nothing to say so.
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}","fallback_model":7}},
                    "models":[{{"name":"a","path":"{}","anchors":["x","y"]}},
                              {{"name":"b","path":"{}","anchors":["x","y"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        let rendered = report.render();
        assert!(rendered.contains("embedding.fallback_model"), "{rendered}");
        assert!(rendered.contains("must be a string or null"), "{rendered}");
        assert!(
            rendered.contains("a number"),
            "name what it found: {rendered}"
        );
    }

    #[test]
    fn an_invalid_anchor_is_not_also_reported_as_too_few() {
        // One complaint per mistake. The error already names the bad entry, and
        // warning that the model is then short an anchor buries it.
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}"}},
                    "models":[{{"name":"a","path":"{}","anchors":["hi",42]}},
                              {{"name":"b","path":"{}","anchors":["x","y"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        let rendered = report.render();
        assert!(
            !rendered.contains("only 1 anchor"),
            "a bad entry should not also trigger the count warning:\n{rendered}"
        );
        assert_eq!(report.warnings().count(), 0, "{rendered}");
    }

    #[test]
    fn warns_about_a_model_with_only_one_anchor() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}"}},
                    "models":[{{"name":"a","path":"{}","anchors":["only one"]}},
                              {{"name":"b","path":"{}","anchors":["x","y"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(report.is_valid, "one anchor is legal, just narrow");
        assert!(
            report.render().contains("only 1 anchor"),
            "{}",
            report.render()
        );
    }

    #[test]
    fn rejects_an_empty_anchor_phrase() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}"}},
                    "models":[{{"name":"a","path":"{}","anchors":["hi","   "]}},
                              {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
                route.display(),
                a.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(report.render().contains("cannot be empty"));
    }

    #[test]
    fn refuses_to_overwrite_an_existing_archive() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = config(
            dir.path(),
            &[("a", a.to_str().unwrap()), ("b", b.to_str().unwrap())],
            route.to_str().unwrap(),
        );

        let out = dir.path().join("existing.asmb");
        fs::write(&out, b"already here").unwrap();

        let report = validate(&cfg, Some(&out), true).unwrap();
        assert!(!report.is_valid);
        assert!(report.render().contains("refusing to overwrite"));
    }

    #[test]
    fn rejects_a_split_gguf_shard() {
        // A shard copies and hashes perfectly but cannot be loaded, so catching
        // it here is the whole point.
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let shard = sharded_gguf(dir.path(), "q4-00002-of-00003.gguf", 3, 1, 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = config(
            dir.path(),
            &[
                ("shard", shard.to_str().unwrap()),
                ("b", b.to_str().unwrap()),
            ],
            route.to_str().unwrap(),
        );

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(
            report.render().contains("shard 2 of 3"),
            "{}",
            report.render()
        );
    }

    #[test]
    fn a_whole_model_is_not_mistaken_for_a_shard() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        // split.count = 1 means no split, whatever the filename says.
        let only = sharded_gguf(dir.path(), "part-00001-of-00001.gguf", 1, 0, 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = config(
            dir.path(),
            &[
                ("whole", only.to_str().unwrap()),
                ("b", b.to_str().unwrap()),
            ],
            route.to_str().unwrap(),
        );

        let report = validate(&cfg, None, true).unwrap();
        assert!(report.is_valid, "{}", report.render());
    }

    #[test]
    fn persona_may_be_absent_null_or_a_string() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);

        for (label, persona) in [
            ("absent", None),
            ("null", Some("null")),
            ("string", Some(r#""be brief""#)),
        ] {
            let body = persona
                .map(|p| format!(r#","persona":{p}"#))
                .unwrap_or_default();
            let cfg = dir.path().join(format!("{label}.json"));
            fs::write(
                &cfg,
                format!(
                    r#"{{"assembly_name":"T"{body},
                        "embedding":{{"name":"r","path":"{}"}},
                        "models":[{{"name":"a","path":"{}","anchors":["x"]}},
                                  {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
                    route.display(),
                    a.display(),
                    b.display()
                ),
            )
            .unwrap();

            let report = validate(&cfg, None, true).unwrap();
            assert!(report.is_valid, "persona {label}: {}", report.render());
            let expected = if label == "string" {
                Some("be brief".to_string())
            } else {
                None
            };
            assert_eq!(report.config.unwrap().persona, expected);
        }
    }

    #[test]
    fn collects_several_problems_at_once() {
        // The point of a report over fail-fast: one run tells you everything.
        let dir = tempfile::tempdir().unwrap();
        let route = not_gguf(dir.path(), "route.gguf", 100);
        let missing = dir.path().join("missing.gguf");
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"",
                    "embedding":{{"name":"r","path":"{}"}},
                    "models":[{{"name":"a","path":"{}","anchors":["x"]}},
                              {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
                route.display(),
                missing.display(),
                b.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        let errors = report.errors().count();
        assert!(
            errors >= 3,
            "expected several errors, got {errors}: {}",
            report.render()
        );
        assert!(report.render().contains("assembly_name"));
        assert!(report.render().contains("not a GGUF"));
        assert!(report.render().contains("missing.gguf"));
    }

    #[test]
    fn a_model_entry_that_is_not_an_object_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let cfg = dir.path().join("config.json");
        fs::write(
            &cfg,
            format!(
                r#"{{"assembly_name":"T",
                    "embedding":{{"name":"r","path":"{}"}},
                    "models":["just a string", 42]}}"#,
                route.display()
            ),
        )
        .unwrap();

        let report = validate(&cfg, None, true).unwrap();
        assert!(!report.is_valid);
        assert!(
            report.render().contains("must be a JSON object"),
            "{}",
            report.render()
        );
    }

    #[test]
    fn free_space_is_reported_for_an_output_path() {
        let dir = tempfile::tempdir().unwrap();
        let route = gguf(dir.path(), "route.gguf", 100);
        let a = gguf(dir.path(), "a.gguf", 200);
        let b = gguf(dir.path(), "b.gguf", 200);
        let cfg = config(
            dir.path(),
            &[("a", a.to_str().unwrap()), ("b", b.to_str().unwrap())],
            route.to_str().unwrap(),
        );

        let out = dir.path().join("fresh.asmb");
        let report = validate(&cfg, Some(&out), true).unwrap();
        assert!(report.is_valid, "{}", report.render());
        assert!(
            report.free_disk_bytes > 0,
            "a temp dir always has room; 0 means the probe failed"
        );
    }
}
