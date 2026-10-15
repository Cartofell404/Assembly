//! End-to-end tests for the `assembly` binary.
//!
//! These run the built binary rather than calling into the library, so they
//! cover the parts unit tests cannot reach: argument parsing as the shell sees
//! it, exit codes, and what a person actually reads on a terminal.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A path, escaped for the inside of a JSON string literal.
///
/// `Path::display` uses the platform separator, so a Windows temporary path
/// arrives as `C:\Users\...` and `\U` is not a valid JSON escape, which would
/// make the generated config unparseable before any check could see it. Running
/// the text through `serde_json` gives exactly the escaping JSON expects.
fn jp(path: impl std::fmt::Display) -> String {
    let quoted = serde_json::to_string(&path.to_string()).unwrap();
    quoted[1..quoted.len() - 1].to_string()
}

/// Path to the binary under test.
///
/// `CARGO_BIN_EXE_*` is set by cargo for integration tests, which is the
/// supported way to find it: the debug path differs by profile, by target, and
/// by whether the workspace has been built from a subdirectory.
fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_assembly")
}

struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Fixture {
        Fixture {
            dir: tempfile::tempdir().unwrap(),
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// A GGUF-shaped file with a valid header, so pre-flight accepts it.
    fn gguf(&self, name: &str, size: usize) -> PathBuf {
        let path = self.path(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&vec![0u8; size.saturating_sub(24)]).unwrap();
        path
    }

    /// A config naming a router plus two models.
    fn config(&self, router: &Path, models: &[(&str, &Path)]) -> PathBuf {
        let entries: Vec<String> = models
            .iter()
            .map(|(name, path)| {
                format!(
                    r#"{{"name":"{name}","path":"{}","anchors":["some work"]}}"#,
                    jp(path.display())
                )
            })
            .collect();
        let json = format!(
            r#"{{
                "assembly_name": "Integration",
                "embedding": {{"name":"route","path":"{}","confidence_threshold":0.6}},
                "models": [{}]
            }}"#,
            jp(router.display()),
            entries.join(",")
        );
        let path = self.path("config.json");
        std::fs::write(&path, json).unwrap();
        path
    }

    /// A config with three models, all valid.
    fn good(&self) -> PathBuf {
        let router = self.gguf("route.gguf", 512);
        let a = self.gguf("a.gguf", 4096);
        let b = self.gguf("b.gguf", 8192);
        let c = self.gguf("c.gguf", 1024);
        self.config(
            &router,
            &[
                ("general", a.as_path()),
                ("code", b.as_path()),
                ("small", c.as_path()),
            ],
        )
    }
}

fn run(args: &[&str]) -> Output {
    Command::new(binary())
        .args(args)
        .output()
        .expect("the binary under test should be runnable")
}

fn stdout_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr_of(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn code(out: &Output) -> i32 {
    out.status.code().expect("the process exited normally")
}

#[test]
fn help_lists_every_subcommand() {
    for flag in ["--help", "-h", "help"] {
        let out = run(&[flag]);
        assert_eq!(code(&out), 0, "{flag} exited nonzero");
        let text = stderr_of(&out);
        for expected in ["build", "validate", "info", "--verify", "embed"] {
            assert!(text.contains(expected), "{flag} omits {expected}:\n{text}");
        }
    }
}

#[test]
fn version_prints_the_crate_version() {
    let out = run(&["--version"]);
    assert_eq!(code(&out), 0);
    assert_eq!(
        stdout_of(&out).trim(),
        format!("assembly {}", asmb_format::VERSION)
    );
}

#[test]
fn no_arguments_is_an_error_with_usage() {
    let out = run(&[]);
    assert_ne!(code(&out), 0);
    assert!(
        stderr_of(&out).contains("no subcommand given"),
        "{}",
        stderr_of(&out)
    );
}

#[test]
fn an_unknown_subcommand_is_rejected() {
    let out = run(&["frobnicate"]);
    assert_ne!(code(&out), 0);
    assert!(
        stderr_of(&out).contains("unknown subcommand"),
        "{}",
        stderr_of(&out)
    );
}

#[test]
fn build_writes_an_archive_info_can_read() {
    let fx = Fixture::new();
    let config = fx.good();
    let archive = fx.path("mesh.asmb");

    let out = run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        archive.to_str().unwrap(),
        "--no-embed",
    ]);
    assert_eq!(code(&out), 0, "build failed:\n{}", stderr_of(&out));
    assert!(archive.exists(), "no archive at {}", archive.display());

    // An archive built with --no-embed has phrases but no vectors, so it can
    // never win a routing decision. It has to say so, rather than leaving the
    // discovery to whoever tries to serve it.
    let stderr = stderr_of(&out);
    assert!(
        stderr.contains("--no-embed") && stderr.contains("cannot route"),
        "build did not warn that a vector-less archive cannot route:\n{stderr}"
    );

    // The whole point of writing it is that `info` can read it back.
    let out = run(&["info", archive.to_str().unwrap()]);
    assert_eq!(code(&out), 0, "info failed:\n{}", stderr_of(&out));
    let text = stdout_of(&out);
    assert!(text.contains("Integration"), "{text}");
    for model in ["general", "code", "small"] {
        assert!(text.contains(model), "info omits {model}:\n{text}");
    }
    assert!(text.contains("route"), "info omits the router:\n{text}");

    // Anchor phrases must show up, even when no vectors exist for them yet.
    // Reporting these as absent is what a user would act on, so it has to be
    // true.
    assert!(
        text.contains("ANCHORS"),
        "info has no anchor column:\n{text}"
    );
    let general_line = text
        .lines()
        .find(|l| l.starts_with("general"))
        .unwrap_or_else(|| panic!("no line for general:\n{text}"));
    assert!(
        general_line.contains('1'),
        "general has one anchor phrase but info does not say so: {general_line}"
    );
    assert!(
        general_line.contains('-'),
        "general has no vectors yet and info should show that: {general_line}"
    );
}

#[test]
fn info_separates_anchor_phrases_from_anchor_vectors() {
    // The two are independent: a phrase without a vector is normal when the
    // archive was built without embedding, and collapsing them into one count
    // reported a model with phrases as having none.
    let fx = Fixture::new();
    let router = fx.gguf("route.gguf", 512);
    let a = fx.gguf("a.gguf", 4096);
    let b = fx.gguf("b.gguf", 8192);
    let cfg = fx.path("anchors.json");
    std::fs::write(
        &cfg,
        format!(
            r#"{{"assembly_name":"Anchors",
                "embedding":{{"name":"route","path":"{}"}},
                "models":[{{"name":"phrases_only","path":"{}","anchors":["hello","goodbye"]}},
                          {{"name":"other","path":"{}","anchors":["x"]}}]}}"#,
            jp(router.display()),
            jp(a.display()),
            jp(b.display())
        ),
    )
    .unwrap();

    let archive = fx.path("anchors.asmb");
    let built = run(&[
        "build",
        cfg.to_str().unwrap(),
        "-o",
        archive.to_str().unwrap(),
        "--no-embed",
    ]);
    assert_eq!(code(&built), 0, "{}", stderr_of(&built));

    let text = stdout_of(&run(&["info", archive.to_str().unwrap()]));
    let line = text
        .lines()
        .find(|l| l.starts_with("phrases_only"))
        .unwrap_or_else(|| panic!("no line for phrases_only:\n{text}"));

    assert!(line.contains('2'), "both phrases should be counted: {line}");
    assert!(
        line.contains('-'),
        "no vectors, so the vector columns should say so: {line}"
    );
}

#[test]
fn a_built_archive_verifies_against_its_own_digests() {
    let fx = Fixture::new();
    let config = fx.good();
    let archive = fx.path("mesh.asmb");

    assert_eq!(
        code(&run(&[
            "build",
            config.to_str().unwrap(),
            "-o",
            archive.to_str().unwrap(),
            "--no-embed",
        ])),
        0
    );

    let out = run(&["info", archive.to_str().unwrap(), "--verify"]);
    assert_eq!(code(&out), 0, "verification failed:\n{}", stdout_of(&out));
    let text = stdout_of(&out);
    assert!(text.contains("all blobs match"), "{text}");
    assert_eq!(
        text.matches(" ok").count(),
        4,
        "router plus three models:\n{text}"
    );
}

#[test]
fn a_corrupted_blob_fails_verification() {
    let fx = Fixture::new();
    let config = fx.good();
    let archive = fx.path("mesh.asmb");
    run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        archive.to_str().unwrap(),
        "--no-embed",
    ]);

    // Flip a byte inside the first model's blob.
    let mut bytes = std::fs::read(&archive).unwrap();
    let victim = bytes.len() - 1;
    bytes[victim] ^= 0xff;
    std::fs::write(&archive, &bytes).unwrap();

    let out = run(&["info", archive.to_str().unwrap(), "--verify"]);
    assert_ne!(code(&out), 0, "a corrupted blob verified clean");
    assert!(stdout_of(&out).contains("MISMATCH"), "{}", stdout_of(&out));
}

#[test]
fn the_output_default_sits_beside_the_config() {
    let fx = Fixture::new();
    let config = fx.good();
    let out = run(&["build", config.to_str().unwrap(), "--no-embed"]);
    assert_eq!(code(&out), 0, "{}", stderr_of(&out));
    assert!(
        fx.path("config.asmb").exists(),
        "expected config.asmb beside the config, found: {:?}",
        std::fs::read_dir(fx.path(""))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect::<Vec<_>>()
    );
}

#[test]
fn build_refuses_to_overwrite_without_force() {
    let fx = Fixture::new();
    let config = fx.good();
    let archive = fx.path("mesh.asmb");
    assert_eq!(
        code(&run(&[
            "build",
            config.to_str().unwrap(),
            "-o",
            archive.to_str().unwrap(),
            "--no-embed",
        ])),
        0
    );

    let out = run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        archive.to_str().unwrap(),
        "--no-embed",
    ]);
    assert_ne!(
        code(&out),
        0,
        "the second build replaced the first without --force"
    );
    assert!(
        stderr_of(&out).contains("refusing to overwrite"),
        "{}",
        stderr_of(&out)
    );
}

#[test]
fn force_replaces_an_existing_archive() {
    let fx = Fixture::new();
    let config = fx.good();
    let archive = fx.path("mesh.asmb");
    run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        archive.to_str().unwrap(),
        "--no-embed",
    ]);

    let out = run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        archive.to_str().unwrap(),
        "--force",
        "--no-embed",
    ]);
    assert_eq!(code(&out), 0, "{}", stderr_of(&out));
    assert_eq!(
        code(&run(&["info", archive.to_str().unwrap(), "--verify"])),
        0,
        "the replaced archive should still verify"
    );
}

#[test]
fn no_partial_is_left_behind() {
    let fx = Fixture::new();
    let config = fx.good();
    let archive = fx.path("mesh.asmb");
    run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        archive.to_str().unwrap(),
        "--no-embed",
    ]);

    let leftovers: Vec<String> = std::fs::read_dir(fx.path(""))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".partial"))
        .collect();
    assert!(leftovers.is_empty(), "found {leftovers:?}");
}

#[test]
fn a_missing_config_file_fails_with_a_readable_message() {
    let out = run(&["build", "/definitely/not/here.json"]);
    assert_ne!(code(&out), 0);
    let text = stderr_of(&out);
    assert!(text.contains("config_path"), "{text}");
    assert!(text.contains("nothing was written"), "{text}");
    assert!(
        !text.contains("panicked"),
        "a user error should not panic:\n{text}"
    );
}

#[test]
fn a_config_with_one_model_is_rejected() {
    let fx = Fixture::new();
    let router = fx.gguf("route.gguf", 256);
    let only = fx.gguf("solo.gguf", 256);
    let config = fx.config(&router, &[("solo", only.as_path())]);

    let out = run(&["build", config.to_str().unwrap(), "--no-embed"]);
    assert_ne!(code(&out), 0);
    assert!(
        stderr_of(&out).contains("at least 2"),
        "{}",
        stderr_of(&out)
    );
}

#[test]
fn a_corrupt_model_is_caught_before_anything_is_copied() {
    let fx = Fixture::new();
    let router = fx.gguf("route.gguf", 256);
    let broken = fx.path("broken.gguf");
    std::fs::write(&broken, b"<html>404 not found</html>").unwrap();
    let good = fx.gguf("good.gguf", 256);
    let config = fx.config(
        &router,
        &[("broken", broken.as_path()), ("ok", good.as_path())],
    );

    let archive = fx.path("out.asmb");
    let out = run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        archive.to_str().unwrap(),
        "--no-embed",
    ]);
    assert_ne!(code(&out), 0);
    assert!(
        stderr_of(&out).contains("not a GGUF"),
        "{}",
        stderr_of(&out)
    );
    assert!(!archive.exists(), "nothing should have been written");
}

#[test]
fn every_problem_is_reported_at_once() {
    let fx = Fixture::new();
    let router = fx.gguf("route.gguf", 256);
    let good = fx.gguf("good.gguf", 256);
    let config = fx.path("config.json");
    std::fs::write(
        &config,
        format!(
            r#"{{"assembly_name":"",
                "embedding":{{"name":"r","path":"{}"}},
                "models":[{{"name":"a","path":"{}","anchors":["x"]}},
                          {{"name":"a","path":"{}","anchors":["x"]}}]}}"#,
            jp(router.display()),
            jp(fx.path("gone.gguf").display()),
            jp(good.display())
        ),
    )
    .unwrap();

    let out = run(&["build", config.to_str().unwrap(), "--no-embed"]);
    assert_ne!(code(&out), 0);
    let text = stderr_of(&out);
    // Three separate mistakes; one run should name all of them.
    assert!(text.contains("assembly_name"), "{text}");
    assert!(text.contains("does not exist"), "{text}");
    assert!(text.contains("duplicate"), "{text}");
}

#[test]
fn validate_accepts_a_good_config() {
    let fx = Fixture::new();
    let config = fx.good();
    let out = run(&["validate", config.to_str().unwrap()]);
    assert_eq!(code(&out), 0, "{}", stderr_of(&out));
    assert!(stdout_of(&out).contains("is valid"), "{}", stdout_of(&out));
}

#[test]
fn validate_writes_nothing() {
    let fx = Fixture::new();
    let config = fx.good();
    let before = std::fs::read_dir(fx.path("")).unwrap().count();
    assert_eq!(code(&run(&["validate", config.to_str().unwrap()])), 0);
    assert_eq!(
        std::fs::read_dir(fx.path("")).unwrap().count(),
        before,
        "validate must not create files"
    );
    assert!(!fx.path("config.asmb").exists());
}

#[test]
fn validate_reports_every_error_and_exits_nonzero() {
    let fx = Fixture::new();
    let config = fx.path("config.json");
    std::fs::write(&config, "{ not json").unwrap();

    let out = run(&["validate", config.to_str().unwrap()]);
    assert_ne!(code(&out), 0);
    assert!(
        stderr_of(&out).contains("json_syntax"),
        "{}",
        stderr_of(&out)
    );
}

#[test]
fn validate_finds_a_duplicate_the_build_would_also_reject() {
    let fx = Fixture::new();
    let router = fx.gguf("route.gguf", 256);
    let a = fx.gguf("a.gguf", 256);
    let config = fx.config(&router, &[("same", a.as_path()), ("same", a.as_path())]);

    let out = run(&["validate", config.to_str().unwrap()]);
    assert_ne!(code(&out), 0);
    assert!(stderr_of(&out).contains("duplicate"), "{}", stderr_of(&out));
}

#[test]
fn a_corrupt_archive_is_rejected_by_info() {
    let fx = Fixture::new();
    let path = fx.path("fake.asmb");
    std::fs::write(&path, b"this is not an archive").unwrap();

    let out = run(&["info", path.to_str().unwrap()]);
    assert_ne!(code(&out), 0);
    assert!(stderr_of(&out).contains("magic"), "{}", stderr_of(&out));
}

#[test]
fn info_without_a_path_is_an_error() {
    let out = run(&["info"]);
    assert_ne!(code(&out), 0);
    assert!(
        stderr_of(&out).contains("needs an archive path"),
        "{}",
        stderr_of(&out)
    );
}

#[test]
fn build_reports_the_planned_size_and_where_to_check_it() {
    let fx = Fixture::new();
    let config = fx.good();
    let archive = fx.path("mesh.asmb");
    let out = run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        archive.to_str().unwrap(),
        "--no-embed",
    ]);

    let text = stdout_of(&out);
    // 512 + 4096 + 8192 + 1024 bytes of models, rounded to pages in the archive.
    assert!(
        text.contains("13.50 KiB"),
        "expected the summed size:\n{text}"
    );
    assert!(
        text.contains("21.00 KiB"),
        "expected the page-aligned archive size:\n{text}"
    );
    assert!(text.contains("--verify"), "{text}");
    assert!(text.contains("built"), "{text}");
}

#[test]
fn the_built_archive_contains_the_configured_persona_and_threshold() {
    let fx = Fixture::new();
    let router = fx.gguf("route.gguf", 256);
    let a = fx.gguf("a.gguf", 256);
    let b = fx.gguf("b.gguf", 256);
    let config = fx.path("config.json");
    std::fs::write(
        &config,
        format!(
            r#"{{"assembly_name":"Persona",
                "persona":"answer like a pirate",
                "embedding":{{"name":"route","path":"{}","confidence_threshold":0.33,"fallback_model":"code"}},
                "models":[{{"name":"general","path":"{}","anchors":["chat"]}},
                          {{"name":"code","path":"{}","anchors":["code"]}}]}}"#,
            jp(router.display()),
            jp(a.display()),
            jp(b.display())
        ),
    )
    .unwrap();

    let archive = fx.path("mesh.asmb");
    let out = run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        archive.to_str().unwrap(),
        "--no-embed",
    ]);
    assert_eq!(code(&out), 0, "{}", stderr_of(&out));

    let text = stdout_of(&run(&["info", archive.to_str().unwrap()]));
    assert!(text.contains("answer like a pirate"), "{text}");
    assert!(text.contains("0.33"), "the configured threshold:\n{text}");
    assert!(text.contains("code"), "the fallback:\n{text}");
}

#[test]
fn a_config_pointing_at_a_missing_fallback_is_rejected() {
    let fx = Fixture::new();
    let router = fx.gguf("route.gguf", 256);
    let a = fx.gguf("a.gguf", 256);
    let b = fx.gguf("b.gguf", 256);
    let config = fx.path("config.json");
    std::fs::write(
        &config,
        format!(
            r#"{{"assembly_name":"T",
                "embedding":{{"name":"route","path":"{}","fallback_model":"ghost"}},
                "models":[{{"name":"a","path":"{}","anchors":["x"]}},
                          {{"name":"b","path":"{}","anchors":["x"]}}]}}"#,
            jp(router.display()),
            jp(a.display()),
            jp(b.display())
        ),
    )
    .unwrap();

    let out = run(&["build", config.to_str().unwrap(), "--no-embed"]);
    assert_ne!(code(&out), 0);
    assert!(stderr_of(&out).contains("ghost"), "{}", stderr_of(&out));
}

#[test]
fn build_output_is_byte_identical_across_runs() {
    // Same config twice must produce the same bytes: an archive that differs
    // run to run cannot be checksummed, cached or compared.
    let fx = Fixture::new();
    let config = fx.good();
    let first = fx.path("first.asmb");
    let second = fx.path("second.asmb");

    run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        first.to_str().unwrap(),
        "--no-embed",
    ]);
    run(&[
        "build",
        config.to_str().unwrap(),
        "-o",
        second.to_str().unwrap(),
        "--no-embed",
    ]);

    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap(),
        "two builds of one config produced different archives"
    );
}

#[test]
fn embed_needs_a_model_and_something_to_embed() {
    // Argument errors are worth checking without a GGUF, because these paths are
    // what a mistyped command hits first.
    let out = run(&["embed"]);
    assert_ne!(code(&out), 0, "embed with no arguments should fail");
    assert!(
        stderr_of(&out).contains("model path is required"),
        "{}",
        stderr_of(&out)
    );

    let fx = Fixture::new();
    let model = fx.gguf("router", 64);
    let out = run(&["embed", model.to_str().unwrap()]);
    assert_ne!(code(&out), 0, "embed with no text should fail");
    assert!(
        stderr_of(&out).contains("text to embed is required"),
        "{}",
        stderr_of(&out)
    );
}

#[test]
fn embed_reports_a_file_that_is_not_a_model() {
    // The fake GGUF has a valid magic number but nothing else, so llama.cpp
    // rejects it. What matters is that the CLI says which file failed rather
    // than letting llama.cpp's own message escape unaided.
    let fx = Fixture::new();
    let model = fx.gguf("router", 64);

    let out = run(&["embed", model.to_str().unwrap(), "hello"]);
    assert_ne!(code(&out), 0, "a truncated GGUF should fail");
    let text = stderr_of(&out);
    // The fixture is named `router` with no extension, so the message is checked
    // against what was actually passed rather than a nicer-sounding name.
    assert!(
        text.contains("cannot load") && text.contains("router"),
        "the error should say which file failed to load: {text}"
    );
}

#[test]
fn embed_rejects_an_unknown_pooling_type() {
    let fx = Fixture::new();
    let model = fx.gguf("router", 64);

    let out = run(&[
        "embed",
        model.to_str().unwrap(),
        "hello",
        "--pooling",
        "average",
    ]);
    assert_ne!(code(&out), 0);
    let text = stderr_of(&out);
    assert!(
        text.contains("mean") || text.contains("pooling"),
        "the error should mention pooling: {text}"
    );
}

#[test]
fn run_local_without_an_archive_points_at_the_flag() {
    let out = run(&["run", "--local", "hello"]);
    assert_ne!(
        code(&out),
        0,
        "run --local with nothing to load should fail"
    );
    let text = stderr_of(&out);
    assert!(
        text.contains("--local needs an archive"),
        "the error should name the flag:\n{text}"
    );
}

#[test]
fn run_reports_a_server_that_is_not_there() {
    // Port 1 is never listening, so this is a refusal rather than a probe of
    // whatever happens to be running on the developer's machine.
    let out = run(&["run", "--port", "1", "hello"]);
    assert_ne!(code(&out), 0);
    let text = stderr_of(&out);
    assert!(
        text.contains("no assembly server answered"),
        "the error should say nothing answered:\n{text}"
    );
}

#[test]
fn run_with_a_missing_archive_fails_cleanly() {
    let fx = Fixture::new();
    let missing = fx.path("missing.asmb");
    let out = run(&["run", "--local", &missing.display().to_string(), "hi"]);
    assert_ne!(code(&out), 0);
    let text = stderr_of(&out);
    assert!(
        text.contains("cannot open archive"),
        "the error should name what went wrong:\n{text}"
    );
}

#[test]
fn help_describes_run_as_a_chat() {
    let out = run(&["--help"]);
    let text = stderr_of(&out);
    for expected in ["--server", "--local", "--archive", "Chat interactively"] {
        assert!(text.contains(expected), "help omits {expected}:\n{text}");
    }
}
