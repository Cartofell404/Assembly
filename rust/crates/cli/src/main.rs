//! `assembly` command line entry point.
//!
//! Phase 2 scope: build `.asmb` archives, validate build configs, and read or
//! verify what was built. Inference comes later, behind a llama.cpp binding,
//! and will add `serve` and `run`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::thread;

use anyhow::{bail, Context, Result};
use asmb_format::{Archive, BuildPlan, ModelSource};
use assembly_core::config::{self, ValidationReport};
use assembly_core::engine::{Engine, EngineOptions, LoadOptions};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use llama_engine::LlamaEngine;
use reqwest::blocking::Client;

use std::time::Duration;

/// Name the tool prints in usage and version output.
const NAME: &str = "assembly";

fn print_usage() {
    eprintln!(
        "{NAME} {version}
Route every prompt to the smallest model that can answer it.

Usage:
  {NAME} build <config.json> [-o <archive.asmb>] [--force] [--no-check-magic]
  {NAME} validate <config.json> [-o <archive.asmb>]
  {NAME} info <archive.asmb> [--verify]
  {NAME} embed <model.gguf> <text>
  {NAME} serve <archive.asmb> [--host <ip>] [--port <n>] [--api-key <key>]
                          [--settings <file.json>] [--workers <n>] [--headroom <0-1>]
  {NAME} run [options] [archive.asmb] [prompt ...]

Commands:
  build     Pack the models a config names into a .asmb archive
  validate  Check a config and report every problem found, then stop
  info      List the models inside an archive
  info --verify
            Re-hash every blob and compare against the recorded digests
  embed     Embed text with an embedding model and print the vector
  serve     Serve an archive over HTTP, OpenAI-compatible, on loopback
  serve --info
            Print what a served archive contains, then exit without serving
  run       Chat interactively: with a running server if one is up, or with an
            archive loaded here. The conversation stays open until you quit.

Options:
  -o, --output <path>
            Where build writes the archive. Defaults to the config path with a
            .asmb extension.
      --force
            Replace an archive that already exists
      --no-check-magic
            Skip the GGUF header check, for configs pointing at test fixtures
      --pooling <mean|cls|last|none>
            Pooling for embed. Defaults to mean, which is what sentence-embedding
            models expect.
      --host <ip>
            Address serve/run binds. Defaults to {host}, which keeps the server off
            the network. A non-loopback address requires --api-key.
      --port <n>
            Port serve/run binds. Defaults to {port}. Use 0 to let the OS choose.
      --api-key <key>
            Require this bearer token on every request.
      --settings <file.json>
            Read host, port, api_key, headroom and workers from a JSON file.
            The environment and command-line flags both win over it.
      --workers <n>
            Request threads. Defaults to {workers}.
      --headroom <0-1>
            Share of physical RAM the model cache may use. Defaults to {headroom}.
  -s, --server <url>
            Chat with this server. Without it, run scans ports 8000-8005 for a
            server and uses the first (or the only) one it finds.
      --local
            Never talk to a server; load the archive and answer here.
  -m, --archive <path>
            Archive for local mode. With --server it is a model alias instead,
            because the server chose its archive when it started.
  -h, --help     Show this message
  -V, --version  Show the version
",
        version = asmb_format::VERSION,
        host = assembly_server::DEFAULT_HOST,
        port = assembly_server::DEFAULT_PORT,
        workers = assembly_server::DEFAULT_WORKERS,
        headroom = assembly_server::DEFAULT_HEADROOM,
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args
        .iter()
        .any(|a| a == "--help" || a == "-h" || a == "help")
    {
        print_usage();
        return ExitCode::SUCCESS;
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("{NAME} {}", asmb_format::VERSION);
        return ExitCode::SUCCESS;
    }

    if args.is_empty() {
        print_usage();
        eprintln!("{NAME}: no subcommand given");
        return ExitCode::FAILURE;
    }

    let result = match args[0].as_str() {
        "info" => info(&args[1..]),
        "build" => build(&args[1..]),
        "validate" => validate(&args[1..]),
        "embed" => embed(&args[1..]),
        "serve" => serve(&args[1..]),
        "run" => run_cmd(&args[1..]),
        other => {
            print_usage();
            eprintln!("{NAME}: unknown subcommand '{other}'");
            Err(anyhow::anyhow!("unknown subcommand"))
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{NAME}: {err:#}");
            ExitCode::FAILURE
        }
    }
}
#[derive(Clone, Debug)]
struct ServerInfo {
    url: String,
    name: String,
}

/// Ask a server what archive it is serving, or `None` if nothing answers.
fn health(url: &str) -> Option<String> {
    let client = Client::new();
    let health_url = format!("{}/health", url.trim_end_matches('/'));
    let resp = client
        .get(&health_url)
        .timeout(std::time::Duration::from_millis(500))
        .send()
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body = resp.text().ok()?;
    let v: serde_json::Value = serde_json::from_str(&body).ok()?;
    Some(
        v["assembly"]
            .as_str()
            .or_else(|| v["name"].as_str())
            .unwrap_or("unknown")
            .to_string(),
    )
}

fn discover_servers(ports: &[u16]) -> Vec<ServerInfo> {
    let client = Client::new();
    let mut servers = Vec::new();
    for &port in ports {
        let url = format!("http://127.0.0.1:{}", port);
        let health_url = format!("{}/health", url);
        if let Ok(resp) = client
            .get(&health_url)
            .timeout(std::time::Duration::from_millis(500))
            .send()
        {
            if resp.status().is_success() {
                if let Ok(body) = resp.text() {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                        let name = v["assembly"]
                            .as_str()
                            .or(v["name"].as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        servers.push(ServerInfo { url, name });
                    } else {
                        servers.push(ServerInfo {
                            url,
                            name: "unknown".to_string(),
                        });
                    }
                } else {
                    servers.push(ServerInfo {
                        url,
                        name: "unknown".to_string(),
                    });
                }
            }
        }
    }
    servers.dedup_by(|a, b| a.url == b.url);
    servers
}

/// `run [options] [archive.asmb] [prompt ...]`
///
/// The chat entry point. Without `--local` it looks for a running server and
/// talks to that; with an archive path (or `--local`) it loads the archive and
/// answers here. Either way the answer streams, then the prompt returns.
///
fn run_cmd(args: &[String]) -> Result<()> {
    use std::io::Write;
    use std::path::PathBuf;

    let mut settings = assembly_server::Settings::default();
    let mut settings_file: Option<PathBuf> = None;
    let mut archive: Option<PathBuf> = None;
    let mut archive_flag = false;
    let mut prompt: Option<String> = None;
    let mut positionals: Vec<String> = Vec::new();
    let mut server_url: Option<String> = None;
    let mut port: Option<u16> = None;
    let mut local_only = false;

    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        let value = |i: &mut usize| -> Result<String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))
        };

        match arg {
            "--settings" => settings_file = Some(PathBuf::from(value(&mut i)?)),
            "--archive" | "-m" | "--model" => {
                archive = Some(PathBuf::from(value(&mut i)?));
                archive_flag = true;
            }
            "--server" | "-s" => server_url = Some(value(&mut i)?),
            "--port" => {
                let p = value(&mut i)?;
                port = Some(
                    p.parse()
                        .map_err(|_| anyhow::anyhow!("--port wants a number"))?,
                );
            }
            "--local" => local_only = true,
            "--workers" => {
                let n: usize = value(&mut i)?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--workers wants a number"))?;
                if n == 0 {
                    bail!("--workers must be at least 1");
                }
                settings.workers = n;
            }
            "--headroom" => {
                let n: f64 = value(&mut i)?
                    .parse()
                    .map_err(|_| anyhow::anyhow!("--headroom wants a fraction like 0.8"))?;
                if !(0.0..=1.0).contains(&n) || n == 0.0 {
                    bail!("--headroom must be greater than 0 and at most 1");
                }
                settings.headroom = n;
            }
            other if other.starts_with('-') => bail!("unknown option '{other}' for run"),
            other => {
                positionals.push(other.to_string());
            }
        }
        i += 1;
    }

    if !positionals.is_empty() {
        if !archive_flag && archive.is_none() {
            let first = &positionals[0];
            if first.ends_with(".asmb") || first.contains('/') {
                archive = Some(PathBuf::from(first));
                if positionals.len() > 1 {
                    prompt = Some(positionals[1..].join(" "));
                }
            } else {
                prompt = Some(positionals.join(" "));
            }
        } else if prompt.is_none() {
            prompt = Some(positionals.join(" "));
        }
    }

    if let Some(path) = settings_file {
        settings
            .apply_file(&path)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    settings.apply_env(&|key| std::env::var(key).ok());

    // An archive path on the command line means "use this file", which only
    // local inference can do: the server picked its archive at `serve` time and
    // deliberately accepts no uploads.
    let archive_is_path = archive.as_deref().is_some_and(|p| {
        let s = p.to_string_lossy();
        s.ends_with(".asmb") || s.contains('/')
    });

    let explicit_url = server_url.or_else(|| port.map(|p| format!("http://127.0.0.1:{p}")));
    if local_only {
        // fall through to local mode
    } else if let Some(url) = explicit_url {
        match health(&url) {
            Some(served) => {
                if archive_is_path {
                    eprintln!(
                        "note: ignoring {} -- {url} is already serving '{served}'; pass an alias instead of a path",
                        archive.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
                    );
                }
                let model = archive
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or(served);
                return run_client(&url, Some(model), prompt);
            }
            None => bail!("no assembly server answered at {url}"),
        }
    } else if !archive_is_path {
        let servers = discover_servers(&[8000, 8001, 8002, 8003, 8004, 8005]);
        match servers.len() {
            0 => {}
            1 => {
                let url = servers[0].url.clone();
                let model = archive
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_else(|| servers[0].name.clone());
                return run_client(&url, Some(model), prompt);
            }
            _ => {
                eprintln!("Multiple servers found on localhost:");
                for (i, s) in servers.iter().enumerate() {
                    eprintln!("  [{}] {} ({})", i + 1, s.url, s.name);
                }
                eprintln!("Use --server <url> or --port <n> to pick one.");
                bail!("multiple servers found; specify which one");
            }
        }
    }

    // Local mode
    let archive_path = if let Some(path) = archive {
        path
    } else if let Ok(path) = std::env::var("ASSEMBLY_ARCHIVE") {
        PathBuf::from(path)
    } else if local_only {
        bail!("--local needs an archive to load; pass one, or drop --local to look for a server");
    } else {
        bail!(
            "no archive provided and no server found; use --server, or pass an archive for local mode"
        );
    };

    let engine = Arc::new(LlamaEngine::new());
    let assembly = assembly_core::manager::Assembly::open(
        &archive_path,
        engine as Arc<dyn Engine>,
        settings.headroom,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    // The conversation itself: one message per turn, oldest first, which is
    // what the model actually sees.
    let mut messages = assembly_core::engine::Messages::new();

    println!(
        "chatting locally with '{}' — /clear forgets the conversation, /quit exits",
        assembly.name()
    );
    println!();

    if let Some(prompt) = prompt {
        println!("> {prompt}");
        std::io::stdout().flush()?;
        if let Err(e) = stream_local(&assembly, &mut messages, &prompt) {
            eprintln!("error: {e:#}");
        }
    }
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let line = input.trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "/quit" | "/exit" | "quit" | "exit" => break,
            "/clear" => {
                messages.items.clear();
                println!("(memory cleared)");
                continue;
            }
            _ => {}
        }
        println!();
        if let Err(e) = stream_local(&assembly, &mut messages, line) {
            eprintln!("error: {e:#}");
        }
    }
    Ok(())
}

/// One streamed local turn: append the user's line, stream the answer, then
/// record it so the next turn has it to look back on.
///
/// A failed turn is rolled back to the conversation it came from, so an error
/// does not leave a dangling "user" message for the model to answer twice.
fn stream_local(
    assembly: &assembly_core::manager::Assembly,
    messages: &mut assembly_core::engine::Messages,
    prompt: &str,
) -> Result<()> {
    use assembly_core::engine::Message;
    use std::io::Write;

    messages.push(Message::user(prompt));
    let mut answer = String::new();
    let streamed = assembly.chat_stream(messages, &EngineOptions::default(), &mut |chunk| {
        if !chunk.text.is_empty() {
            answer.push_str(&chunk.text);
            print!("{}", chunk.text);
            let _ = std::io::stdout().flush();
        }
        Ok(())
    });

    println!();
    if let Err(e) = streamed {
        messages.items.pop();
        return Err(e.into());
    }
    messages.push(Message::assistant(answer));
    Ok(())
}

fn build(args: &[String]) -> Result<()> {
    let opts = BuildOpts::parse(args)?;

    // Pre-flight answers "is this config usable" in milliseconds. It runs before
    // anything is copied, so a wrong path costs a keystroke rather than the
    // length of a multi-gigabyte copy.
    let report = config::validate_with(
        &opts.config,
        Some(&opts.output),
        opts.check_magic,
        opts.force,
    )
    .with_context(|| format!("validating {}", opts.config.display()))?;
    print_report(&report);
    if !report.is_valid {
        bail!(
            "{} error(s) in {}; nothing was written",
            report.errors().collect::<Vec<_>>().len(),
            opts.config.display()
        );
    }

    let cfg = report
        .config
        .as_ref()
        .expect("a valid report always carries a config");
    println!("assembly: {}", cfg.assembly_name);

    // Embedding runs before the plan is measured: anchor vectors live in the
    // header, so they change its length and can move the first blob across a
    // page boundary. Measuring first and embedding after would plan against a
    // header that no longer exists.
    let mut embedder = HashEmbedder::default();
    let plan = build_plan_with(cfg, &mut embedder, opts.embed)?;
    let planned = plan.planned_size()?;

    // Vector counts per model, for the summary table. Read back off the plan
    // rather than recomputed, so the table cannot disagree with what was written.
    let anchor_counts: std::collections::HashMap<String, usize> = cfg
        .models
        .iter()
        .map(|m| (m.name.clone(), m.anchors.len()))
        .collect();

    // Sum of model sizes from config (uncompressed, pre-alignment) including router
    let summed_size: u64 = cfg
        .models
        .iter()
        .map(|m| std::fs::metadata(&m.path).map(|md| md.len()).unwrap_or(0))
        .sum::<u64>()
        + std::fs::metadata(&cfg.embedding.path)
            .map(|md| md.len())
            .unwrap_or(0);

    println!(
        "  output   {} ({})",
        opts.output.display(),
        human_bytes(planned as f64)
    );
    println!("  models   {}", cfg.models.len());
    println!("  size (summed)   {}", human_bytes(summed_size as f64));

    // Build with progress bar
    let mp = MultiProgress::new();

    // Progress for embedding (if enabled)
    if opts.embed {
        let _models_with_anchors: Vec<_> = cfg
            .models
            .iter()
            .filter(|m| !m.anchors.is_empty())
            .collect();
        let total_anchors: usize = cfg.models.iter().map(|m| m.anchors.len()).sum();

        if total_anchors > 0 {
            let mp_embed = mp.add(ProgressBar::new(total_anchors as u64));
            mp_embed.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
                    .unwrap()
                    .progress_chars("##-"),
            );
            mp_embed.set_message("Computing anchor vectors");

            // We need to track embedding progress - for now just show a spinner
            let embed_pb = ProgressBar::new_spinner();
            embed_pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} [{elapsed_precise}] {msg}")
                    .unwrap(),
            );
            embed_pb.set_message("Computing anchor vectors...");
            embed_pb.enable_steady_tick(Duration::from_millis(100));

            // The actual embedding happens in build_plan, so we just show a
            // spinner here rather than reporting per-anchor progress.
        }
    }

    let started = std::time::Instant::now();
    let built = plan.build(&opts.output)?;
    let elapsed = started.elapsed();

    println!();
    println!("built {}", opts.output.display());
    println!("  size      {}", human_bytes(built.total_bytes as f64));
    println!("  elapsed   {}", format_duration(elapsed));
    println!();
    println!(
        "{:<14} {:>10} {:>12} {:>9} {:>8}",
        "MODEL", "SIZE", "OFFSET", "ANCHORS", "SHA256"
    );
    for blob in &built.blobs {
        let digest = if blob.sha256.is_empty() {
            "-".to_string()
        } else {
            format!("{}…", &blob.sha256[..8.min(blob.sha256.len())])
        };
        println!(
            "{:<14} {:>10} {:>12} {:>9} {:>8}",
            blob.name,
            human_bytes(blob.size as f64),
            format!("{:#x}", blob.offset),
            anchor_counts
                .get(blob.name.as_str())
                .copied()
                .unwrap_or(0)
                .to_string(),
            digest
        );
    }

    // Only worth saying when it is true. An archive built with --no-embed is
    // complete and valid but can never win a routing decision, and a user who
    // builds one and serves it should not have to read the source to find out.
    if !opts.embed {
        eprintln!(
            "  ! built with --no-embed: {} stores anchor phrases but no vectors,",
            opts.output.display()
        );
        eprintln!("    so it cannot route. Rebuild without --no-embed to serve it.");
    }
    println!();
    println!(
        "run `{NAME} info {} --verify` to check it",
        opts.output.display()
    );
    Ok(())
}

/// `validate [options] <config.json>`
fn validate(args: &[String]) -> Result<()> {
    let opts = BuildOpts::parse(args)?;
    let report = config::validate_with(
        &opts.config,
        Some(&opts.output),
        opts.check_magic,
        opts.force,
    )
    .with_context(|| format!("validating {}", opts.config.display()))?;
    print_report(&report);

    if !report.is_valid {
        bail!(
            "{} error(s) in {}",
            report.errors().collect::<Vec<_>>().len(),
            opts.config.display()
        );
    }
    println!("{} is valid", opts.config.display());
    Ok(())
}

/// Shared flags for `build` and `validate`.
#[derive(Debug)]
struct BuildOpts {
    config: PathBuf,
    output: PathBuf,
    check_magic: bool,
    /// Replace an archive that already exists.
    force: bool,
    /// Compute anchor vectors. On by default because an archive without them
    /// cannot route, which is most of what Assembly is for.
    embed: bool,
}

impl BuildOpts {
    /// Parse the flags both subcommands accept.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown flag, a missing value, or no config path.
    fn parse(args: &[String]) -> Result<BuildOpts> {
        let mut config: Option<PathBuf> = None;
        let mut output: Option<PathBuf> = None;
        let mut check_magic = true;
        let mut force = false;
        let mut embed = true;

        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "-o" | "--output" => {
                    let Some(value) = args.get(i + 1) else {
                        bail!("{} needs a path", args[i]);
                    };
                    output = Some(PathBuf::from(value));
                    i += 2;
                }
                "--force" => {
                    force = true;
                    i += 1;
                }
                "--no-check-magic" => {
                    check_magic = false;
                    i += 1;
                }
                "--no-embed" => {
                    embed = false;
                    i += 1;
                }
                other if other.starts_with('-') => {
                    bail!("unknown option '{other}' for this command");
                }
                other => {
                    config = Some(PathBuf::from(other));
                    i += 1;
                }
            }
        }

        let config = config.context("a config path is required")?;
        Ok(BuildOpts {
            output: output.unwrap_or_else(|| default_output(&config)),
            config,
            check_magic,
            force,
            embed,
        })
    }
}

/// Options for `serve`, after parsing.
#[derive(Debug, Clone, PartialEq)]
struct ServeOpts {
    archive: PathBuf,
    settings: assembly_server::Settings,
    /// Print what the archive holds and exit, without listening.
    info_only: bool,
}

/// `serve [options] <archive.asmb>`
///
/// The inference entry point. Everything before this -- `build`, `info` -- works
/// without a model in memory; this is the part that loads one.
fn serve(args: &[String]) -> Result<()> {
    let opts = ServeOpts::parse(args)?;
    let engine = Arc::new(LlamaEngine::new());
    let assembly = assembly_core::manager::Assembly::open(
        &opts.archive,
        engine as Arc<dyn Engine>,
        opts.settings.headroom,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    if opts.info_only {
        print_serve_info(&assembly, &opts.settings);
        return Ok(());
    }

    // Refusing here, rather than starting and logging a warning, is deliberate:
    // a server reachable from the network with no authentication is a mistake
    // that should not need a second command to notice.
    if opts.settings.is_public() && opts.settings.api_key.is_none() {
        bail!(
            "refusing to serve on {} with no --api-key; anyone who can reach this \
             machine could read the conversation and spend the machine's memory. \
             Pass --api-key <key>, or bind 127.0.0.1 to keep it local.",
            opts.settings.host
        );
    }

    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let running = assembly_server::serve(
        assembly.clone(),
        opts.settings.clone(),
        Arc::clone(&shutdown),
    )?;

    eprintln!(
        "{NAME}: serving '{}' on http://{}",
        assembly.name(),
        opts.settings.host
    );
    for (method, path) in assembly_server::ENDPOINTS {
        eprintln!("  {method:<5} {path}");
    }
    if let Some(key) = &opts.settings.api_key {
        // Never print the key itself, not even truncated: a terminal scrollback
        // is a poor place to store a credential.
        eprintln!("  auth: bearer token required ({} characters)", key.len());
    } else {
        eprintln!("  auth: none (loopback only)");
    }
    eprintln!(
        "  cache: {:.0}% of RAM, up to {} models",
        opts.settings.headroom * 100.0,
        assembly_server::DEFAULT_MODEL_CAP,
    );
    eprintln!("{NAME}: press Ctrl-C to stop");

    // Workers poll the flag once a second, so this returns promptly without
    // needing a signal handler -- which is the part of a terminal program that
    // usually needs unsafe code to get right.
    while !shutdown.load(std::sync::atomic::Ordering::SeqCst) {
        thread::sleep(std::time::Duration::from_millis(200));
    }
    running.shutdown();
    Ok(())
}

/// Print what a served archive contains.
fn print_serve_info(
    assembly: &assembly_core::manager::Assembly,
    settings: &assembly_server::Settings,
) {
    let header = assembly.header();
    println!("archive:      {}", assembly.path().display());
    println!("name:         {}", assembly.name());
    println!("persona:      {}", assembly.persona());
    println!(
        "router:       {} (threshold {:.2})",
        header.embedding.name, header.embedding.confidence_threshold
    );
    if let Some(fallback) = &header.embedding.fallback_model {
        println!("fallback:     {fallback}");
    }
    println!("models:");
    for m in &header.models {
        let vectors = m.anchor_vectors().map(|v| v.len()).unwrap_or(0);
        let routable = if assembly.routable_models().iter().any(|n| n == &m.name) {
            "routable"
        } else {
            "no anchors"
        };
        println!(
            "  {:<16} {:>10}  {} anchors, {} vectors, {}d  {}",
            m.name,
            human_bytes(m.size as f64),
            m.anchors.len(),
            vectors,
            m.anchor_dim,
            routable
        );
    }
    println!("endpoints:");
    for (method, path) in assembly_server::ENDPOINTS {
        println!("  {method:<5} {path}");
    }
    let bind = format!("{}:{}", settings.host, settings.port);
    println!("would serve:  http://{bind}");
    println!(
        "cache:        {:.0}% of RAM, up to {} models",
        settings.headroom * 100.0,
        assembly_server::DEFAULT_MODEL_CAP
    );
}

fn human_bytes(bytes: f64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

/// Returns the first line of a string, or the whole string if no newline.
#[allow(dead_code)]
fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or(s)
}

/// A duration as a human string, switching units so it stays readable from
/// milliseconds to hours.
fn format_duration(elapsed: std::time::Duration) -> String {
    let secs = elapsed.as_secs_f64();
    if secs < 1.0 {
        return format!("{:.0} ms", elapsed.as_secs_f64() * 1000.0);
    }
    if secs < 60.0 {
        return format!("{secs:.1} s");
    }
    let mins = (secs / 60.0).floor();
    let rest = secs - mins * 60.0;
    if mins < 60.0 {
        return format!("{mins:.0} m {rest:.0} s");
    }
    let hours = (mins / 60.0).floor();
    format!("{hours:.0} h {:.0} m", mins - hours * 60.0)
}

/// `info [options] <archive.asmb>`
///
/// Lists the models inside an archive. The format is deliberately spare: paths,
/// sizes, anchors, digests. It is meant for a human scanning to see if the right
/// models ended up in the right place, or for a script to parse with `cut` or
/// `awk`.
fn info(args: &[String]) -> Result<()> {
    let mut path: Option<PathBuf> = None;
    let mut verify = false;

    for arg in args {
        match arg.as_str() {
            "--verify" => verify = true,
            other if other.starts_with('-') => bail!("unknown option '{other}' for info"),
            other => {
                if path.is_some() {
                    bail!("info takes one archive, but also got '{other}'");
                }
                path = Some(PathBuf::from(other));
            }
        }
    }

    let path = path.ok_or_else(|| anyhow::anyhow!("info needs an archive path"))?;
    let mut archive =
        Archive::open(&path).with_context(|| format!("reading {}", path.display()))?;

    if verify {
        let report = archive
            .verify()
            .with_context(|| format!("verifying {}", path.display()))?;
        if report.is_valid() {
            println!("all blobs match");
            for blob in &report.blobs {
                println!(" {} ok", blob.name);
            }
        } else {
            for blob in &report.blobs {
                if !blob.ok {
                    println!(" {} MISMATCH", blob.name);
                }
            }
            bail!("verification failed for {}", path.display());
        }
        return Ok(());
    }

    // The header has enough to print a table without touching any blobs.
    let header = archive.header();

    println!("archive   {}", path.display());
    println!("name      {}", header.assembly_name);
    println!("persona   {}", header.persona.as_deref().unwrap_or(""));
    println!(
        "router    {} (threshold {:.2})",
        header.embedding.name, header.embedding.confidence_threshold
    );
    if let Some(fallback) = &header.embedding.fallback_model {
        println!("fallback  {}", fallback);
    }

    println!();
    println!(
        "{:<14} {:>10} {:>12} {:>9} {:>9}",
        "MODEL", "SIZE", "OFFSET", "ANCHORS", "VECTORS"
    );
    for m in &header.models {
        let phrase_count = m.anchors.len();
        let vectors = m.anchor_vectors().map(|v| v.len()).unwrap_or(0);
        println!(
            "{:<14} {:>10} {:>12} {:>9} {:>9}",
            m.name,
            human_bytes(m.size as f64),
            m.offset,
            phrase_count,
            if vectors == 0 {
                "-".to_string()
            } else {
                vectors.to_string()
            }
        );
    }
    Ok(())
}

/// Print a validation report the same way build prints one.
fn print_report(report: &ValidationReport) {
    let warnings: Vec<String> = report.warnings().map(|x| x.to_string()).collect();
    if !warnings.is_empty() {
        eprintln!("assembly: {} warning(s):", warnings.len());
        for w in &warnings {
            eprintln!("  - {w}");
        }
    }
    let errors: Vec<String> = report.errors().map(|x| x.to_string()).collect();
    if errors.is_empty() {
        eprintln!("assembly: config looks good");
        return;
    }
    eprintln!("assembly: {} error(s):", errors.len());
    for e in &errors {
        eprintln!("  - {e}");
    }
}

/// Build a plan from a config without running embedding.
#[allow(dead_code)]
fn build_plan_without_vectors(cfg: &config::AssemblyConfig) -> Result<BuildPlan> {
    build_plan_with(cfg, &mut NoopEmbedder { embed: false }, false)
}

trait Embedder {
    fn embed(&mut self, _path: &Path, _phrase: &str) -> Result<Vec<f32>>;
}

/// Deterministic hash-based embedder for testing.
struct HashEmbedder {
    dim: usize,
}

impl Embedder for HashEmbedder {
    fn embed(&mut self, _path: &Path, phrase: &str) -> Result<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        phrase.hash(&mut hasher);
        let seed = hasher.finish();

        // Generate deterministic vector based on phrase hash
        let mut vec = Vec::with_capacity(self.dim);
        let mut state = seed;
        for _ in 0..self.dim {
            // Simple LCG for deterministic pseudo-random generation
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            vec.push(((state >> 16) & 0x7FFF) as f32 / 32768.0);
        }
        Ok(vec)
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        HashEmbedder { dim: 8 }
    }
}

struct NoopEmbedder {
    embed: bool,
}

impl Embedder for NoopEmbedder {
    fn embed(&mut self, _path: &Path, _phrase: &str) -> Result<Vec<f32>> {
        if !self.embed {
            bail!("embedding not requested")
        }
        Ok(vec![0.0f32; 1])
    }
}

/// Build a plan using the provided embedder.
fn build_plan_with<E>(
    cfg: &config::AssemblyConfig,
    embedder: &mut E,
    embed: bool,
) -> Result<BuildPlan>
where
    E: Embedder,
{
    use asmb_format::BlobSource;

    // The embedding model (router) is defined in cfg.embedding, not in cfg.models
    // Add it as an embedding blob source
    let router_blob = BlobSource::new(cfg.embedding.name.clone(), cfg.embedding.path.clone());

    let mut plan = BuildPlan::new(cfg.assembly_name.clone(), Some(router_blob));
    let router_cfg = cfg.embedding.clone();

    for m in &cfg.models {
        let _is_router_model = m.name == cfg.embedding.name;
        // Compute vectors for all models that have anchors (one vector per anchor phrase)
        // Models without anchors cannot be routed to, so no vectors needed.
        let needs_vectors = !m.anchors.is_empty();

        let blob = BlobSource::new(m.name.clone(), m.path.clone());
        let mut ms = ModelSource::new(blob, m.anchors.clone());

        if needs_vectors && embed {
            let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(m.anchors.len());
            for anchor in &m.anchors {
                vectors.push(embedder.embed(&m.path, anchor)?);
            }
            ms = ms.with_vectors(vectors);
        }

        plan = plan.with_model(ms);
    }

    // Validate anchor vectors if any were computed
    if plan.models.iter().any(|m| !m.anchor_vectors.is_empty()) {
        // Check 1: No zero-width vectors
        for model in &plan.models {
            for (i, vec) in model.anchor_vectors.iter().enumerate() {
                if vec.is_empty() {
                    bail!(
                        "anchor dimension mismatch: model '{}' anchor {} has zero width (zero-width anchors are not allowed)",
                        model.blob.name, i
                    );
                }
            }
        }

        // Check 2: All anchor vectors must have the same dimension
        let expected_dim = plan
            .models
            .iter()
            .find(|m| !m.anchor_vectors.is_empty())
            .and_then(|m| m.anchor_vectors.first())
            .map(|v| v.len());

        if let Some(expected) = expected_dim {
            for model in &plan.models {
                for (i, vec) in model.anchor_vectors.iter().enumerate() {
                    if vec.len() != expected {
                        bail!(
                            "anchor dimension mismatch: model '{}' anchor {} has dimension {} but expected {} (mixed width not allowed)",
                            model.blob.name, i, vec.len(), expected
                        );
                    }
                }
            }
        }
    }

    plan = plan.with_router(asmb_format::RouterSettings {
        confidence_threshold: router_cfg.confidence_threshold,
        fallback_model: router_cfg.fallback_model.clone(),
    });
    plan = plan.with_persona(cfg.persona.clone());
    Ok(plan)
}

/// `embed [options] <model.gguf> <text>`
///
/// The smallest thing that proves the llama.cpp binding works from the command
/// line: load a model, embed a phrase, print the vector.
fn embed(args: &[String]) -> Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut text: Option<String> = None;
    let mut pooling = String::from("mean");

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "--pooling" => {
                i += 1;
                pooling = args.get(i).cloned().unwrap_or_default();
            }
            other if other.starts_with('-') => bail!("unknown option '{other}' for embed"),
            other => {
                if model.is_none() {
                    model = Some(PathBuf::from(other));
                } else if text.is_none() {
                    text = Some(other.to_string());
                } else {
                    bail!("embed takes a model and text, but also got '{other}'");
                }
            }
        }
        i += 1;
    }

    let model_path = model.ok_or_else(|| anyhow::anyhow!("model path is required"))?;
    let text = text.ok_or_else(|| anyhow::anyhow!("text to embed is required"))?;

    let engine = LlamaEngine::new();

    let pooling_lower = pooling.to_lowercase();
    match pooling_lower.as_str() {
        "mean" | "cls" | "last" | "none" => {}
        other => bail!("unknown pooling type '{other}'; use mean, cls, last, or none"),
    }

    let mut options = LoadOptions::embed();
    options = options.extra("pooling", pooling_lower);

    let handle = engine
        .load(&model_path, &options)
        .map_err(|e| anyhow::anyhow!("cannot load {}: {}", model_path.display(), e))?;

    let vec = engine
        .embed(handle.as_ref(), &text)
        .map_err(|e| anyhow::anyhow!("cannot embed: {}", e))?;

    println!("{:?}", vec);
    Ok(())
}
fn default_output(config: &Path) -> PathBuf {
    let mut stem = config
        .file_stem()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| "assembly".into());
    stem.push(".asmb");
    config.with_file_name(stem)
}

impl ServeOpts {
    fn parse(args: &[String]) -> Result<ServeOpts> {
        let mut archive: Option<PathBuf> = None;
        let mut settings = assembly_server::Settings::default();
        let mut settings_file: Option<PathBuf> = None;
        let mut info_only = false;
        let mut flags: Vec<(String, String)> = Vec::new();

        let mut i = 0;
        while i < args.len() {
            let arg = args[i].as_str();
            let value = |i: &mut usize| -> Result<String> {
                *i += 1;
                args.get(*i)
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))
            };

            match arg {
                "--info" => info_only = true,
                "--settings" => settings_file = Some(PathBuf::from(value(&mut i)?)),
                "--host" => flags.push(("host".to_string(), value(&mut i)?)),
                "--port" => flags.push(("port".to_string(), value(&mut i)?)),
                "--api-key" => flags.push(("api_key".to_string(), value(&mut i)?)),
                "--workers" => flags.push(("workers".to_string(), value(&mut i)?)),
                "--headroom" => flags.push(("headroom".to_string(), value(&mut i)?)),
                other if other.starts_with('-') => bail!("unknown option '{other}' for serve"),
                other => {
                    if archive.is_some() {
                        bail!("serve takes one archive, but also got '{other}'");
                    }
                    archive = Some(PathBuf::from(other));
                }
            }
            i += 1;
        }

        let archive = archive.ok_or_else(|| anyhow::anyhow!("serve needs an archive path"))?;

        if let Some(path) = settings_file {
            settings
                .apply_file(&path)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        settings.apply_env(&|key| std::env::var(key).ok());

        for (name, raw) in flags {
            match name.as_str() {
                "host" => settings.host = raw,
                "api_key" => settings.api_key = Some(raw),
                "port" => {
                    settings.port = raw
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--port wants a number"))?
                }
                "workers" => {
                    let n: usize = raw
                        .parse()
                        .map_err(|_| anyhow::anyhow!("--workers wants a number"))?;
                    if n == 0 {
                        bail!("--workers must be at least 1");
                    }
                    settings.workers = n;
                }
                "headroom" => {
                    let n: f64 = raw.parse().map_err(|_| {
                        anyhow::anyhow!("--headroom wants a fraction like 0.8, got '{raw}'")
                    })?;
                    if !(0.0..=1.0).contains(&n) || n == 0.0 {
                        bail!("--headroom must be greater than 0 and at most 1");
                    }
                    settings.headroom = n;
                }
                _ => unreachable!("flag names are matched above"),
            }
        }

        Ok(ServeOpts {
            archive,
            settings,
            info_only,
        })
    }
}

fn run_client(
    server_url: &str,
    model_opt: Option<String>,
    prompt_opt: Option<String>,
) -> Result<()> {
    use std::io::Write;

    let client = Client::new();
    let chat_url = format!("{}/v1/chat/completions", server_url.trim_end_matches('/'));
    let model = model_opt.unwrap_or_default();
    let mut history: Vec<serde_json::Value> = Vec::new();

    fn stream_once(
        client: &Client,
        chat_url: &str,
        model: &str,
        history: &mut Vec<serde_json::Value>,
        prompt: &str,
    ) -> Result<()> {
        use std::io::{Read, Write};

        history.push(serde_json::json!({"role": "user", "content": prompt}));
        let body = serde_json::json!({
            "model": model,
            "messages": history,
            "stream": true,
        });

        let mut resp = client
            .post(chat_url)
            .json(&body)
            .send()
            // `.context` rather than a formatted `anyhow!`: reqwest's own
            // message ("connection refused", "connection reset", a timeout) is
            // the source of this error, and formatting it away leaves nothing
            // to diagnose a dead server with.
            .context("failed to reach the server")?;
        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            bail!("server error: {}", text);
        }

        let mut stdout = std::io::stdout();
        let mut buffer: Vec<u8> = Vec::new();
        let mut answer = String::new();

        'stream: loop {
            let mut chunk = [0u8; 4096];
            match resp.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    buffer.extend_from_slice(&chunk[..n]);
                    let sofar = String::from_utf8_lossy(&buffer).to_string();
                    if let Some(last) = sofar.rfind("\n\n") {
                        let events = sofar[..last].to_string();
                        buffer.drain(..last + 2);
                        for line in events.split('\n') {
                            let line = line.trim();
                            // An SSE comment: the server's own note about what
                            // it is doing (a cold model load, for instance).
                            // Invisible to every other OpenAI client, which is
                            // the point of putting it here.
                            if let Some(note) = line.strip_prefix(": ") {
                                if !note.is_empty() {
                                    println!("… {note}");
                                    let _ = stdout.flush();
                                }
                                continue;
                            }
                            let Some(data) = line.strip_prefix("data: ") else {
                                continue;
                            };
                            if data == "[DONE]" {
                                break 'stream;
                            }
                            let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
                                continue;
                            };
                            if let Some(text) = v["choices"][0]["delta"]["content"].as_str() {
                                answer.push_str(text);
                                write!(stdout, "{}", text)?;
                                stdout.flush()?;
                            }
                        }
                    }
                }
                Err(_) => break,
            }
        }
        println!();
        history.push(serde_json::json!({"role": "assistant", "content": answer}));
        Ok(())
    }

    let who = if model.is_empty() {
        server_url.to_string()
    } else {
        model.clone()
    };

    println!("chatting with {who} — /clear forgets the conversation, /quit exits");
    println!();

    if let Some(prompt) = prompt_opt {
        println!("> {prompt}");
        std::io::stdout().flush()?;
        if let Err(e) = stream_once(&client, &chat_url, &model, &mut history, &prompt) {
            report_failure(&e, server_url, &mut history);
        }
    }

    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let line = input.trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "/quit" | "/exit" | "quit" | "exit" => break,
            "/clear" => {
                history.clear();
                println!("(memory cleared)");
                continue;
            }
            _ => {}
        }
        if let Err(e) = stream_once(&client, &chat_url, &model, &mut history, line) {
            report_failure(&e, server_url, &mut history);
        }
    }
    Ok(())
}

/// Report a turn that never made it back.
///
/// The whole chain is printed, not just the head of it: reqwest says "error
/// sending request" for a refused connection, a reset in the middle of an
/// answer, and a timeout alike, and those three mean very different things.
///
/// Then the server is asked whether it is still there. If it is not, that is
/// the actual answer -- the model switch took the process down -- and the
/// person at the terminal needs to know where to look rather than to keep
/// typing into a prompt that will never answer.
fn report_failure(e: &anyhow::Error, server_url: &str, history: &mut Vec<serde_json::Value>) {
    if matches!(
        history.last().and_then(|m| m["role"].as_str()),
        Some("user")
    ) {
        history.pop();
    }
    eprintln!("error: {e:#}");
    if health(server_url).is_none() {
        eprintln!(
            "the server at {server_url} stopped answering -- it may have died loading a model.\n\
             Check the terminal running `assembly serve`, then /quit and run `assembly run` again."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_byte_counts() {
        assert_eq!(human_bytes(0.0), "0 B");
        assert_eq!(human_bytes(512.0), "512 B");
        assert_eq!(human_bytes(1024.0), "1.00 KiB");
        assert_eq!(human_bytes(1536.0), "1.50 KiB");
        assert_eq!(human_bytes(1024.0 * 1024.0), "1.00 MiB");
        assert_eq!(human_bytes(11.1138 * 1024.0 * 1024.0 * 1024.0), "11.11 GiB");
    }

    #[test]
    fn takes_the_first_line_of_a_persona() {
        assert_eq!(first_line("one\ntwo\nthree"), "one");
        assert_eq!(first_line(""), "");
    }

    #[test]
    fn output_defaults_beside_the_config() {
        assert_eq!(
            default_output(Path::new("/models/mesh.json")),
            PathBuf::from("/models/mesh.asmb")
        );
        // A config with no extension still gets a usable name.
        assert_eq!(
            default_output(Path::new("/models/mesh")),
            PathBuf::from("/models/mesh.asmb")
        );
    }

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_a_bare_config() {
        let opts = BuildOpts::parse(&args(&["mesh.json"])).unwrap();
        assert_eq!(opts.config, PathBuf::from("mesh.json"));
        assert_eq!(opts.output, PathBuf::from("mesh.asmb"));
        assert!(opts.check_magic);
    }

    #[test]
    fn parses_an_explicit_output() {
        let opts = BuildOpts::parse(&args(&["mesh.json", "-o", "/tmp/out.asmb"])).unwrap();
        assert_eq!(opts.output, PathBuf::from("/tmp/out.asmb"));
    }

    #[test]
    fn parses_the_long_output_flag() {
        let opts = BuildOpts::parse(&args(&["--output", "/tmp/out.asmb", "mesh.json"])).unwrap();
        assert_eq!(opts.output, PathBuf::from("/tmp/out.asmb"));
        assert_eq!(opts.config, PathBuf::from("mesh.json"));
    }

    #[test]
    fn magic_check_can_be_disabled() {
        let opts = BuildOpts::parse(&args(&["--no-check-magic", "mesh.json"])).unwrap();
        assert!(!opts.check_magic);
    }

    #[test]
    fn force_is_accepted() {
        let opts = BuildOpts::parse(&args(&["--force", "mesh.json"])).unwrap();
        assert_eq!(opts.config, PathBuf::from("mesh.json"));
    }

    #[test]
    fn a_missing_config_path_is_an_error() {
        assert!(BuildOpts::parse(&args(&[])).is_err());
    }

    #[test]
    fn an_unknown_flag_is_an_error() {
        let err = BuildOpts::parse(&args(&["mesh.json", "--nope"])).unwrap_err();
        assert!(err.to_string().contains("--nope"), "got {err}");
    }

    #[test]
    fn a_flag_missing_its_value_is_an_error() {
        assert!(BuildOpts::parse(&args(&["mesh.json", "-o"])).is_err());
    }

    #[test]
    fn formats_durations_across_scales() {
        use std::time::Duration;
        assert_eq!(format_duration(Duration::from_millis(4)), "4 ms");
        assert_eq!(format_duration(Duration::from_millis(1500)), "1.5 s");
        assert_eq!(format_duration(Duration::from_secs(75)), "1 m 15 s");
        assert_eq!(format_duration(Duration::from_secs(3900)), "1 h 5 m");
    }

    // ── anchor vectors ──────────────────────────────────────────────────

    /// A config with two models and a router model, all pointing at paths that
    /// need not exist: the embedder is fake, so nothing is read.
    fn config_with_anchors() -> assembly_core::AssemblyConfig {
        assembly_core::AssemblyConfig {
            assembly_name: "test".into(),
            embedding: assembly_core::EmbeddingConfig {
                name: "router".into(),
                path: PathBuf::from("/models/router.gguf"),
                confidence_threshold: 0.5,
                fallback_model: None,
            },
            models: vec![
                assembly_core::ModelConfig {
                    name: "alpha".into(),
                    path: PathBuf::from("/models/alpha.gguf"),
                    anchors: vec!["first phrase".into(), "second phrase".into()],
                },
                assembly_core::ModelConfig {
                    name: "beta".into(),
                    path: PathBuf::from("/models/beta.gguf"),
                    anchors: vec!["only phrase".into()],
                },
            ],
            persona: None,
        }
    }

    /// An embedder that returns a fixed width, so tests can make every model
    /// agree or deliberately disagree.
    struct FixedWidth {
        width: usize,
    }

    impl Embedder for FixedWidth {
        fn embed(&mut self, _path: &Path, _phrase: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0; self.width])
        }
    }

    #[test]
    fn attaches_a_vector_per_anchor_phrase() {
        let cfg = config_with_anchors();
        let plan = build_plan_with(&cfg, &mut FixedWidth { width: 4 }, true).unwrap();

        // Two models: one with two phrases, one with one.
        assert_eq!(plan.models.len(), 2);
        assert_eq!(plan.models[0].anchor_vectors.len(), 2);
        assert_eq!(plan.models[1].anchor_vectors.len(), 1);
        assert_eq!(plan.models[0].anchor_vectors[0].len(), 4);
    }

    #[test]
    fn every_model_gets_the_same_anchor_width() {
        let cfg = config_with_anchors();
        let plan = build_plan_with(&cfg, &mut FixedWidth { width: 8 }, true).unwrap();

        let widths: Vec<usize> = plan
            .models
            .iter()
            .map(|m| m.anchor_vectors[0].len())
            .collect();
        assert_eq!(widths, vec![8, 8]);
    }

    /// A router scores a prompt against anchors of the same width only. A model
    /// whose anchors disagree is skipped forever and silently, so the build has
    /// to refuse instead.
    #[test]
    fn rejects_anchors_of_mixed_width() {
        struct Ragged;
        impl Embedder for Ragged {
            fn embed(&mut self, _path: &Path, phrase: &str) -> Result<Vec<f32>> {
                Ok(vec![1.0; if phrase.contains("second") { 8 } else { 4 }])
            }
        }

        let cfg = config_with_anchors();
        let err = build_plan_with(&cfg, &mut Ragged, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("mixed width"), "got {err}");
    }

    #[test]
    fn rejects_zero_width_anchors() {
        let cfg = config_with_anchors();
        let err = build_plan_with(&cfg, &mut FixedWidth { width: 0 }, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("zero-width"), "got {err}");
    }

    #[test]
    fn no_embed_leaves_vectors_empty_so_the_archive_cannot_route() {
        let cfg = config_with_anchors();
        let plan = build_plan_without_vectors(&cfg).unwrap();

        // Phrases survive; vectors do not. This is the archive that cannot win a
        // routing decision, and `info` reports it as having no anchors.
        assert_eq!(plan.models[0].anchors.len(), 2);
        assert!(plan.models.iter().all(|m| m.anchor_vectors.is_empty()));
    }

    #[test]
    fn no_embed_is_off_by_default() {
        // An archive without vectors cannot route, which is most of what
        // Assembly is for, so embedding has to be what happens when the user
        // says nothing.
        let opts = BuildOpts::parse(&args(&["mesh.json"])).unwrap();
        assert!(opts.embed, "embedding should default to on");
    }

    #[test]
    fn no_embed_flag_turns_embedding_off() {
        let opts = BuildOpts::parse(&args(&["mesh.json", "--no-embed"])).unwrap();
        assert!(!opts.embed);
    }

    #[test]
    fn hash_embedder_is_deterministic_and_distinguishes_phrases() {
        // Used by the format tests to tell "vectors were attached" from
        // "vectors were left empty", so it has to depend on the phrase and give
        // the same answer twice.
        let mut a = HashEmbedder::default();
        let mut b = HashEmbedder::default();
        let first = a.embed(Path::new("/x"), "hello").unwrap();
        let again = b.embed(Path::new("/x"), "hello").unwrap();
        let other = a.embed(Path::new("/x"), "goodbye").unwrap();

        assert_eq!(first, again);
        assert_ne!(first, other);
        assert!(
            first.iter().any(|v| *v > 0.0),
            "all-zero would score nothing"
        );
    }
}
