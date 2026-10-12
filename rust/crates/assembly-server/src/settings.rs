//! Server settings, and the precedence rules that decide who wins.
//!
//! Four layers can set the same knob, from least to most specific:
//!
//! 1. the defaults here,
//! 2. a JSON settings file,
//! 3. environment variables,
//! 4. command-line flags.
//!
//! The order matters more than it looks. A file is something a user writes once
//! and forgets; a flag is something they type *now* while thinking about a
//! problem, so a flag must win. Environment sits between them because it is how
//! a container gets configured and should override a baked-in file without
//! needing a flag wired into the entrypoint.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The port used when nothing says otherwise.
pub const DEFAULT_PORT: u16 = 8000;

/// The address used when nothing says otherwise.
pub const DEFAULT_HOST: &str = "127.0.0.1";

/// Largest request body accepted, in bytes.
///
/// One mebibyte is far more than any chat request needs -- even a long
/// conversation with a large system prompt sits well inside it -- and small
/// enough that a hostile client cannot make the server allocate a gigabyte by
/// lying in a header.
pub const MAX_BODY_BYTES: usize = 1 << 20;

/// Default share of physical RAM the resident cache may use.
///
/// Eighty percent leaves room for the operating system, the HTTP workers and
/// whatever else the user is running. The router and execution models are large
/// enough that the difference between 80% and 95% is the difference between a
/// working machine and a swap storm.
pub const DEFAULT_HEADROOM: f64 = 0.8;

/// How many workers handle requests.
///
/// Requests are slow -- generation takes seconds -- so a small pool is right.
/// More threads than this mostly means more models loaded at once, which is the
/// memory budget's problem, not the server's.
pub const DEFAULT_WORKERS: usize = 4;

/// How long a worker waits for a request before checking whether the server is
/// shutting down, in seconds.
pub const SHUTDOWN_POLL_SECS: u64 = 1;

/// Everything the server needs to start.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// Address to bind.
    pub host: String,
    /// Port to bind.
    pub port: u16,
    /// Bearer token required on every request, if any.
    pub api_key: Option<String>,
    /// Share of physical RAM the resident cache may use, in `0.0..=1.0`.
    pub headroom: f64,
    /// Worker threads.
    pub workers: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            host: DEFAULT_HOST.to_string(),
            port: DEFAULT_PORT,
            api_key: None,
            headroom: DEFAULT_HEADROOM,
            workers: DEFAULT_WORKERS,
        }
    }
}

impl Settings {
    /// Merge a settings file over the current values.
    ///
    /// Absent keys are left alone, so a file naming only `port` does not reset
    /// the host. Returns `Ok(None)` when the file does not exist, which lets the
    /// CLI accept an optional `--settings` without a special case.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsError::Io`] or [`SettingsError::Parse`] if the file is
    /// named but cannot be read or understood. A file that exists but is broken
    /// is an error rather than a silent fallback: quietly serving on defaults
    /// after the user configured something else is the worse failure.
    pub fn apply_file(&mut self, path: impl AsRef<Path>) -> Result<(), SettingsError> {
        let path = path.as_ref();
        let raw = match std::fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                return Err(SettingsError::Io {
                    path: path.into(),
                    source: e,
                })
            }
        };
        let file: FileSettings = serde_json::from_str(&raw).map_err(|e| SettingsError::Parse {
            path: path.into(),
            reason: e.to_string(),
        })?;
        file.apply(self);
        Ok(())
    }

    /// Merge environment variables over the current values.
    ///
    /// A variable that is set but empty is ignored rather than treated as an
    /// empty value, because `--env ""` in a shell is far more often "no opinion"
    /// than a deliberate request for an empty string.
    pub fn apply_env(&mut self, env: &dyn Fn(&str) -> Option<String>) {
        if let Some(host) = non_empty(env, "ASSEMBLY_HOST") {
            self.host = host;
        }
        if let Some(port) = non_empty(env, "ASSEMBLY_PORT") {
            match port.parse() {
                Ok(port) => self.port = port,
                Err(_) => {
                    // Ignored rather than fatal: a stray port in the environment
                    // should not stop a server that would otherwise work.
                }
            }
        }
        if let Some(key) = non_empty(env, "ASSEMBLY_API_KEY") {
            self.api_key = Some(key);
        }
        if let Some(raw) = non_empty(env, "ASSEMBLY_HEADROOM") {
            if let Ok(headroom) = raw.parse() {
                if (0.0..=1.0).contains(&headroom) {
                    self.headroom = headroom;
                }
            }
        }
        if let Some(raw) = non_empty(env, "ASSEMBLY_WORKERS") {
            if let Ok(workers) = raw.parse::<usize>() {
                if workers > 0 {
                    self.workers = workers;
                }
            }
        }
    }

    /// The address to bind, with the host resolved.
    ///
    /// # Errors
    ///
    /// Returns [`SettingsError::BadHost`] when the host is not an IP address.
    /// Only IPs are accepted: a hostname would mean a DNS lookup at startup, and
    /// a lookup that resolves somewhere unexpected is a way to bind a public
    /// interface by accident.
    pub fn bind_addr(&self) -> Result<SocketAddr, SettingsError> {
        let ip: IpAddr = self
            .host
            .parse()
            .map_err(|_| SettingsError::BadHost(self.host.clone()))?;
        Ok(SocketAddr::new(ip, self.port))
    }

    /// Whether the bind address is reachable from outside this machine.
    ///
    /// Used to decide whether a missing API key is worth refusing to start over.
    pub fn is_public(&self) -> bool {
        match self.host.parse::<IpAddr>() {
            // An unspecified address binds every interface, which is public even
            // though it is not itself an address anyone can connect to.
            Ok(IpAddr::V4(ip)) => !ip.is_loopback(),
            Ok(IpAddr::V6(ip)) => !ip.is_loopback() && !ip.is_unspecified(),
            // Unparseable hosts are refused by `bind_addr`, but treating them as
            // public here means the safer answer wins if that ever changes.
            Err(_) => true,
        }
    }

    /// The loopback address, for a startup message.
    pub fn local_addr(&self) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), self.port)
    }
}

/// The JSON shape of a settings file.
///
/// Every field optional, so a file may set exactly one thing.
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
struct FileSettings {
    host: Option<String>,
    port: Option<u16>,
    api_key: Option<String>,
    headroom: Option<f64>,
    workers: Option<usize>,
}

impl FileSettings {
    fn apply(self, settings: &mut Settings) {
        if let Some(host) = self.host {
            settings.host = host;
        }
        if let Some(port) = self.port {
            settings.port = port;
        }
        if let Some(key) = self.api_key {
            settings.api_key = Some(key);
        }
        if let Some(headroom) = self.headroom {
            if (0.0..=1.0).contains(&headroom) {
                settings.headroom = headroom;
            }
        }
        if let Some(workers) = self.workers {
            if workers > 0 {
                settings.workers = workers;
            }
        }
    }
}

/// A value from the environment, treating blank as absent.
fn non_empty(env: &dyn Fn(&str) -> Option<String>, key: &str) -> Option<String> {
    env(key).filter(|v| !v.trim().is_empty())
}

/// Why settings could not be resolved.
#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    /// The settings file could not be read.
    #[error("cannot read settings file {path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying reason.
        #[source]
        source: std::io::Error,
    },

    /// The settings file was not valid JSON, or had the wrong shape.
    #[error("cannot parse settings file {path}: {reason}")]
    Parse {
        /// The file.
        path: PathBuf,
        /// What was wrong.
        reason: String,
    },

    /// The host was not an IP address.
    #[error("'{0}' is not an IP address; use 127.0.0.1 or ::1")]
    BadHost(String),
}

/// Maximum number of models to keep in memory at once, unless overridden.
pub const DEFAULT_MODEL_CAP: usize = assembly_core::resident::DEFAULT_MODEL_CAP;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn defaults_are_a_loopback_port_eighty_thousand() {
        let s = Settings::default();
        assert_eq!(s.host, "127.0.0.1");
        assert_eq!(s.port, DEFAULT_PORT);
        assert!(s.api_key.is_none(), "no key means no authentication");
        assert!((s.headroom - 0.8).abs() < f64::EPSILON);
        assert_eq!(s.bind_addr().unwrap().port(), DEFAULT_PORT);
    }

    #[test]
    fn loopback_is_not_public() {
        let mut s = Settings::default();
        assert!(!s.is_public());
        s.host = "0.0.0.0".into();
        assert!(s.is_public(), "every interface is not just this machine");
        s.host = "::1".into();
        assert!(!s.is_public());
    }

    #[test]
    fn a_missing_settings_file_is_not_an_error() {
        let mut s = Settings::default();
        s.apply_file("/nonexistent/definitely-not-here.json")
            .unwrap();
        assert_eq!(s, Settings::default(), "defaults survive");
    }

    #[test]
    fn a_settings_file_overrides_defaults_but_leaves_absent_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        std::fs::write(&path, r#"{"port": 9999, "headroom": 0.5}"#).unwrap();

        let mut s = Settings::default();
        s.apply_file(&path).unwrap();
        assert_eq!(s.port, 9999);
        assert!((s.headroom - 0.5).abs() < f64::EPSILON);
        assert_eq!(s.host, DEFAULT_HOST, "an absent key is left alone");
    }

    #[test]
    fn a_broken_settings_file_is_reported_not_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        std::fs::write(&path, "{not json").unwrap();

        let mut s = Settings::default();
        let err = s.apply_file(&path).unwrap_err();
        assert!(matches!(err, SettingsError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn the_environment_overrides_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.json");
        std::fs::write(&path, r#"{"port": 1111}"#).unwrap();

        let mut s = Settings::default();
        s.apply_file(&path).unwrap();
        s.apply_env(&env_from(&[("ASSEMBLY_PORT", "2222")]));
        assert_eq!(s.port, 2222, "env wins over the file");
    }

    #[test]
    fn a_blank_environment_value_is_no_opinion() {
        let mut s = Settings::default();
        s.apply_env(&env_from(&[
            ("ASSEMBLY_HOST", "   "),
            ("ASSEMBLY_API_KEY", ""),
        ]));
        assert_eq!(s.host, DEFAULT_HOST);
        assert!(s.api_key.is_none());
    }

    #[test]
    fn nonsense_environment_values_are_ignored_rather_than_fatal() {
        let mut s = Settings::default();
        s.apply_env(&env_from(&[
            ("ASSEMBLY_PORT", "eight thousand"),
            ("ASSEMBLY_HEADROOM", "half"),
            ("ASSEMBLY_WORKERS", "0"),
        ]));
        assert_eq!(s.port, DEFAULT_PORT);
        assert!((s.headroom - 0.8).abs() < f64::EPSILON);
        assert_eq!(s.workers, DEFAULT_WORKERS);
    }

    #[test]
    fn a_host_that_is_not_an_ip_is_refused() {
        let s = Settings {
            host: "example.com".into(),
            ..Settings::default()
        };
        assert!(matches!(s.bind_addr(), Err(SettingsError::BadHost(_))));
    }
}
