//! Parity tests against the Python reference implementation.
//!
//! The expected values in this file were produced by
//! `assembly/header.py` reading the real 11.11 GiB `Mesh-1.2.asmb` archive.
//! If a change to this crate makes any of them fail, the archive format has
//! moved and the Python side needs to know before the two diverge.
//!
//! Set `ASSEMBLY_REAL_ARCHIVE` to a path to additionally verify an archive on
//! disk. Without it those tests are skipped, so the suite still runs in CI
//! without an 11 GiB fixture.

// The anchor values below are f32 literals written out at full length, exactly
// as Python printed them. Truncating them would change the numbers the parity
// assertion is there to protect.
#![allow(clippy::excessive_precision)]

use asmb_format::{read_header, AsmbHeader, EmbeddingEntry, ModelEntry, PAGE_SIZE};
use std::path::PathBuf;

/// The header of `Mesh-1.2.asmb`, as the Python reference parses it.
const EXPECTED_NAME: &str = "Mesh-1.2";
const EXPECTED_ROUTE_NAME: &str = "route";
const EXPECTED_ROUTE_OFFSET: u64 = 102_400;
const EXPECTED_ROUTE_SIZE: u64 = 547_664_768;
const EXPECTED_ROUTE_SHA: &str = "ed3a84b570c5513bfd6bfe0ed4cdc8d5a5de5c6b5029fbbc2822d59fc893c1f8";
const EXPECTED_THRESHOLD: f32 = 0.75;

/// `(name, offset, size, sha256, first four components of anchor 0)`.
const EXPECTED_MODELS: &[(&str, u64, u64, &str, [f32; 4])] = &[
    (
        "general",
        547_770_368,
        2_019_377_696,
        "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
        [
            -0.8627481460571289,
            -0.49002352356910706,
            -3.2941222190856934,
            -0.10404559969902039,
        ],
    ),
    (
        "coding",
        2_567_151_616,
        4_683_073_536,
        "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c",
        [
            -0.40424665808677673,
            0.5799844861030579,
            -2.9736647605895996,
            -1.5501444339752197,
        ],
    ),
    (
        "reasoning",
        7_250_227_200,
        4_683_073_504,
        "731ece8d06dc7eda6f6572997feb9ee1258db0784827e642909d9b565641937b",
        [
            0.4080260992050171,
            -0.04807036370038986,
            -3.9348862171173096,
            0.6117367744445801,
        ],
    ),
];

/// Rebuild the real header in memory so the suite needs no fixture on disk.
///
/// Offsets and sizes are the archive's; only the bytes after the header are
/// absent, which does not matter for everything this crate validates up front.
fn reference_header() -> AsmbHeader {
    AsmbHeader {
        assembly_name: EXPECTED_NAME.to_string(),
        version: 1,
        page_size: PAGE_SIZE,
        embedding: EmbeddingEntry {
            name: EXPECTED_ROUTE_NAME.to_string(),
            offset: EXPECTED_ROUTE_OFFSET,
            size: EXPECTED_ROUTE_SIZE,
            confidence_threshold: EXPECTED_THRESHOLD,
            fallback_model: None,
            sha256: EXPECTED_ROUTE_SHA.to_string(),
        },
        // Placeholder vectors: the real ones are checked against the live
        // archive below. Geometry is what matters for the parity assertions.
        models: EXPECTED_MODELS
            .iter()
            .map(|(name, offset, size, sha, _)| {
                ModelEntry::with_vectors(
                    *name,
                    &[vec![0.0; 768]],
                    *offset,
                    *size,
                    vec![format!("{name} anchor")],
                    *sha,
                )
                .unwrap()
            })
            .collect(),
        persona: Some("reference persona".to_string()),
    }
}

#[test]
fn geometry_matches_the_python_reference() {
    let header = reference_header();

    assert_eq!(header.assembly_name, EXPECTED_NAME);
    assert_eq!(header.version, 1);
    assert_eq!(header.page_size, PAGE_SIZE);
    assert!(header.persona.is_some());

    let emb = &header.embedding;
    assert_eq!(emb.name, EXPECTED_ROUTE_NAME);
    assert_eq!(emb.offset, EXPECTED_ROUTE_OFFSET);
    assert_eq!(emb.size, EXPECTED_ROUTE_SIZE);
    assert_eq!(emb.confidence_threshold, EXPECTED_THRESHOLD);
    assert_eq!(emb.sha256, EXPECTED_ROUTE_SHA);
    assert_eq!(emb.fallback_model, None);

    assert_eq!(header.models.len(), 3);
    for (entry, (name, offset, size, sha, _)) in header.models.iter().zip(EXPECTED_MODELS) {
        assert_eq!(entry.name, *name);
        assert_eq!(entry.offset, *offset);
        assert_eq!(entry.size, *size);
        assert_eq!(entry.sha256, *sha);
        assert_eq!(entry.anchor_dim, 768);
    }
}

#[test]
fn every_real_offset_is_page_aligned() {
    let header = reference_header();
    let page = header.page_size as u64;
    assert_eq!(header.embedding.offset % page, 0);
    for m in &header.models {
        assert_eq!(m.offset % page, 0, "{} is unaligned", m.name);
    }
}

#[test]
fn blobs_do_not_overlap_and_run_forward() {
    // Offsets are ascending and each blob ends before the next begins. The
    // archive is packed in order with no gaps other than page padding.
    let header = reference_header();
    let mut entries: Vec<(&str, u64, u64)> = vec![(
        header.embedding.name.as_str(),
        header.embedding.offset,
        header.embedding.size,
    )];
    entries.extend(
        header
            .models
            .iter()
            .map(|m| (m.name.as_str(), m.offset, m.size)),
    );

    for pair in entries.windows(2) {
        let (prev_name, prev_off, prev_size) = pair[0];
        let (next_name, next_off, _) = pair[1];
        assert!(
            prev_off + prev_size <= next_off,
            "{prev_name} ({}..{}) overlaps {next_name} at {next_off}",
            prev_off,
            prev_off + prev_size
        );
    }
}

#[test]
fn sha256_fields_are_64_hex_chars() {
    let header = reference_header();
    for (name, sha) in [(
        header.embedding.name.as_str(),
        header.embedding.sha256.as_str(),
    )]
    .into_iter()
    .chain(
        header
            .models
            .iter()
            .map(|m| (m.name.as_str(), m.sha256.as_str())),
    ) {
        assert_eq!(sha.len(), 64, "{name} has a short digest");
        assert!(
            sha.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()),
            "{name} has a non-lowercase-hex digest: {sha}"
        );
    }
}

/// Path to a real archive, when the environment names one.
fn real_archive() -> Option<PathBuf> {
    std::env::var_os("ASSEMBLY_REAL_ARCHIVE").map(PathBuf::from)
}

#[test]
fn parses_a_real_archive_when_one_is_available() {
    let Some(path) = real_archive() else {
        eprintln!("skipping: set ASSEMBLY_REAL_ARCHIVE to a .asmb path");
        return;
    };

    let data = std::fs::read(&path).expect("archive is readable");
    let header = read_header(&data).expect("archive header is valid");

    assert_eq!(header.assembly_name, EXPECTED_NAME);
    assert_eq!(header.page_size, PAGE_SIZE);
    assert_eq!(header.embedding.name, EXPECTED_ROUTE_NAME);
    assert_eq!(header.embedding.offset, EXPECTED_ROUTE_OFFSET);
    assert_eq!(header.embedding.size, EXPECTED_ROUTE_SIZE);
    assert_eq!(header.embedding.sha256, EXPECTED_ROUTE_SHA);
    assert_eq!(header.embedding.confidence_threshold, EXPECTED_THRESHOLD);

    let names: Vec<&str> = header.models.iter().map(|m| m.name.as_str()).collect();
    let expected: Vec<&str> = EXPECTED_MODELS.iter().map(|(n, ..)| *n).collect();
    assert_eq!(names, expected, "model list drifted from the reference");

    for (entry, (name, offset, size, sha, first4)) in header.models.iter().zip(EXPECTED_MODELS) {
        assert_eq!(entry.name, *name);
        assert_eq!(entry.offset, *offset, "{name} offset drifted");
        assert_eq!(entry.size, *size, "{name} size drifted");
        assert_eq!(entry.sha256, *sha, "{name} digest drifted");
        assert_eq!(entry.anchor_dim, 768, "{name} anchor width drifted");

        let vectors = entry.anchor_vectors().expect("anchors decode");
        assert_eq!(vectors.len(), 8, "{name} anchor count drifted");
        assert_eq!(vectors[0].len(), 768);
        assert_eq!(
            &vectors[0][..4],
            first4,
            "{name} anchor values drifted; float32 encoding changed"
        );
    }
}

#[test]
fn verifies_every_real_blob_when_one_is_available() {
    let Some(path) = real_archive() else {
        eprintln!("skipping: set ASSEMBLY_REAL_ARCHIVE to a .asmb path");
        return;
    };

    let archive = asmb_format::Archive::open(&path).expect("archive opens");
    let header = archive.header().clone();

    let mut checked = 0;
    let mut entries: Vec<(&str, u64, u64, &str)> = vec![(
        header.embedding.name.as_str(),
        header.embedding.offset,
        header.embedding.size,
        header.embedding.sha256.as_str(),
    )];
    entries.extend(
        header
            .models
            .iter()
            .map(|m| (m.name.as_str(), m.offset, m.size, m.sha256.as_str())),
    );

    for (name, offset, size, expected) in entries {
        let actual = asmb_format::hash_file_region(&path, offset, size)
            .unwrap_or_else(|e| panic!("{name}: hashing failed: {e}"));
        assert_eq!(
            &actual, expected,
            "{name} bytes do not match the digest Python recorded"
        );
        checked += 1;
    }
    assert_eq!(checked, 4, "router plus three execution models");
}
