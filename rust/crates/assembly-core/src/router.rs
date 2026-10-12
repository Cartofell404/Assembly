//! Embedding-based model routing.
//!
//! Given an embedding of the user's prompt, pick the execution model whose
//! anchor vectors sit closest to it by cosine similarity. Each model is
//! scored by its *best* anchor rather than its mean, because a prompt that
//! matches any one thing a model claims to handle should reach that model.
//!
//! # Why pre-normalise
//!
//! Cosine similarity of two vectors is their dot product once both have unit
//! length. Normalising the anchors once at construction turns each routing
//! decision into a plain matrix-vector product, so scoring a prompt is
//! linear in the number of anchors with no per-query square roots.
//!
//! # Dimension mismatch
//!
//! The prompt embedding comes from the archive's own router model, so its
//! width should match the anchors. It sometimes does not: an archive built
//! with one embedder can be loaded alongside a prompt embedded by another.
//! Rather than panic or silently score garbage, a mismatched model is skipped
//! and reported through [`RouteResult::skipped`].

use asmb_format::AsmbHeader;

/// What one routing decision produced.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteResult {
    /// Name of the chosen model.
    pub model_name: String,
    /// Index of the chosen model in the header's model list.
    pub model_index: usize,
    /// Cosine similarity of the winning anchor, in `-1.0..=1.0`.
    ///
    /// Meaningful even when the fallback was used, where it records how weak
    /// the best real match was.
    pub score: f32,
    /// Whether the confidence threshold sent this to the fallback.
    pub is_fallback: bool,
    /// Models excluded because their anchors did not match the prompt's
    /// dimension.
    pub skipped: Vec<String>,
}

/// Picks a model for a prompt embedding.
#[derive(Debug, Clone)]
pub struct Router {
    /// `(name, unit-length anchors, original width)`, header order.
    models: Vec<ModelAnchors>,
    threshold: f32,
    fallback: Option<usize>,
}

#[derive(Debug, Clone)]
struct ModelAnchors {
    name: String,
    /// Rows pre-normalised to unit length.
    rows: Vec<Vec<f32>>,
    /// Width the anchors were stored with.
    dim: usize,
}

impl Router {
    /// Build a router from a parsed archive header.
    ///
    /// A model with no anchors keeps its name and index but never scores, so
    /// the returned indices always line up with the header's model list.
    ///
    /// # Errors
    ///
    /// Returns [`asmb_format::Error::Anchors`] when stored anchors do not
    /// match their recorded dimension.
    pub fn new(header: &AsmbHeader) -> asmb_format::Result<Router> {
        let mut models = Vec::with_capacity(header.models.len());

        for entry in &header.models {
            let vectors = entry.anchor_vectors()?;
            let dim = entry.anchor_dim;

            let rows = vectors
                .into_iter()
                .map(|mut row| {
                    let norm = l2(&row);
                    // A zero vector has no direction, so it cannot be compared.
                    // Leaving it at zero makes every similarity with it zero,
                    // which is the honest answer.
                    if norm > 0.0 {
                        for v in &mut row {
                            *v /= norm;
                        }
                    }
                    row
                })
                .collect();

            models.push(ModelAnchors {
                name: entry.name.clone(),
                rows,
                dim,
            });
        }

        let fallback = header
            .embedding
            .fallback_model
            .as_ref()
            .and_then(|name| models.iter().position(|m| &m.name == name));

        Ok(Router {
            models,
            threshold: header.embedding.confidence_threshold,
            fallback,
        })
    }

    /// The confidence threshold below which the fallback is used.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// The model names in header order.
    pub fn model_names(&self) -> Vec<&str> {
        self.models.iter().map(|m| m.name.as_str()).collect()
    }

    /// Models that have at least one anchor and can therefore be chosen.
    pub fn routable_models(&self) -> Vec<&str> {
        self.models
            .iter()
            .filter(|m| !m.rows.is_empty())
            .map(|m| m.name.as_str())
            .collect()
    }

    /// Pick a model for `prompt`.
    ///
    /// `prompt` is normalised here, so callers pass the raw embedding.
    ///
    /// The decision rule, in order:
    ///
    /// 1. Highest anchor similarity wins.
    /// 2. If that similarity is at or above the threshold, take it.
    /// 3. Otherwise use the fallback, if the archive names one.
    /// 4. Otherwise take the best match anyway.
    ///
    /// Step 4 matters: refusing to answer because nothing scored well is not
    /// obviously better than answering with the closest model, and an archive
    /// with no threshold configured should never be unanswerable.
    ///
    /// # Errors
    ///
    /// Never fails. The archive is checked at construction, and an empty
    /// prompt falls back to the first model so there is always an answer.
    pub fn route(&self, prompt: &[f32]) -> RouteResult {
        let mut skipped = Vec::new();

        let norm = l2(prompt);
        let unit: Vec<f32> = if norm > 0.0 {
            prompt.iter().map(|v| v / norm).collect()
        } else {
            vec![0.0; prompt.len()]
        };

        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        let mut any_scored = false;

        for (idx, model) in self.models.iter().enumerate() {
            if model.rows.is_empty() {
                continue;
            }
            if model.dim != unit.len() {
                // A different embedder produced this prompt than built these
                // anchors. Scoring it would compare unrelated columns.
                skipped.push(model.name.clone());
                continue;
            }

            let max_sim = model
                .rows
                .iter()
                .map(|row| dot(row, &unit))
                .fold(f32::NEG_INFINITY, f32::max);

            any_scored = true;
            if max_sim > best_score {
                best_score = max_sim;
                best_idx = idx;
            }
        }

        if !any_scored {
            // Nothing could be scored: no anchors at all, or every one had the
            // wrong width. The first model is the only sensible answer.
            return RouteResult {
                model_name: self
                    .models
                    .first()
                    .map(|m| m.name.clone())
                    .unwrap_or_default(),
                model_index: 0,
                score: 0.0,
                is_fallback: false,
                skipped,
            };
        }

        if best_score >= self.threshold {
            return RouteResult {
                model_name: self.models[best_idx].name.clone(),
                model_index: best_idx,
                score: best_score,
                is_fallback: false,
                skipped,
            };
        }

        if let Some(fallback_idx) = self.fallback {
            if fallback_idx != best_idx {
                return RouteResult {
                    model_name: self.models[fallback_idx].name.clone(),
                    model_index: fallback_idx,
                    // The score still describes the best real match, so a
                    // caller can log how far below the line it fell.
                    score: best_score,
                    is_fallback: true,
                    skipped,
                };
            }
        }

        RouteResult {
            model_name: self.models[best_idx].name.clone(),
            model_index: best_idx,
            score: best_score,
            is_fallback: false,
            skipped,
        }
    }
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use asmb_format::{EmbeddingEntry, ModelEntry};

    /// Build a header where each model has the given anchor vectors.
    fn header(
        models: &[(&str, &[Vec<f32>])],
        threshold: f32,
        fallback: Option<&str>,
    ) -> AsmbHeader {
        let entries: Vec<ModelEntry> = models
            .iter()
            .map(|(name, vectors)| {
                ModelEntry::with_vectors(*name, vectors, 0, 100, vec![], "").unwrap()
            })
            .collect();

        AsmbHeader {
            assembly_name: "T".into(),
            version: 1,
            page_size: 4096,
            embedding: EmbeddingEntry {
                name: "route".into(),
                offset: 0,
                size: 10,
                confidence_threshold: threshold,
                fallback_model: fallback.map(str::to_string),
                sha256: String::new(),
            },
            models: entries,
            persona: None,
        }
    }

    #[test]
    fn picks_the_model_whose_anchor_matches() {
        // code anchors point along +x, prose along +y.
        let h = header(
            &[("code", &[vec![1.0, 0.0]]), ("prose", &[vec![0.0, 1.0]])],
            0.5,
            None,
        );
        let router = Router::new(&h).unwrap();

        let got = router.route(&[5.0, 0.0]);
        assert_eq!(got.model_name, "code");
        assert_eq!(got.model_index, 0);
        assert!(!got.is_fallback);
        assert!((got.score - 1.0).abs() < 1e-6, "got {}", got.score);

        let got = router.route(&[0.0, 5.0]);
        assert_eq!(got.model_name, "prose");
        assert_eq!(got.model_index, 1);
    }

    #[test]
    fn takes_the_best_anchor_not_the_average() {
        // A model with one anchor along +x and one along +y should win a
        // prompt that is purely +y, even though its mean direction is diagonal.
        let h = header(&[("mixed", &[vec![1.0, 0.0], vec![0.0, 1.0]])], 0.5, None);
        let router = Router::new(&h).unwrap();
        let got = router.route(&[0.0, 1.0]);
        assert_eq!(got.model_name, "mixed");
        assert!((got.score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn falls_back_below_the_threshold() {
        let h = header(
            &[("code", &[vec![1.0, 0.0]]), ("prose", &[vec![0.0, 1.0]])],
            0.9,
            Some("prose"),
        );
        let router = Router::new(&h).unwrap();

        // Orthogonal to both, so similarity is 0 and nothing clears 0.9.
        let got = router.route(&[1.0, 1.0]);
        assert_eq!(got.model_name, "prose");
        assert!(got.is_fallback);
        assert!(
            got.score < 0.9,
            "score records the weak match, got {}",
            got.score
        );
    }

    #[test]
    fn no_fallback_means_best_match_anyway() {
        let h = header(
            &[("code", &[vec![1.0, 0.0]]), ("prose", &[vec![0.0, 1.0]])],
            0.99,
            None,
        );
        let router = Router::new(&h).unwrap();
        let got = router.route(&[1.0, 1.0]);
        assert!(!got.is_fallback, "the closest model is still the answer");
        assert_eq!(got.model_name, "code", "first wins a tie at 0.0");
    }

    #[test]
    fn a_model_with_no_anchors_is_skipped() {
        let h = header(&[("empty", &[]), ("real", &[vec![1.0, 0.0]])], 0.5, None);
        let router = Router::new(&h).unwrap();
        assert_eq!(router.routable_models(), vec!["real"]);
        assert_eq!(router.model_names().len(), 2, "indices still line up");

        let got = router.route(&[1.0, 0.0]);
        assert_eq!(got.model_name, "real");
        assert_eq!(got.model_index, 1);
    }

    #[test]
    fn a_dimension_mismatch_is_skipped_not_scored() {
        // Two models, one with 2-wide anchors and one with 3-wide.
        let mut h = header(&[("narrow", &[vec![1.0, 0.0]])], 0.0, None);
        h.models.push(
            ModelEntry::with_vectors("wide", &[vec![1.0, 0.0, 0.0]], 0, 100, vec![], "").unwrap(),
        );
        let router = Router::new(&h).unwrap();

        let got = router.route(&[1.0, 0.0]);
        assert_eq!(got.model_name, "narrow");
        assert_eq!(
            got.skipped,
            vec!["wide"],
            "wide cannot score a 2-wide prompt"
        );
    }

    #[test]
    fn every_model_skipped_falls_back_to_the_first() {
        let mut h = header(
            &[
                ("narrow", &[vec![1.0, 0.0]]),
                ("also_narrow", &[vec![0.0, 1.0]]),
            ],
            0.5,
            None,
        );
        h.embedding.fallback_model = None;
        let router = Router::new(&h).unwrap();

        // A 4-wide prompt matches neither model.
        let got = router.route(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(got.model_name, "narrow", "never unanswerable");
        assert_eq!(got.model_index, 0);
        assert_eq!(got.score, 0.0);
        assert_eq!(got.skipped.len(), 2);
    }

    #[test]
    fn a_zero_prompt_routes_without_dividing_by_zero() {
        let h = header(
            &[("code", &[vec![1.0, 0.0]]), ("prose", &[vec![0.0, 1.0]])],
            0.5,
            None,
        );
        let router = Router::new(&h).unwrap();
        let got = router.route(&[0.0, 0.0]);
        assert!(got.score.is_finite(), "got {}", got.score);
        assert_eq!(got.model_index, 0);
    }

    #[test]
    fn a_zero_anchor_cannot_win() {
        let h = header(
            &[("dead", &[vec![0.0, 0.0]]), ("live", &[vec![1.0, 0.0]])],
            0.0,
            None,
        );
        let router = Router::new(&h).unwrap();
        let got = router.route(&[1.0, 0.0]);
        assert_eq!(got.model_name, "live", "a zero anchor has no direction");
    }

    #[test]
    fn scale_does_not_change_the_decision() {
        let h = header(
            &[("code", &[vec![3.0, 4.0]]), ("prose", &[vec![-4.0, 3.0]])],
            0.5,
            None,
        );
        let router = Router::new(&h).unwrap();

        let small = router.route(&[3.0, 4.0]);
        let large = router.route(&[300.0, 400.0]);
        assert_eq!(small.model_name, large.model_name);
        assert!((small.score - large.score).abs() < 1e-5);
        assert!((small.score - 1.0).abs() < 1e-5, "got {}", small.score);
    }

    #[test]
    fn the_threshold_is_read_from_the_header() {
        let h = header(&[("a", &[vec![1.0, 0.0]])], 0.42, None);
        assert!((Router::new(&h).unwrap().threshold() - 0.42).abs() < 1e-6);
    }

    #[test]
    fn a_named_fallback_that_is_absent_is_ignored() {
        let h = header(&[("code", &[vec![1.0, 0.0]])], 0.99, Some("ghost"));
        let router = Router::new(&h).unwrap();
        let got = router.route(&[1.0, 0.0]);
        assert_eq!(got.model_name, "code");
        assert!(!got.is_fallback, "no such model to fall back to");
    }
}
