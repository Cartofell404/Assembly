"""
Router — embedding-based model selection via Cosine Similarity.

This module is intentionally dependency-light at import time:
``numpy`` is always needed, but ``llama_cpp`` is imported lazily so that
config parsing and packing can work without the heavy C library installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from assembly.header import AsmbHeader


@dataclass(frozen=True, slots=True)
class RouteResult:
    """Outcome of a single routing decision."""

    model_name: str
    model_index: int
    score: float
    is_fallback: bool


class Router:
    """Stateless router — given a prompt embedding, pick the best model.

    Parameters
    ----------
    header:
        Parsed `.asmb` header (carries anchor embeddings + thresholds).
    """

    def __init__(self, header: AsmbHeader) -> None:
        self._header = header
        self._threshold = header.embedding.confidence_threshold
        self._fallback = header.embedding.fallback_model

        # Build a mapping: model_index → (model_name, anchor_matrix)
        self._models: list[tuple[str, NDArray[np.float32]]] = []
        for entry in header.models:
            if entry.anchor_embeddings:
                mat = np.array(entry.anchor_embeddings, dtype=np.float32)
                # Pre-normalise anchors for fast cosine similarity.
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                mat = mat / norms
            else:
                mat = np.empty((0, 0), dtype=np.float32)
            self._models.append((entry.name, mat))

        # Map fallback name → index for O(1) lookup.
        self._fallback_idx: Optional[int] = None
        if self._fallback:
            for idx, (name, _) in enumerate(self._models):
                if name == self._fallback:
                    self._fallback_idx = idx
                    break

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, prompt_embedding: NDArray[np.float32]) -> RouteResult:
        """Pick the best model for a given prompt embedding.

        Parameters
        ----------
        prompt_embedding:
            1-D float32 vector (output of the embedding model).

        Returns
        -------
        RouteResult with the chosen model's name, index, score,
        and whether the fallback was used.
        """
        prompt_vec = np.asarray(prompt_embedding, dtype=np.float32).ravel()

        # Normalise prompt vector.
        pnorm = np.linalg.norm(prompt_vec)
        if pnorm > 0:
            prompt_vec = prompt_vec / pnorm

        best_idx = 0
        best_score = -1.0

        for idx, (name, anchor_mat) in enumerate(self._models):
            if anchor_mat.size == 0:
                continue
            # Cosine similarity with every anchor (anchors are pre-normalised).
            similarities = anchor_mat @ prompt_vec  # shape: (n_anchors,)
            max_sim = float(similarities.max())
            if max_sim > best_score:
                best_score = max_sim
                best_idx = idx

        best_name = self._models[best_idx][0]

        # Decision rule ───────────────────────────────────────────────
        if best_score >= self._threshold:
            return RouteResult(
                model_name=best_name,
                model_index=best_idx,
                score=best_score,
                is_fallback=False,
            )

        if self._fallback is not None and self._fallback_idx is not None:
            return RouteResult(
                model_name=self._fallback,
                model_index=self._fallback_idx,
                score=best_score,
                is_fallback=True,
            )

        # No fallback — use best anyway.
        return RouteResult(
            model_name=best_name,
            model_index=best_idx,
            score=best_score,
            is_fallback=False,
        )
