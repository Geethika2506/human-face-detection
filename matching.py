"""Match one embedding against a local gallery.

Each enrolled person can have several reference embeddings. The identity score
is the maximum cosine similarity against that person's references (nearest
reference). Cosine similarity is the dot product of two L2-normalized vectors.
The score runs from -1 to 1. It is not a probability.

Decision, using the configured acceptance threshold and margin:

- No enrolled identities: unknown, and no score is reported.
- Best score below the acceptance threshold: unknown.
- Best score passes, and another identity is within the margin: uncertain.
- Best score passes, and either there is no second identity or the gap is at
  least the margin: known.

With a single enrolled person there is no second identity, so only the
acceptance threshold applies. Published model thresholds are starting points
and can need a new calibration when the gallery grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

EMBEDDING_DIM = 128


@dataclass(frozen=True)
class IdentityRecord:
    person_id: str
    display_name: str
    embeddings: np.ndarray


@dataclass(frozen=True)
class MatchDecision:
    status: str
    person_id: Optional[str] = None
    display_name: Optional[str] = None
    match_score: Optional[float] = None
    match_margin: Optional[float] = None
    second_person_id: Optional[str] = None
    second_score: Optional[float] = None
    # Top identity before the threshold and margin rules. Useful for calibration.
    best_person_id: Optional[str] = None
    best_display_name: Optional[str] = None


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length copy with shape (128,)."""
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    if flat.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"Expected an embedding of length {EMBEDDING_DIM}, got shape {np.asarray(vector).shape}"
        )
    norm = float(np.linalg.norm(flat))
    if norm < 1e-8:
        raise ValueError("Embedding norm is zero")
    return flat / np.float32(norm)


def max_cosine(query: np.ndarray, embeddings: np.ndarray) -> float:
    """Highest cosine between a unit query and each unit reference row."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM or matrix.shape[0] < 1:
        raise ValueError(f"Reference embeddings must have shape (N, {EMBEDDING_DIM})")
    return float(np.max(matrix @ query))


def decide_identity(
    query: np.ndarray,
    identities: Sequence[IdentityRecord],
    accept_threshold: float,
    match_margin: float,
) -> MatchDecision:
    """Choose known, unknown, or uncertain for one query embedding."""
    if not identities:
        return MatchDecision(status="unknown")

    query = l2_normalize(query)
    ranked: list[tuple[float, IdentityRecord]] = []
    for identity in identities:
        ranked.append((max_cosine(query, identity.embeddings), identity))
    ranked.sort(key=lambda item: item[0], reverse=True)

    best_score, best = ranked[0]
    second_score = None
    second_id = None
    margin = None
    if len(ranked) > 1:
        second_score = ranked[1][0]
        second_id = ranked[1][1].person_id
        margin = best_score - second_score

    below_threshold = best_score < accept_threshold
    ambiguous = margin is not None and margin < match_margin
    if below_threshold:
        status = "unknown"
        person_id = None
        display_name = None
    elif ambiguous:
        status = "uncertain"
        person_id = None
        display_name = None
    else:
        status = "known"
        person_id = best.person_id
        display_name = best.display_name

    return MatchDecision(
        status=status,
        person_id=person_id,
        display_name=display_name,
        match_score=best_score,
        match_margin=margin,
        second_person_id=second_id,
        second_score=second_score,
        best_person_id=best.person_id,
        best_display_name=best.display_name,
    )
