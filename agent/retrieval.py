"""Hybrid retrieval: embeddings + BM25 with Reciprocal Rank Fusion."""

import logging

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .embeddings import embed_query
from .models import Candidate

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard RRF constant — not tuned


def build_bm25_index(descriptions: list[str]) -> BM25Okapi:
    """Build a BM25 index over catalog descriptions.

    Tokenization: lowercase + whitespace split. Fastener descriptions
    are mostly single tokens — no need for stemming or subword tricks.
    """
    tokenized = [desc.lower().split() for desc in descriptions]
    return BM25Okapi(tokenized)


def _cosine_top_k(
    query_embedding: np.ndarray,
    catalog_embeddings: np.ndarray,
    k: int,
) -> list[tuple[int, float]]:
    """Return top-k catalog indices by cosine similarity."""
    # Normalize for cosine
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    norms = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
    catalog_norm = catalog_embeddings / norms

    similarities = catalog_norm @ query_norm
    top_indices = np.argsort(similarities)[::-1][:k]
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def _bm25_top_k(
    query: str,
    index: BM25Okapi,
    k: int,
) -> list[tuple[int, float]]:
    """Return top-k catalog indices by BM25 score."""
    tokens = query.lower().split()
    scores = index.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def hybrid_retrieve(
    query: str,
    catalog: pd.DataFrame,
    catalog_embeddings: np.ndarray,
    bm25_index: BM25Okapi,
    k: int = 15,
) -> list[Candidate]:
    """Hybrid retrieval with Reciprocal Rank Fusion.

    1. Embedding similarity -> top 10
    2. BM25 -> top 10
    3. RRF merges both rankings
    4. Return top k candidates
    """
    query_emb = embed_query(query)
    embed_results = _cosine_top_k(query_emb, catalog_embeddings, k=10)
    bm25_results = _bm25_top_k(query, bm25_index, k=10)

    # Build rank maps (1-indexed)
    embed_ranks: dict[int, int] = {}
    for rank, (idx, _score) in enumerate(embed_results, 1):
        embed_ranks[idx] = rank

    bm25_ranks: dict[int, int] = {}
    for rank, (idx, _score) in enumerate(bm25_results, 1):
        bm25_ranks[idx] = rank

    # RRF: score = 1/(rank_embed + k) + 1/(rank_bm25 + k)
    all_indices = set(embed_ranks.keys()) | set(bm25_ranks.keys())
    rrf_scores: dict[int, float] = {}
    for idx in all_indices:
        score = 0.0
        if idx in embed_ranks:
            score += 1.0 / (embed_ranks[idx] + RRF_K)
        if idx in bm25_ranks:
            score += 1.0 / (bm25_ranks[idx] + RRF_K)
        rrf_scores[idx] = score

    # Sort by RRF score, take top k
    sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:k]

    # Determine retrieval method per candidate
    candidates = []
    for idx in sorted_indices:
        in_embed = idx in embed_ranks
        in_bm25 = idx in bm25_ranks
        if in_embed and in_bm25:
            method = "both"
        elif in_embed:
            method = "embedding"
        else:
            method = "bm25"

        row = catalog.iloc[idx]
        candidates.append(
            Candidate(
                catalog_id=row["catalog_id"],
                sku=row["sku"],
                catalog_description=row["catalog_description"],
                retrieval_score=rrf_scores[idx],
                retrieval_method=method,
            )
        )

    return candidates
