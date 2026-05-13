"""OpenAI embedding generation and caching."""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

MODEL = "text-embedding-3-small"
BATCH_SIZE = 2048


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _csv_hash(csv_path: Path) -> str:
    """Quick MD5 of the catalog CSV for cache invalidation."""
    return hashlib.md5(csv_path.read_bytes()).hexdigest()


def _embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts via OpenAI API."""
    client = _get_client()
    response = client.embeddings.create(model=MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def get_or_compute_catalog_embeddings(
    catalog_path: Path,
    cache_path: Path,
) -> np.ndarray:
    """Load cached embeddings or compute and cache them.

    Cache is invalidated if the catalog CSV changes (checked via MD5 hash).
    """
    hash_path = cache_path.with_suffix(".hash")

    current_hash = _csv_hash(catalog_path)
    if cache_path.exists() and hash_path.exists():
        cached_hash = hash_path.read_text().strip()
        if cached_hash == current_hash:
            logger.info("Loading cached embeddings from %s", cache_path)
            return np.load(cache_path)

    logger.info("Computing embeddings for %s", catalog_path)
    df = pd.read_csv(catalog_path)
    df = df[df["active"] == "Y"].reset_index(drop=True)
    descriptions = df["catalog_description"].tolist()

    all_embeddings = []
    for i in range(0, len(descriptions), BATCH_SIZE):
        batch = descriptions[i : i + BATCH_SIZE]
        logger.info("Embedding batch %d-%d", i, i + len(batch))
        all_embeddings.append(_embed_batch(batch))

    embeddings = np.concatenate(all_embeddings, axis=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    hash_path.write_text(current_hash)
    logger.info("Cached %d embeddings to %s", len(embeddings), cache_path)

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    return _embed_batch([query])[0]
