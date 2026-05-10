"""Top-level match() entry point — orchestrates the full pipeline."""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from .affinity import apply_affinity
from .models import CustomerProfile, MatchResult, RankedMatch
from .rerank import rerank
from .retrieval import build_bm25_index, hybrid_retrieve

logger = logging.getLogger(__name__)

# --- History-reference detection ---

HISTORY_REFERENCE_PATTERNS = [
    r"\bsame as last\b",
    r"\bsame as before\b",
    r"\blast time\b",
    r"\blast order\b",
    r"\bprevious order\b",
    r"\bsame washers as\b",
    r"\bsame.*as.*previous\b",
    r"\breorder\b",
    r"\bsame\b.*\bagain\b",
]

_PRODUCT_TYPE_PATTERNS = {
    "washer": ["washer", "wshr", "flat washer", "lock washer"],
    "nut": ["nut", "hex nut"],
    "bolt": ["bolt", "hex bolt", "cap screw", "tap bolt"],
    "screw": ["screw", "lag screw", "machine screw", "cap screw"],
    "rod": ["rod", "threaded rod"],
}


def is_history_reference_query(query: str) -> bool:
    """Check if query references past orders."""
    return any(re.search(p, query, re.IGNORECASE) for p in HISTORY_REFERENCE_PATTERNS)


def _extract_product_type(query: str) -> Optional[str]:
    """Extract product type keyword from a history-reference query."""
    q = query.lower()
    for ptype, keywords in _PRODUCT_TYPE_PATTERNS.items():
        for kw in keywords:
            if kw in q:
                return ptype
    return None


def retrieve_from_customer_history(
    query: str,
    customer_id: str,
    history: pd.DataFrame,
) -> list[RankedMatch]:
    """For 'same X as last time' queries, retrieve from customer's recent history."""
    customer_orders = history[history["customer_id"] == customer_id].sort_values(
        "order_date", ascending=False
    )

    if customer_orders.empty:
        return []

    product_type = _extract_product_type(query)

    if product_type:
        # Filter to matching product type
        mask = customer_orders["catalog_description"].str.lower().apply(
            lambda d: any(kw in d for kw in _PRODUCT_TYPE_PATTERNS.get(product_type, []))
        )
        filtered = customer_orders[mask]
    else:
        filtered = customer_orders

    if filtered.empty:
        return []

    # Deduplicate by SKU, keep most recent
    deduped = filtered.drop_duplicates(subset="sku", keep="first")

    matches = []
    for _, row in deduped.head(3).iterrows():
        matches.append(
            RankedMatch(
                catalog_id=row.get("catalog_id", row["sku"]),
                sku=row["sku"],
                catalog_description=row["catalog_description"],
                confidence=0.88,
                reasoning=(
                    f"From {customer_id}'s order history "
                    f"({row['order_date']}, qty {row['quantity']})"
                ),
            )
        )

    return matches


# --- Preloaded pipeline state ---


class MatchPipeline:
    """Holds pre-loaded indexes and profiles for query-time matching."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        catalog_embeddings: np.ndarray,
        bm25_index: BM25Okapi,
        profiles: dict[str, CustomerProfile],
        history: pd.DataFrame,
    ):
        self.catalog = catalog
        self.catalog_embeddings = catalog_embeddings
        self.bm25_index = bm25_index
        self.profiles = profiles
        self.history = history

    def match(
        self,
        query: str,
        customer_id: Optional[str] = None,
    ) -> MatchResult:
        """Top-level entry point.

        1. If customer selected AND query references history -> Stage 1.5
        2. Otherwise: hybrid_retrieve -> rerank -> (optional) apply_affinity
        3. Return top 3 as MatchResult
        """
        # Stage 1.5: history reference shortcut
        if customer_id and is_history_reference_query(query):
            history_matches = retrieve_from_customer_history(
                query, customer_id, self.history
            )
            if history_matches:
                return MatchResult(
                    query=query,
                    customer_id=customer_id,
                    matches=history_matches[:3],
                    used_history_path=True,
                    profile_applied=False,
                )
            # Fall through to standard pipeline if no history matches

        # Stage 1: hybrid retrieval
        candidates = hybrid_retrieve(
            query,
            self.catalog,
            self.catalog_embeddings,
            self.bm25_index,
            k=15,
        )

        # Stage 2: LLM rerank
        matches = rerank(query, candidates)

        # Stage 3: customer affinity (if applicable)
        profile_applied = False
        if customer_id and customer_id in self.profiles:
            profile = self.profiles[customer_id]
            matches = apply_affinity(matches, query, profile)
            profile_applied = profile.profile_confidence >= 0.5

        return MatchResult(
            query=query,
            customer_id=customer_id,
            matches=matches[:3],
            used_history_path=False,
            profile_applied=profile_applied,
        )
