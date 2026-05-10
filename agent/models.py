"""Pydantic models for the catalog matching pipeline."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """A catalog item surfaced by hybrid retrieval (Stage 1)."""

    catalog_id: str
    sku: str
    catalog_description: str
    retrieval_score: float
    retrieval_method: Literal["embedding", "bm25", "both"]


class RankedMatch(BaseModel):
    """A single ranked match returned by the LLM reranker (Stage 2)."""

    catalog_id: str
    sku: str
    catalog_description: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    affinity_note: Optional[str] = None
    conflict_flag: Optional[str] = None


class RerankerOutput(BaseModel):
    """Structured output schema for the LLM reranker."""

    matches: list[RankedMatch]


class CustomerProfile(BaseModel):
    """LLM-generated customer purchasing profile (Stage 3 input)."""

    customer_id: str
    customer_name: str
    industry_inference: str
    fingerprint_string: str
    material_distribution: dict[str, float]
    thread_system_distribution: dict[str, float]
    finish_distribution: dict[str, float]
    notable_patterns: list[str]
    profile_confidence: float = Field(ge=0.0, le=1.0)
    order_count: int


class MatchResult(BaseModel):
    """Top-level result returned by the matcher."""

    query: str
    customer_id: Optional[str] = None
    matches: list[RankedMatch]
    used_history_path: bool = False
    profile_applied: bool = False
