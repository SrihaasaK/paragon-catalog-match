"""LLM reranker using Claude Sonnet 4.6 with structured output via tool use."""

import json
import logging

import anthropic

from .models import Candidate, RankedMatch

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None

MODEL = "claude-sonnet-4-20250514"


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


RERANKER_SYSTEM = """<identity>
You are a senior parts specialist at an industrial fastener distributor.
Given a customer query and a list of catalog candidates, identify the top 3
most likely matches with calibrated confidence scores.
</identity>

<context>
The customer's query may use industry shorthand (SHCS = socket head cap screw,
BHCS = button head cap screw, HHB = hex head bolt, HHC = hex head cap screw).
Catalog descriptions use specific abbreviations (HX = hex, SCR = screw,
ZC/ZN = zinc, HDG = hot-dipped galvanized, BO = black oxide, SS = stainless
steel, PLN/PL = plain, WSHR = washer, MACH = machine). Same product, many
spellings.

Confidence interpretation:
- 0.90+: query is unambiguous and one candidate matches exactly
- 0.70-0.90: best candidate is highly likely but minor specs missing
- 0.50-0.70: leading candidate but real ambiguity remains
- below 0.50: query is too vague or no candidate is a strong match
</context>

<critical_calibration>
Use the FULL 0.0-1.0 confidence range. Do NOT cluster scores at 0.6 or 0.9.
If you are 73% sure, output 0.73. If you are 81% sure, output 0.81.
This is the single most important property of your output.
</critical_calibration>

<calibration_examples>
Example 1:
  Query: "1/4-20 X 3/4 HEX CAP SCREW STEEL ZINC"
  Candidate matches all specs exactly: same size, thread, head type, material, finish.
  Confidence: 0.95
  Reasoning: Every spec in the query matches the candidate description verbatim.

Example 2:
  Query: "M8 hex nut steel zinc"
  Candidate: "M8-1.25 HEX NUT CLASS 8 STEEL ZINC". Match on size, type, material, finish.
  Pitch (1.25) inferred as standard for M8.
  Confidence: 0.85
  Reasoning: All specified attributes match. M8-1.25 is standard pitch for M8.

Example 3:
  Query: "M8 hex nut" (no material)
  Candidate: "M8-1.25 HEX NUT CLASS 8 STEEL ZINC". Material unspecified — could be
  stainless, brass, or zinc.
  Confidence: 0.72
  Reasoning: Size and type match exactly, but query is silent on material.

Example 4:
  Query: "1/2 inch hex nut" (no thread pitch)
  Candidate: "1/2-13 HEX NUT ISO 7380 BRASS ZINC". Multiple 1/2-inch hex nuts
  exist with different pitches (1/2-13, 1/2-20) and materials.
  Confidence: 0.55
  Reasoning: Size matches but multiple thread pitches and materials exist.

Example 5:
  Query: "brass stuff, call me". No product type, no specs.
  Confidence: 0.30
  Reasoning: Query is too vague to commit to a specific candidate.
</calibration_examples>"""


RERANKER_TOOL = {
    "name": "submit_ranked_matches",
    "description": "Submit the top 3 catalog matches ranked by confidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "catalog_id": {"type": "string"},
                        "sku": {"type": "string"},
                        "catalog_description": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": [
                        "catalog_id",
                        "sku",
                        "catalog_description",
                        "confidence",
                        "reasoning",
                    ],
                },
                "minItems": 5,
                "maxItems": 5,
            },
        },
        "required": ["matches"],
    },
}


def rerank(query: str, candidates: list[Candidate]) -> list[RankedMatch]:
    """Send query + candidates to Sonnet, get back top 5 ranked matches.

    Returns 5 so that the affinity stage has room to reorder before trimming
    to the final 3. Uses Anthropic tool-use for structured output.
    """
    client = _get_client()

    candidate_text = "\n".join(
        f"- {c.catalog_id} | {c.sku} | {c.catalog_description}"
        for c in candidates
    )

    user_message = f"""<query>
{query}
</query>

<candidates>
{candidate_text}
</candidates>

Rank the top 5 most likely catalog matches for this query. Use the submit_ranked_matches tool."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0.0,
        system=RERANKER_SYSTEM,
        tools=[RERANKER_TOOL],
        tool_choice={"type": "tool", "name": "submit_ranked_matches"},
        messages=[{"role": "user", "content": user_message}],
    )

    # Extract tool input from response
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_ranked_matches":
            raw_matches = block.input["matches"]
            return [
                RankedMatch(
                    catalog_id=m["catalog_id"],
                    sku=m["sku"],
                    catalog_description=m["catalog_description"],
                    confidence=m["confidence"],
                    reasoning=m["reasoning"],
                )
                for m in raw_matches
            ]

    logger.error("Reranker did not return tool use; raw response: %s", response.content)
    raise ValueError("Reranker failed to produce structured output")
