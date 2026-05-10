"""Customer profile generation using Claude Sonnet 4.6."""

import json
import logging
from pathlib import Path

import anthropic
import pandas as pd

from .models import CustomerProfile

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None

MODEL = "claude-sonnet-4-20250514"


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


PROFILE_SYSTEM = """<identity>
You are a sales analyst at an industrial fastener distributor. Read this
customer's order history and produce a structured profile that captures
how they buy.
</identity>

<context>
This profile will be used to personalize future product recommendations.
The profile must be:
- Specific (not generic platitudes)
- Probabilistic (preferences are tendencies, not laws)
- Honest about sparsity (low-confidence profiles for low-volume customers)

The downstream system will use the distribution fields to mathematically
boost matching candidates. The fingerprint_string is shown to users.

Materials in this catalog: STEEL, 18-8 SS (stainless), BRASS, ALLOY.
Finishes: ZINC, HDG (hot-dipped galvanized), BLACK OXIDE, PLAIN, YELLOW ZINC, MECH ZINC.
Thread systems: metric (M-prefix sizes) and imperial (fractional or # sizes).
</context>"""


PROFILE_TOOL = {
    "name": "submit_customer_profile",
    "description": "Submit the structured customer profile.",
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
            "customer_name": {"type": "string"},
            "industry_inference": {
                "type": "string",
                "description": "1-line industry phrase, e.g. 'Pharma cleanroom manufacturing'",
            },
            "fingerprint_string": {
                "type": "string",
                "description": "1-2 sentences for UI display summarizing buying patterns",
            },
            "material_distribution": {
                "type": "object",
                "description": "Material+finish combos as probabilities summing to 1.0. Keys like 'STEEL ZINC', '18-8 SS PLAIN', 'BRASS HDG', 'ALLOY BLACK OXIDE'.",
                "additionalProperties": {"type": "number"},
            },
            "thread_system_distribution": {
                "type": "object",
                "description": '{"metric": X, "imperial": Y} where X+Y=1.0',
                "properties": {
                    "metric": {"type": "number"},
                    "imperial": {"type": "number"},
                },
            },
            "finish_distribution": {
                "type": "object",
                "description": "Finish family probabilities summing to 1.0",
                "additionalProperties": {"type": "number"},
            },
            "notable_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "2-4 specific observations about this customer's ordering",
            },
            "profile_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in profile accuracy. ~0.4 for 6 orders, ~0.95 for 18 consistent orders.",
            },
            "order_count": {"type": "integer"},
        },
        "required": [
            "customer_id",
            "customer_name",
            "industry_inference",
            "fingerprint_string",
            "material_distribution",
            "thread_system_distribution",
            "finish_distribution",
            "notable_patterns",
            "profile_confidence",
            "order_count",
        ],
    },
}


def generate_profile(
    customer_id: str,
    order_history: pd.DataFrame,
) -> CustomerProfile:
    """Generate a customer profile from their order history via Sonnet 4.6."""
    client = _get_client()

    customer_orders = order_history[order_history["customer_id"] == customer_id]
    customer_name = customer_orders["customer_name"].iloc[0]

    # Format order history as readable text
    history_lines = []
    for _, row in customer_orders.iterrows():
        history_lines.append(
            f"  {row['order_date']} | {row['sku']} | {row['catalog_description']} | qty {row['quantity']}"
        )

    history_text = "\n".join(history_lines)

    user_message = f"""<customer>
Customer ID: {customer_id}
Customer Name: {customer_name}
Total Orders: {len(customer_orders)}
</customer>

<order_history>
{history_text}
</order_history>

Analyze this customer's ordering patterns and submit a structured profile using the submit_customer_profile tool."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=PROFILE_SYSTEM,
        tools=[PROFILE_TOOL],
        tool_choice={"type": "tool", "name": "submit_customer_profile"},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_customer_profile":
            return CustomerProfile(**block.input)

    raise ValueError(f"Profile generation failed for {customer_id}")


def get_or_generate_profiles(
    history_path: Path,
    cache_dir: Path,
) -> dict[str, CustomerProfile]:
    """Load or generate profiles for all customers in order history."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(history_path)
    customer_ids = history["customer_id"].unique()

    profiles: dict[str, CustomerProfile] = {}
    for cid in customer_ids:
        cache_file = cache_dir / f"{cid}.json"
        if cache_file.exists():
            logger.info("Loading cached profile for %s", cid)
            profiles[cid] = CustomerProfile.model_validate_json(cache_file.read_text())
        else:
            logger.info("Generating profile for %s", cid)
            profile = generate_profile(cid, history)
            cache_file.write_text(profile.model_dump_json(indent=2))
            profiles[cid] = profile

    return profiles
