"""Customer affinity reranking — math-based confidence adjustment using profiles."""

import logging
import re
from typing import Optional

from .models import CustomerProfile, RankedMatch

logger = logging.getLogger(__name__)

AFFINITY_WEIGHT = 0.4
SPARSE_HISTORY_THRESHOLD = 8

# --- Spec extraction (regex/keyword, NO LLM) ---

_MATERIAL_PATTERNS = {
    "stainless": "18-8 SS",
    "18-8": "18-8 SS",
    "ss": "18-8 SS",
    "brass": "BRASS",
    "alloy": "ALLOY",
    "steel": "STEEL",
}

_FINISH_PATTERNS = {
    "zinc": "ZINC",
    "hdg": "HDG",
    "galvanized": "HDG",
    "black oxide": "BLACK OXIDE",
    "plain": "PLAIN",
    "yellow zinc": "YELLOW ZINC",
    "mech zinc": "MECH ZINC",
}


def extract_query_specs(query: str) -> dict[str, Optional[str]]:
    """Heuristically detect which features the query already specifies.

    Conservative: returns None when unsure rather than guessing.
    """
    q = query.lower()
    specs: dict[str, Optional[str]] = {
        "material": None,
        "thread_system": None,
        "finish": None,
    }

    # Material detection — short patterns (<=3 chars) use word boundaries
    # to avoid false positives (e.g., "pressure" matching "ss")
    _SHORT_PATTERNS = {"ss", "hdg"}
    for pattern, canonical in _MATERIAL_PATTERNS.items():
        if pattern in _SHORT_PATTERNS:
            if re.search(rf"\b{re.escape(pattern)}\b", q):
                specs["material"] = canonical
                break
        elif pattern in q:
            specs["material"] = canonical
            break

    # Thread system detection
    if re.search(r"\bm\d", q):
        specs["thread_system"] = "metric"
    elif re.search(r"\bmm\b", q):
        specs["thread_system"] = "metric"
    elif re.search(r"\d+/\d+", q):
        specs["thread_system"] = "imperial"
    elif re.search(r"#\d+", q):
        specs["thread_system"] = "imperial"

    # Finish detection — check before material overrides
    for pattern, canonical in _FINISH_PATTERNS.items():
        if pattern in q:
            specs["finish"] = canonical
            # If "zinc" matched as finish and no material was explicitly set,
            # default to steel — unless the query mentions a non-steel material
            if pattern == "zinc" and specs["material"] is None:
                _NON_STEEL = ["brass", "stainless", "ss", "316", "18-8", "alloy", "aluminum", "copper", "bronze"]
                if not any(re.search(rf"\b{re.escape(m)}\b", q) for m in _NON_STEEL):
                    specs["material"] = "STEEL"
            break

    return specs


def extract_specs_from_description(description: str) -> dict[str, Optional[str]]:
    """Extract material/thread/finish from a catalog description."""
    d = description.upper()
    specs: dict[str, Optional[str]] = {
        "material": None,
        "thread_system": None,
        "finish": None,
    }

    # Material
    if "18-8" in d or "18/8" in d:
        specs["material"] = "18-8 SS"
    elif "316 SS" in d or "316SS" in d:
        specs["material"] = "316 SS"
    elif "A2 SS" in d or "A4 SS" in d:
        specs["material"] = "SS"
    elif " SS " in d or d.endswith(" SS"):
        specs["material"] = "SS"
    elif "BRASS" in d:
        specs["material"] = "BRASS"
    elif "ALLOY" in d:
        specs["material"] = "ALLOY"
    elif "STEEL" in d:
        specs["material"] = "STEEL"

    # Thread system
    if re.search(r"\bM\d", d):
        specs["thread_system"] = "metric"
    elif re.search(r"\d+/\d+", d):
        specs["thread_system"] = "imperial"
    elif re.search(r"#\d+", d):
        specs["thread_system"] = "imperial"

    # Finish
    if "BLACK OXIDE" in d or " BO" in d:
        specs["finish"] = "BLACK OXIDE"
    elif "YELLOW" in d and "ZINC" in d:
        specs["finish"] = "YELLOW ZINC"
    elif "MECH" in d and ("ZINC" in d or "ZN" in d or "ZC" in d):
        specs["finish"] = "MECH ZINC"
    elif "HDG" in d or "GALVANIZED" in d:
        specs["finish"] = "HDG"
    elif "PLAIN" in d or "PLN" in d or " PL" in d:
        specs["finish"] = "PLAIN"
    elif "ZINC" in d or "ZN" in d or "ZC" in d:
        specs["finish"] = "ZINC"

    return specs


def _canonicalize_material_finish(specs: dict[str, Optional[str]]) -> Optional[str]:
    """Combine material + finish into a profile distribution key."""
    material = specs.get("material")
    finish = specs.get("finish")
    if material and finish:
        return f"{material} {finish}"
    return material


# Material family groups — members within a family get partial affinity
_MATERIAL_FAMILIES: dict[str, str] = {
    "18-8 SS": "SS",
    "316 SS": "SS",
    "A2 SS": "SS",
    "A4 SS": "SS",
    "SS": "SS",
    "STEEL": "STEEL",
    "BRASS": "BRASS",
    "ALLOY": "ALLOY",
}


def _profile_material_affinity(
    cand_key: str,
    profile_dist: dict[str, float],
) -> float:
    """Look up candidate material in profile distribution.

    First tries exact match. If none, tries family match (e.g., 316 SS
    matches profile entries for 18-8 SS) at 70% strength.
    """
    # Exact match on full key (material + finish)
    if cand_key in profile_dist:
        return profile_dist[cand_key]

    # Extract just the material part (before the finish)
    cand_material = cand_key.split()[0] if " " in cand_key else cand_key
    # Handle multi-word materials like "18-8 SS"
    for mat in _MATERIAL_FAMILIES:
        if cand_key.startswith(mat):
            cand_material = mat
            break

    cand_family = _MATERIAL_FAMILIES.get(cand_material)
    if not cand_family:
        return 0.0

    # Sum profile probability for same family, discount to 70%
    family_total = 0.0
    for prof_key, prob in profile_dist.items():
        prof_material = prof_key.split()[0] if " " in prof_key else prof_key
        for mat in _MATERIAL_FAMILIES:
            if prof_key.startswith(mat):
                prof_material = mat
                break
        if _MATERIAL_FAMILIES.get(prof_material) == cand_family:
            family_total += prob

    return family_total * 0.7


def apply_affinity(
    matches: list[RankedMatch],
    query: str,
    profile: CustomerProfile,
) -> list[RankedMatch]:
    """Apply customer profile as multiplicative boost on confidence.

    Only boosts on features the query did NOT specify.
    Surfaces affinity_note and conflict_flag on each match.
    """
    if profile.order_count < SPARSE_HISTORY_THRESHOLD:
        return matches
    if profile.profile_confidence < 0.5:
        return matches

    query_specs = extract_query_specs(query)

    for match in matches:
        cand_specs = extract_specs_from_description(match.catalog_description)
        boost = 1.0
        boost_reasons = []

        # Material affinity (only if query is silent on material)
        if query_specs.get("material") is None:
            cand_key = _canonicalize_material_finish(cand_specs)
            if cand_key:
                material_pref = _profile_material_affinity(
                    cand_key, profile.material_distribution
                )
                boost *= 1.0 + AFFINITY_WEIGHT * material_pref
                if material_pref > 0.5:
                    preferred = max(
                        profile.material_distribution.items(),
                        key=lambda x: x[1],
                    )[0]
                    boost_reasons.append(
                        f"customer typically orders {preferred} ({material_pref:.0%} family match)"
                    )

        # Thread system affinity (only if query is silent)
        if query_specs.get("thread_system") is None:
            cand_thread = cand_specs.get("thread_system")
            if cand_thread:
                thread_pref = profile.thread_system_distribution.get(cand_thread, 0.0)
                boost *= 1.0 + AFFINITY_WEIGHT * thread_pref
                if thread_pref > 0.6:
                    boost_reasons.append(
                        f"customer prefers {cand_thread} threading ({thread_pref:.0%})"
                    )

        match.confidence = min(match.confidence * boost, 0.95)
        if boost_reasons:
            match.affinity_note = "Boosted: " + "; ".join(boost_reasons)

    # Conflict detection: query explicitly specifies a material the customer rarely uses
    if query_specs.get("material") is not None:
        top = matches[0]
        top_specs = extract_specs_from_description(top.catalog_description)
        top_key = _canonicalize_material_finish(top_specs)
        if top_key and profile.material_distribution.get(top_key, 0.0) < 0.1:
            preferred = max(
                profile.material_distribution.items(), key=lambda x: x[1]
            )
            top.conflict_flag = (
                f"Customer typically orders {preferred[0]} "
                f"({preferred[1]:.0%} of history). "
                f"This query specifies {top_key}. Verify intent."
            )

    return sorted(matches, key=lambda m: -m.confidence)
