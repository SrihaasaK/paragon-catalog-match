"""Paragon Catalog Match — Streamlit UI."""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.embeddings import get_or_compute_catalog_embeddings
from agent.matcher import MatchPipeline
from agent.profile import get_or_generate_profiles
from agent.retrieval import build_bm25_index

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
CACHE_DIR = Path("cache")


@st.cache_resource
def load_pipeline() -> MatchPipeline:
    """Load all indexes and profiles once, cached across reruns."""
    catalog = pd.read_csv(DATA_DIR / "catalog.csv")
    embeddings = get_or_compute_catalog_embeddings(
        DATA_DIR / "catalog.csv",
        CACHE_DIR / "embeddings.npy",
    )
    bm25_index = build_bm25_index(catalog["catalog_description"].tolist())
    profiles = get_or_generate_profiles(
        DATA_DIR / "order_history.csv",
        CACHE_DIR / "profiles",
    )
    history = pd.read_csv(DATA_DIR / "order_history.csv")
    return MatchPipeline(catalog, embeddings, bm25_index, profiles, history)


def main():
    st.set_page_config(
        page_title="Paragon Catalog Match",
        page_icon=":wrench:",
        layout="wide",
    )

    st.title("Paragon Catalog Match")
    st.caption(
        "Free-text catalog matching with customer-aware ranking "
        "| [GitHub](https://github.com/destroyer123456-dev/paragon-catalog-match)"
    )

    pipeline = load_pipeline()

    # --- Input controls ---
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "What are you looking for?",
            value="M8 hex nut",
            placeholder="e.g., M8 hex nut, SHCS 7/16 x 2-1/2, brass stuff call me",
        )

    customer_options = {
        "None (no personalization)": None,
        "CUST-001 — Midwest Industrial Supply": "CUST-001",
        "CUST-002 — CleanRoom Pharma MFG": "CUST-002",
        "CUST-003 — Marine Electrical Corp": "CUST-003",
        "CUST-004 — Heavy Machinery Solutions": "CUST-004",
        "CUST-005 — Summit General Maintenance": "CUST-005",
    }

    with col2:
        customer_label = st.selectbox(
            "Customer",
            options=list(customer_options.keys()),
            index=2,  # Default to CUST-002
        )
    customer_id = customer_options[customer_label]

    # --- Customer profile display ---
    if customer_id and customer_id in pipeline.profiles:
        profile = pipeline.profiles[customer_id]
        st.info(f"**{profile.customer_name}** — {profile.fingerprint_string}")
        with st.expander("View full profile"):
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.markdown("**Material Preferences**")
                for mat, pct in sorted(
                    profile.material_distribution.items(),
                    key=lambda x: -x[1],
                ):
                    st.markdown(f"- {mat}: {pct:.0%}")
            with pcol2:
                st.markdown("**Thread System**")
                for ts, pct in sorted(
                    profile.thread_system_distribution.items(),
                    key=lambda x: -x[1],
                ):
                    st.markdown(f"- {ts}: {pct:.0%}")
            with pcol3:
                st.markdown("**Finish Preferences**")
                for fin, pct in sorted(
                    profile.finish_distribution.items(),
                    key=lambda x: -x[1],
                ):
                    st.markdown(f"- {fin}: {pct:.0%}")
            st.markdown(f"**Industry:** {profile.industry_inference}")
            st.markdown(f"**Profile confidence:** {profile.profile_confidence:.0%}")
            st.markdown(f"**Order count:** {profile.order_count}")

    # --- Search ---
    search_clicked = st.button("Search", type="primary", use_container_width=True)

    if search_clicked and query.strip():
        with st.spinner("Matching..."):
            result = pipeline.match(query.strip(), customer_id)

        # --- Results ---
        st.subheader("Top 3 Matches")

        for i, match in enumerate(result.matches):
            with st.container(border=True):
                # Header row
                hcol1, hcol2 = st.columns([4, 1])
                with hcol1:
                    st.markdown(
                        f"**#{i + 1}** — `{match.sku}`"
                    )
                with hcol2:
                    confidence_pct = match.confidence * 100
                    if confidence_pct >= 80:
                        color = "green"
                    elif confidence_pct >= 60:
                        color = "orange"
                    else:
                        color = "red"
                    st.markdown(
                        f"**Confidence:** :{color}[{confidence_pct:.0f}%]"
                    )

                st.markdown(f"**{match.catalog_description}**")
                st.markdown(f"*{match.reasoning}*")

                if match.affinity_note:
                    st.success(match.affinity_note)

                if match.conflict_flag:
                    st.warning(match.conflict_flag)

        # --- Pipeline trace ---
        with st.expander("Pipeline trace"):
            if result.used_history_path:
                st.markdown("**Stage 1.5:** History-reference query detected — results from customer order history")
            else:
                st.markdown("**Stage 1:** Hybrid retrieval (embedding + BM25 + RRF) → top 15 candidates")
                st.markdown("**Stage 2:** LLM rerank (Claude Sonnet) → top 5 with confidence")
                if result.profile_applied:
                    st.markdown("**Stage 3:** Customer affinity reranking applied → top 3")
                elif customer_id:
                    st.markdown("**Stage 3:** Customer affinity skipped (sparse history or low confidence)")
                else:
                    st.markdown("**Stage 3:** No customer selected — affinity skipped")

            st.json(result.model_dump(), expanded=False)


if __name__ == "__main__":
    main()
