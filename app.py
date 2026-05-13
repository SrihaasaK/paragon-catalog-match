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

# --- Custom CSS ---

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Global overrides */
.stApp {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Hide default Streamlit header and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container spacing */
.block-container {
    padding-top: 2rem !important;
    max-width: 960px !important;
}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(3, 105, 161, 0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-size: 1.75rem;
    font-weight: 800;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
    position: relative;
}

.hero-subtitle {
    font-size: 0.9rem;
    color: #94A3B8;
    margin: 0;
    font-weight: 400;
    position: relative;
}

.hero-links {
    margin-top: 0.75rem;
    position: relative;
}

.hero-links a {
    color: #38BDF8;
    text-decoration: none;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 1.25rem;
    transition: color 0.2s;
}

.hero-links a:hover {
    color: #7DD3FC;
}

/* Profile card */
.profile-card {
    background: #F0F9FF;
    border: 1px solid #BAE6FD;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}

.profile-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}

.profile-name {
    font-weight: 700;
    font-size: 0.95rem;
    color: #0F172A;
}

.profile-industry {
    font-size: 0.75rem;
    color: #0369A1;
    background: #E0F2FE;
    padding: 2px 8px;
    border-radius: 100px;
    font-weight: 500;
}

.profile-fingerprint {
    font-size: 0.85rem;
    color: #334155;
    line-height: 1.5;
}

/* Distribution bars */
.dist-section {
    margin-top: 0.75rem;
}

.dist-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}

.dist-bar-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.25rem;
    font-size: 0.8rem;
}

.dist-label {
    width: 120px;
    color: #334155;
    font-weight: 500;
    flex-shrink: 0;
    text-align: right;
}

.dist-bar-bg {
    flex: 1;
    background: #E2E8F0;
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
}

.dist-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.3s ease;
}

.dist-pct {
    width: 36px;
    text-align: right;
    color: #64748B;
    font-size: 0.75rem;
    font-weight: 500;
    flex-shrink: 0;
}

/* Result cards */
.result-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    position: relative;
}

.result-card:hover {
    border-color: #CBD5E1;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
}

.result-card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.5rem;
}

.result-rank {
    font-size: 0.7rem;
    font-weight: 700;
    color: white;
    background: #0F172A;
    width: 24px;
    height: 24px;
    border-radius: 8px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-right: 0.5rem;
    flex-shrink: 0;
}

.result-sku {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.8rem;
    color: #64748B;
    font-weight: 500;
}

.result-description {
    font-size: 1rem;
    font-weight: 600;
    color: #0F172A;
    margin-bottom: 0.5rem;
    line-height: 1.4;
}

.result-reasoning {
    font-size: 0.85rem;
    color: #475569;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}

/* Confidence badge */
.confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 4px 10px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 700;
}

.confidence-high {
    background: #DCFCE7;
    color: #166534;
}

.confidence-mid {
    background: #FEF3C7;
    color: #92400E;
}

.confidence-low {
    background: #FEE2E2;
    color: #991B1B;
}

/* Pipeline trace */
.trace-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
}

.trace-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.trace-step {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    color: #334155;
    margin-bottom: 0.25rem;
}

.trace-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.trace-active {
    background: #0369A1;
}

.trace-inactive {
    background: #CBD5E1;
}

/* Section headers */
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

/* Input styling */
div[data-testid="stTextInput"] input {
    border-radius: 10px !important;
    border: 1.5px solid #E2E8F0 !important;
    padding: 0.6rem 0.9rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #0369A1 !important;
    box-shadow: 0 0 0 3px rgba(3, 105, 161, 0.1) !important;
}

/* Button styling */
.stButton > button[kind="primary"] {
    background: #0369A1 !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding: 0.55rem 1.5rem !important;
    border: none !important;
    transition: background 0.2s !important;
}

.stButton > button[kind="primary"]:hover {
    background: #075985 !important;
}

/* Selectbox styling */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border: 1.5px solid #E2E8F0 !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 1.5rem 0;
}
</style>
"""


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


def render_confidence_badge(confidence: float) -> str:
    """Render a colored confidence badge."""
    pct = confidence * 100
    if pct >= 75:
        cls = "confidence-high"
    elif pct >= 50:
        cls = "confidence-mid"
    else:
        cls = "confidence-low"
    return f'<span class="confidence-badge {cls}">{pct:.0f}%</span>'


def render_distribution_bars(
    dist: dict[str, float], color: str = "#0369A1"
) -> str:
    """Render horizontal distribution bars."""
    rows = []
    for label, value in sorted(dist.items(), key=lambda x: -x[1]):
        width = max(value * 100, 2)
        rows.append(
            f'<div class="dist-bar-row">'
            f'  <span class="dist-label">{label}</span>'
            f'  <div class="dist-bar-bg">'
            f'    <div class="dist-bar-fill" style="width:{width}%;background:{color}"></div>'
            f'  </div>'
            f'  <span class="dist-pct">{value:.0%}</span>'
            f'</div>'
        )
    return "\n".join(rows)


def main():
    st.set_page_config(
        page_title="Paragon Catalog Match",
        page_icon="https://em-content.zobj.net/source/apple/391/wrench_1f527.png",
        layout="centered",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    pipeline = load_pipeline()

    # --- Hero header ---
    st.markdown(
        """
        <div class="hero-header">
            <h1 class="hero-title">Paragon Catalog Match</h1>
            <p class="hero-subtitle">
                Free-text catalog matching with customer-aware ranking for industrial fasteners
            </p>
            <div class="hero-links">
                <a href="https://github.com/SrihaasaK/paragon-catalog-match" target="_blank">GitHub</a>
                <a href="https://github.com/SrihaasaK/paragon-catalog-match#readme" target="_blank">Documentation</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Input controls ---
    customer_options = {
        "No customer (baseline)": None,
        "CUST-001 \u2014 Midwest Industrial Supply": "CUST-001",
        "CUST-002 \u2014 CleanRoom Pharma MFG": "CUST-002",
        "CUST-003 \u2014 Marine Electrical Corp": "CUST-003",
        "CUST-004 \u2014 Heavy Machinery Solutions": "CUST-004",
        "CUST-005 \u2014 Summit General Maintenance": "CUST-005",
    }

    # Customer dropdown outside form for reactive profile updates
    col_q, col_c = st.columns([3, 1])
    with col_c:
        customer_label = st.selectbox(
            "Customer",
            options=list(customer_options.keys()),
            index=2,  # Default to CUST-002
            label_visibility="collapsed",
        )
    customer_id = customer_options[customer_label]

    # Query + search button in form to suppress "Press Enter to apply"
    with col_q:
        with st.form("search_form"):
            query = st.text_input(
                "Search query",
                value="M8 hex nut",
                placeholder="e.g., M8 hex nut, SHCS 7/16 x 2-1/2, the same washers as last time",
                label_visibility="collapsed",
            )
            search_clicked = st.form_submit_button(
                "Search catalog", type="primary", use_container_width=True
            )

    # --- Customer profile display ---
    if customer_id and customer_id in pipeline.profiles:
        profile = pipeline.profiles[customer_id]

        st.markdown(
            f"""
            <div class="profile-card">
                <div class="profile-card-header">
                    <span class="profile-name">{profile.customer_name}</span>
                    <span class="profile-industry">{profile.industry_inference}</span>
                </div>
                <div class="profile-fingerprint">{profile.fingerprint_string}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View full profile details"):
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.markdown(
                    f'<div class="dist-section">'
                    f'<div class="dist-title">Material</div>'
                    f'{render_distribution_bars(profile.material_distribution, "#0369A1")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with pcol2:
                st.markdown(
                    f'<div class="dist-section">'
                    f'<div class="dist-title">Thread System</div>'
                    f'{render_distribution_bars(profile.thread_system_distribution, "#7C3AED")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with pcol3:
                st.markdown(
                    f'<div class="dist-section">'
                    f'<div class="dist-title">Finish</div>'
                    f'{render_distribution_bars(profile.finish_distribution, "#0891B2")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"**Profile confidence:** {profile.profile_confidence:.0%} "
                f"&nbsp;&middot;&nbsp; **Orders:** {profile.order_count}",
            )

    # --- Search results ---
    if search_clicked and query.strip():
        with st.spinner("Matching..."):
            result = pipeline.match(query.strip(), customer_id)

        # --- Results ---
        st.markdown('<div class="section-header">Top 3 Matches</div>', unsafe_allow_html=True)

        for i, match in enumerate(result.matches):
            confidence_html = render_confidence_badge(match.confidence)

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-card-top">
                        <div>
                            <span class="result-rank">{i + 1}</span>
                            <span class="result-sku">{match.sku}</span>
                        </div>
                        {confidence_html}
                    </div>
                    <div class="result-description">{match.catalog_description}</div>
                    <div class="result-reasoning">{match.reasoning}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if match.affinity_note:
                st.success(match.affinity_note)

            if match.conflict_flag:
                st.warning(match.conflict_flag)

        # --- Pipeline trace ---
        if result.used_history_path:
            steps = [
                ("active", "History-reference query detected"),
                ("active", "Results retrieved from customer order history"),
                ("inactive", "Standard retrieval skipped"),
            ]
        else:
            steps = [
                ("active", "Hybrid retrieval (embedding + BM25 + RRF) \u2192 15 candidates"),
                ("active", "LLM rerank (Claude Sonnet) \u2192 5 with confidence"),
            ]
            if result.profile_applied:
                steps.append(("active", "Customer affinity reranking applied"))
            elif customer_id:
                steps.append(("inactive", "Customer affinity skipped (sparse history)"))
            else:
                steps.append(("inactive", "No customer selected"))

        trace_steps_html = "\n".join(
            f'<div class="trace-step">'
            f'  <span class="trace-dot trace-{status}"></span>'
            f'  {label}'
            f'</div>'
            for status, label in steps
        )

        st.markdown(
            f"""
            <div class="trace-card">
                <div class="trace-title">Pipeline Trace</div>
                {trace_steps_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View raw JSON"):
            st.json(result.model_dump(), expanded=False)


if __name__ == "__main__":
    main()
