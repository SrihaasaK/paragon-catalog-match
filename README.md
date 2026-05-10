# Paragon Catalog-Match

A free-text catalog matching system for industrial fastener distribution. Hybrid retrieval (embeddings + BM25) narrows 1,000 SKUs to 15 candidates; Claude Sonnet reranks to top 3 with calibrated confidence scores. Customer order history personalizes rankings when a customer is selected, with explicit handling for sparse-history customers and history-reference queries ("the same washers as last time").

## Architecture

```
Query (+ optional customer_id)
  |
  v
Stage 1: Hybrid Retrieval
  ├── Embed query (OpenAI text-embedding-3-small)
  ├── Cosine similarity vs 1000 catalog embeddings → top 10
  ├── BM25 keyword match vs catalog → top 10
  └── Reciprocal Rank Fusion (k=60) → top 15 candidates
  |
  v
Stage 2: LLM Rerank
  ├── Send query + 15 candidates to Claude Sonnet (structured output via tool use)
  └── Returns top 5 with calibrated confidence (0.0-1.0) + per-result reasoning
  |
  v
Stage 3: Customer Affinity (optional)
  ├── Skip if no customer selected, or sparse history (< 8 orders, confidence < 0.5)
  ├── Only boost on features the query did NOT specify
  ├── Multiplicative: confidence × (1 + 0.4 × preference), clamped to [0, 1]
  ├── Surface affinity_note explaining the boost
  └── Flag conflicts when query contradicts customer profile
  |
  v
Top 3 matches with confidence, reasoning, and optional affinity/conflict notes
```

**Special path (Stage 1.5):** Queries like "the same washers as last time" are pattern-matched and routed directly to the customer's order history, bypassing retrieval entirely.

## Decisions & Reasoning

### Why hybrid retrieval (embedding + BM25)?

Two failure modes need two tools. "M8 hex nut" requires literal token match on "M8" — embedding similarity alone can confuse M8 with M10 because they're semantically close in the vector space. Meanwhile, "lock washer 5/8" requires fuzzy matching — catalog descriptions might say "LOCK WSHR" or "LOCK WASHER ASTM A307". BM25 catches the first case (exact token overlap matters); embeddings catch the second (semantic similarity handles abbreviation variants). Reciprocal Rank Fusion (k=60, the standard constant) merges both rankings without requiring any tuning.

### Why Claude Sonnet over Opus or GPT-4o?

- **Cost and speed.** Sonnet is roughly 5x cheaper than Opus and significantly faster. The reranking task is well-structured: 15 candidates in, top 5 out with reasoning. This is squarely within Sonnet's capability range.
- **Structured output reliability.** Anthropic's tool-use feature provides deterministic structured output by forcing the model to call a tool whose schema matches the desired output format. This eliminates JSON parsing failures.
- **Reasoning quality matters here.** The per-result reasoning text is shown to users and to Kasyap. Sonnet produces substantive reasoning that explains *why* a candidate matches or doesn't, not just "good match."
- **Considered Opus fallback.** Didn't implement it because Sonnet handled all test queries cleanly — including edge cases like "brass stuff, call me" (appropriately low confidence) and SHCS shorthand resolution. Adding a fallback for its own sake adds complexity without observed benefit.

### Why pre-computed customer profiles instead of on-the-fly LLM analysis?

Customer profiles are generated once at startup by sending each customer's full order history to Claude Sonnet, which returns a structured `CustomerProfile` (material distributions, thread system preferences, industry inference). These are cached to disk as JSON.

- **Cost:** 5 LLM calls total (amortized), vs. N calls per session with on-the-fly analysis.
- **Determinism:** Same customer profile every query. No variance from how the LLM happens to read the history on a given call.
- **Debuggability:** Profiles are visible in `cache/profiles/CUST-XXX.json`. When a ranking looks wrong, you can open the profile and see exactly what preferences drove the boost.
- **Tradeoff acknowledged:** Profiles can go stale as new orders arrive. In production with live order data, you'd refresh on new orders or recompute nightly.

### Why math-based affinity boost instead of a second LLM reranking call?

The profile generation already used an LLM for what LLMs are good at: reading messy text, finding patterns, writing summaries. At query time, the affinity boost is a simple multiplicative adjustment on the reranker's confidence scores — deterministic, sub-millisecond, and explainable in a way that a second LLM call wouldn't be. Each tool does what it's best at: LLMs read history and extract patterns; math applies those patterns to scores.

### Why soft boost rather than hard filter?

Customer history is probabilistic, not law. CUST-002 (CleanRoom Pharma) ordering 18-8 SS for 17 of 17 orders doesn't mean they'll *never* need zinc steel. A soft multiplicative boost preserves the base ranking when profile signal is weak (sparse-history customers, underspecified queries) and amplifies it when both signals align. The boost formula — `confidence × (1 + 0.4 × preference)` — means a customer with 100% preference for a material gets at most a 40% confidence increase for matching candidates, which is enough to reorder similar-confidence results without overwhelming strong reranker signals.

### Why surface conflicts rather than block with clarification questions?

The brief asks for top-3 with confidence, not interactive Q&A. But the underlying instinct — a 30-year parts veteran would flag a deviation before quoting — is preserved as a result-level `conflict_flag`. When CUST-002 (who exclusively orders 18-8 SS) explicitly queries "M8 hex nut steel zinc," the system returns steel zinc as the top result (respecting the explicit query) but flags: *"Customer typically orders 18-8 SS PLAIN (100% of history). This query specifies STEEL ZINC. Verify intent."* In production, this flag could trigger a clarification prompt — but that's a separate product layer.

### Why the history-reference special path?

Query #34 from the example set — "the same washers as last time" — is conceptually different from every other query. It can only be answered if customer context is available, and the answer comes from the order history, not the catalog. Standard retrieval-then-rerank can't handle this because there's nothing to embed or BM25-match against. The system pattern-matches these queries (via regex) and routes to a separate path that filters the customer's order history by product type.

Without a customer selected, the query falls through to standard retrieval with appropriately low confidence (0.30-0.35), because the system honestly can't answer "last time" without knowing who's asking.

### Why material family matching in affinity?

The catalog uses multiple stainless steel designations (18-8 SS, 316 SS, A2 SS) that are different alloys but the same material family. A customer who orders 18-8 SS exclusively should get a boost for 316 SS candidates too — they're both stainless. The affinity system maps materials to families and applies a 70% discount for family matches vs. exact matches. This prevents the system from treating 316 SS as completely unrelated to 18-8 SS.

### Stack rationale

| Component | Choice | Why |
|---|---|---|
| Reranker | Claude Sonnet | Cheap, fast, strong structured output + reasoning text |
| Profile gen | Claude Sonnet | Pattern recognition over order text, one-time cost |
| Embeddings | OpenAI text-embedding-3-small | Near-zero cost at 1,000 SKUs, 1536-dim, well-benchmarked |
| BM25 | rank_bm25 | Local, fast, deterministic — no API dependency |
| UI | Streamlit | Single-page, free deploy, sufficient for demo |
| Validation | Pydantic | Type-safe LLM I/O, structured models throughout |

## Test Cases

These are decision illustrators, not pass/fail benchmarks. Each one demonstrates a specific architectural choice.

| ID | Query | Customer | Expected | Observed | Decision Illustrated |
|---|---|---|---|---|---|
| tc_01 | M8 hex nut | None | Baseline: multiple materials in top 3 | 0.85 / 0.82 / 0.78 — steel zinc, steel mech zinc, 316 SS | No personalization without customer context |
| tc_02 | M8 hex nut | CUST-002 (pharma) | SS variant boosted to #1 | 316 SS → #1 at 0.97, affinity note present | Profile boosts material family when query is silent |
| tc_03 | M8 hex nut | CUST-001 (industrial) | Steel zinc boosted to #1 | Steel zinc → #1 at 1.00, affinity note present | Same query, different customer, different top result |
| tc_04 | M8 hex nut steel zinc | CUST-002 | Steel zinc top; conflict flag | Steel zinc #1, conflict flag fires | Explicit query overrides profile; conflict surfaced |
| tc_05 | M8 hex nut | CUST-005 (sparse) | Same as baseline | 0.85 / 0.83 / 0.76, profile_applied=false | Sparse history → skip personalization |
| tc_06 | 1/2 inch hex nut | None | Confidence 0.50-0.75 | 0.67 / 0.65 / 0.63 | Underspecified queries → lower confidence |
| tc_07 | the same washers as last time | CUST-001 | Stage 1.5: returns order history | 3 washers from history, 0.88 each | History-reference queries route to special path |
| tc_08 | the same washers as last time | None | Fallback, low confidence | 0.35 / 0.32 / 0.30 | History path requires customer; graceful fallback |
| tc_09 | SHCS 7/16 x 2-1/2 | None | Resolves SHCS shorthand | Socket head cap screws found, 0.83 / 0.81 / 0.64 | Industry shorthand handled by LLM reranker |
| tc_10 | brass stuff, call me | None | Very low confidence (<0.40) | 0.28 / 0.27 / 0.26 — brass items, vague reasoning | Vague queries → low confidence, not hallucinated specificity |

## Known Limitations

- **Spec extraction uses regex/keywords.** The `extract_query_specs` function that determines what the query already specifies (material, thread system, finish) is heuristic. A structured-output LLM call would be more accurate but adds latency to every query. For this catalog size and query complexity, regex is sufficient.
- **Profile generation is one-shot.** Profiles don't update when new orders arrive without deleting the cached JSON and regenerating. In production, you'd trigger regeneration on new order events.
- **RRF k=60 is a default.** The Reciprocal Rank Fusion constant is the standard value from the original paper. It's not tuned against this specific catalog because the default works well and tuning without a labeled eval set would be overfitting to anecdotes.
- **Affinity only reorders the reranker's top 5.** If the customer's preferred material variant doesn't appear in the reranker's output (e.g., no alloy black oxide M8 hex nut in top 5), affinity can't promote it. Expanding to top 10 from the reranker would help but adds LLM cost.
- **Confidence clamping at 1.0.** Strong affinity boosts can push confidence to the 1.0 cap, which slightly misrepresents certainty. A calibration pass post-boost would be more principled.

## What I'd Build Next (Given Another Week)

1. **Multi-turn clarification** when `conflict_flag` fires — extend the UI to surface a "Did you mean...?" prompt rather than just flagging.
2. **Catalog ingestion pipeline** for real distributor exports (CSV → normalized → embedded → indexed), handling format variations across distributors.
3. **Production evaluation harness** with held-out queries and per-stage attribution of failures (was it retrieval, reranking, or affinity that went wrong?).
4. **Learned reranker** (fine-tuned cross-encoder) for the top-N stage at >10k SKUs where LLM reranking per query becomes cost-prohibitive.
5. **Customer history sync** via ERP connector (SAP/NetSuite) — keep profiles fresh as orders land, which is what Paragon Surge does in production.

## Run Locally

```bash
git clone https://github.com/destroyer123456-dev/paragon-catalog-match.git
cd paragon-catalog-match
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and OPENAI_API_KEY

# Run the app (first run computes embeddings + profiles, ~30s)
streamlit run app.py

# Run eval suite
python3 -m evals.eval
```
