"""Eval runner — decision-illustrator test cases, not pass/fail benchmarks."""

import json
import logging
from pathlib import Path

import pandas as pd
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
EVAL_DIR = Path("evals")


def main():
    # Load pipeline
    catalog = pd.read_csv(DATA_DIR / "catalog.csv")
    catalog = catalog[catalog["active"] == "Y"].reset_index(drop=True)
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
    pipeline = MatchPipeline(catalog, embeddings, bm25_index, profiles, history)

    # Load test cases
    test_cases = json.loads((EVAL_DIR / "test_cases.json").read_text())

    results = []
    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"[{tc['id']}] {tc['query']}")
        print(f"  Customer: {tc['customer_id'] or 'None'}")
        print(f"  Expected: {tc['expected_behavior']}")
        print(f"  Decision: {tc['decision_illustrated']}")

        result = pipeline.match(tc["query"], tc.get("customer_id"))

        print(f"  history_path={result.used_history_path}, profile_applied={result.profile_applied}")
        for i, m in enumerate(result.matches):
            extra = ""
            if m.affinity_note:
                extra += f" | {m.affinity_note}"
            if m.conflict_flag:
                extra += f" | CONFLICT"
            print(f"  #{i+1} {m.confidence:.2f} | {m.catalog_id} | {m.catalog_description}{extra}")
            print(f"       {m.reasoning}")

        results.append(
            {
                "test_case": tc,
                "result": result.model_dump(),
            }
        )

    # Save full results
    report_path = EVAL_DIR / "latest_report.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n\nFull results saved to {report_path}")


if __name__ == "__main__":
    main()
