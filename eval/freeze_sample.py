"""Freeze a manually annotated 20-paper relevance sample from one saved run."""

import argparse
import json
import sqlite3
from pathlib import Path


ANNOTATIONS = {
    "2610.30280": (True, "Directly studies Muon and Newton-Schulz optimization."),
    "2610.28975": (True, "Analyzes orthogonalization for neural-network optimization."),
    "2610.28176": (False, "Focuses on temporal knowledge-graph reasoning."),
    "2610.30436": (False, "Focuses on secure agentic systems for 6G networks."),
    "2610.27444": (True, "Studies low-rank adaptation composition and efficiency."),
    "2610.31069": (False, "Focuses on federated differential privacy, not the profile's LLM methods."),
    "2610.31656": (False, "Studies meta-reinforcement learning rather than LLM training."),
    "2610.30646": (False, "Focuses on federated domain adaptation."),
    "2610.27674": (True, "Evaluates momentum and Muon-style optimization for language models."),
    "2610.29134": (True, "Studies PEFT optimization and reasoning-task performance."),
    "2610.27747": (True, "Directly analyzes Muon-type spectral optimizers."),
    "2610.29357": (True, "Studies agent memory and behavioral evaluation, a stated project interest."),
    "2610.31633": (True, "Measures memory and inference efficiency for looped language models."),
    "2610.31098": (False, "Focuses on inference KV-cache compression rather than training methods."),
    "2610.29424": (False, "Compresses Skip-gram embeddings, outside the profile's LLM focus."),
    "2610.30344": (True, "Evaluates RS-LoRA and efficient continual pretraining for LLMs."),
    "2610.28500": (False, "Focuses on low-bit decoding systems rather than the profile's methods."),
    "2610.27835": (False, "Focuses on online continual learning, not the profile's target methods."),
    "2610.29396": (False, "Studies neuron alignment for implicit neural representations."),
    "2610.30421": (False, "Focuses on reward-model data weighting rather than the profile's core topics."),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/digest.sqlite3"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("eval/sample.json"))
    args = parser.parse_args()

    with sqlite3.connect(args.db) as connection:
        row = connection.execute(
            "SELECT state_json FROM digest_runs WHERE run_id=?", (args.run_id,)
        ).fetchone()
        if row is None:
            parser.error(f"Unknown run ID: {args.run_id}")
        state = json.loads(row[0])
        candidates = connection.execute(
            "SELECT paper_id, paper_json FROM run_candidates "
            "WHERE run_id=? ORDER BY rank",
            (args.run_id,),
        ).fetchall()

    papers = {paper_id: json.loads(payload) for paper_id, payload in candidates}
    missing = set(ANNOTATIONS) - papers.keys()
    if missing:
        parser.error("Saved run lacks annotated IDs: " + ", ".join(sorted(missing)))

    sample = {
        "schema_version": 1,
        "source_run_id": args.run_id,
        "task": "Binary relevance for one fixed researcher profile and paper abstract.",
        "profile": state["profile"],
        "date_from": state["config"]["date_from"],
        "date_to": state["config"]["date_to"],
        "categories": state["config"]["categories"],
        "annotation": "Single-person manual labels; rationale is the annotator's judgment, not model output.",
        "tasks": [
            {
                **papers[paper_id],
                "task_id": paper_id,
                "relevant": relevant,
                "rationale": rationale,
            }
            for paper_id, (relevant, rationale) in ANNOTATIONS.items()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(sample['tasks'])} frozen tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())