"""Extract diagnostic evidence from retained corpus runs; no model calls."""

import argparse
import json
from collections import Counter
from pathlib import Path
from research_digest.llm.usage import summarize_usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    sample = json.loads((root / "sample.json").read_text())
    labels = {p["paper_id"]: p["label"] for p in sample["papers"]}
    ranking = json.loads((root / "bm25.json").read_text())
    result = {
        "initial_top30": dict(Counter(str(p["gold"]) for p in ranking[:30])),
        "runs": [],
    }
    for directory in sorted(root.glob("run-*")):
        events = [
            json.loads(line)
            for line in (directory / "trace.jsonl").read_text().splitlines()
        ]
        calls = [
            json.loads(line)
            for line in (directory / "calls.jsonl").read_text().splitlines()
        ]
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        diagnostics = {
            "run": directory.name,
            "events": dict(Counter(e["type"] for e in events)),
            "usage": summarize_usage([e for e in events if e["type"] == "llm_usage"]),
            "actions": dict(
                Counter(
                    e["decision"]["action"] for e in events if e["type"] == "action"
                )
            ),
            "screening_confusion": dict(
                Counter(
                    f"{labels[pid]} -> {assessment['label']}"
                    for pid, assessment in state.get("screened", {}).items()
                )
            ),
            "rejected_quotes": [],
        }
        for index, call in enumerate(calls, 1):
            if "response" not in call:
                continue
            context = json.loads(call["request"]["input"])
            try:
                response = json.loads(call["response"])
            except ValueError:
                continue
            if response.get("action") != "finish":
                continue
            for paper in response["selections"]:
                for evidence in paper["evidence"]:
                    if evidence["source"] != "inspection":
                        continue
                    passages = (
                        context.get("observations", {})
                        .get("inspect:" + paper["paper_id"], {})
                        .get("passages", [])
                    )
                    if not any(evidence["quote"] in p["text"] for p in passages):
                        diagnostics["rejected_quotes"].append(
                            {
                                "call": index,
                                "paper_id": paper["paper_id"],
                                "quote": evidence["quote"],
                            }
                        )
        result["runs"].append(diagnostics)
    (root / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            [
                {k: v for k, v in run.items() if k != "rejected_quotes"}
                for run in result["runs"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
