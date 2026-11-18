"""Apply the user's post-evaluation CORE label revision without changing frozen runs."""

import importlib.util
import json
import statistics
from pathlib import Path

WEEKLY = Path("eval/runs/weekly-20261118")
REPEATED = Path("eval/runs/repeated-1000-20261118")
PAPER = "2611.07859"


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main():
    revised = WEEKLY / "analysis/core-label-revision"
    revised.mkdir(exist_ok=True)
    (revised / "analyze.py").write_text(
        '"""Reproduce from repo root: python -m eval.recalculate_core_label."""\nfrom eval.recalculate_core_label import main\nif __name__ == "__main__": main()\n'
    )
    labels = read(WEEKLY / "analysis/labeled-sample.json")
    paper = next(p for p in labels["papers"] if p["paper_id"] == PAPER)
    assert paper["label"] == 0
    paper["label"] = 1
    paper["label_revision"] = {
        "previous": 0,
        "new": 1,
        "date": "2026-11-18",
        "reason": "User reclassified CORE as interesting after inspecting evaluation results.",
    }
    save(revised / "labeled-sample.json", labels)
    save(
        revised / "revision.json",
        dict(
            paper_id=PAPER,
            old_label=0,
            new_label=1,
            scope="Both evaluations; post-evaluation human revision. Original snapshots and results retained. No new model calls.",
        ),
    )
    spec = importlib.util.spec_from_file_location(
        "weekly_analysis", WEEKLY / "analysis/analyze.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = revised.resolve()
    module.ROOT = WEEKLY.resolve()
    module.main()
    results = read(REPEATED / "analysis.json")
    updated_labels = {p["paper_id"]: p["label"] for p in labels["papers"]}
    for row in results["runs"]:
        if row["status"] == "completed":
            vals = [updated_labels.get(p, 0) for p in row["selected"]]
            row["relevance_at_10"] = sum(vals) / 10
            row["interesting"] = vals.count(1)
            row["potentially_interesting"] = vals.count(0.5)
    for v, summary in results["summary"].items():
        vals = [
            r["relevance_at_10"]
            for r in results["runs"]
            if r["variant"] == v and r["status"] == "completed"
        ]
        summary.update(
            mean=statistics.mean(vals),
            std=statistics.stdev(vals),
            minimum=min(vals),
            maximum=max(vals),
        )
    for coverage in results["coverage"].values():
        vals = [updated_labels.get(p, 0) for p in coverage["top30_ids"]]
        n = vals.count(1)
        nh = vals.count(0.5)
        coverage.update(
            interesting=n,
            potentially_interesting=nh,
            interesting_coverage=n / 32,
            positive_or_partial_coverage=(n + nh) / 77,
        )
    results["label_revision"] = read(revised / "revision.json")
    save(REPEATED / "analysis-core-label-revision.json", results)
    print("REVISED REPEATED", json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
