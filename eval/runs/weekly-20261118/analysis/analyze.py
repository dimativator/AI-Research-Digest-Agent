"""Reproduce paired weekly analysis from a fixed label export. No network calls."""

import csv
import hashlib
import itertools
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NAMES = {
    "bm25": "BM25",
    "e5": "E5",
    "react": "Agent 1",
    "react-explore": "Agent 2",
    "debate": "Agent 3",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [0.025, 0.975]).tolist()


def main() -> None:
    sample_path = ROOT / "blind-sample.json"
    sample = read(sample_path)
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    labels_path = HERE / "labeled-sample.json"
    if not labels_path.exists():
        with sqlite3.connect(f"file:{ROOT}/labels.sqlite3?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT paper_id,label,note,updated_at FROM labels WHERE sample=?",
                (digest,),
            ).fetchall()
        labels = {r[0]: dict(label=r[1], note=r[2], updated_at=r[3]) for r in rows}
        assert all(
            p["paper_id"] in labels and labels[p["paper_id"]]["label"] in (0, 0.5, 1)
            for p in sample["papers"]
        )
        save(
            labels_path,
            {
                **sample,
                "sample_sha256": digest,
                "papers": [{**p, **labels[p["paper_id"]]} for p in sample["papers"]],
            },
        )
    exported = read(labels_path)
    assert exported["sample_sha256"] == digest
    labels = {p["paper_id"]: p["label"] for p in exported["papers"]}
    manifest = read(ROOT / "manifest.json")
    systems = manifest["systems"]
    weeks = [w["task"] for w in manifest["windows"]]
    selections = read(ROOT / "private-selection-map.json")["selections"]
    assert len(selections) == len(weeks) * len(systems)
    lookup = {(s["task"], s["system"]): s for s in selections}
    assert len(lookup) == len(selections)
    assert set(labels) == {p for s in selections for p in s["selected"]}
    scores = np.empty((len(weeks), len(systems)))
    counts = {s: Counter() for s in systems}
    weekly = []
    for i, week in enumerate(weeks):
        corpus = read(ROOT / week / "corpus.json")
        # Membership is also checked against the frozen metadata collection.
        papers = corpus["papers"] if isinstance(corpus, dict) else corpus
        corpus_ids = {p["paper_id"] for p in papers}
        for j, system in enumerate(systems):
            s = lookup[week, system]
            assert (
                s["status"] == "completed"
                and len(s["selected"]) == len(set(s["selected"])) == 7
            )
            assert set(s["selected"]) <= corpus_ids
            ratings = [labels[p] for p in s["selected"]]
            counts[system].update(ratings)
            scores[i, j] = sum(ratings) / 7
            weekly.append(
                {
                    "week": week,
                    "system": system,
                    "mean_relevance": scores[i, j],
                    "interesting": ratings.count(1),
                    "potentially_interesting": ratings.count(0.5),
                    "not_interesting": ratings.count(0),
                }
            )
    # Exact empirical bootstrap distribution: all 7^7 resamples of whole weeks.
    resamples = np.array(list(itertools.product(range(7), repeat=7)), dtype=np.int8)
    boot = scores[resamples].mean(axis=1)
    summaries = {}
    execution = read(ROOT / "execution-summary.json")["runs"]
    for j, system in enumerate(systems):
        summaries[system] = {
            "mean": float(scores[:, j].mean()),
            "std_across_weeks": float(scores[:, j].std(ddof=1)),
            "bootstrap_95_ci": interval(boot[:, j]),
            "label_counts_across_49_slots": dict(counts[system]),
        }
        runs = [r for r in execution if r["variant"] == system]
        if runs:
            summaries[system]["resources_per_digest"] = {
                "model_calls": float(np.mean([r["stats"]["llm_calls"] for r in runs])),
                "tool_calls": float(np.mean([r["stats"]["tool_calls"] for r in runs])),
                "input_tokens": float(
                    np.mean([r["stats"]["usage"]["input_tokens"] for r in runs])
                ),
                "output_tokens": float(
                    np.mean([r["stats"]["usage"]["output_tokens"] for r in runs])
                ),
                "wall_seconds": float(
                    np.mean(
                        [
                            lookup[w, system]["latency_this_invocation_seconds"]
                            for w in weeks
                        ]
                    )
                ),
                "estimated_api_cost_usd": None,
            }
    signs = np.array(list(itertools.product([-1, 1], repeat=7)))
    comparisons = []
    for a, b in itertools.combinations(range(len(systems)), 2):
        differences = scores[:, a] - scores[:, b]
        effect = float(differences.mean())
        p = float(
            np.mean(np.abs((signs * differences).mean(axis=1)) >= abs(effect) - 1e-12)
        )
        comparisons.append(
            {
                "a": systems[a],
                "b": systems[b],
                "mean_difference_a_minus_b": effect,
                "paired_bootstrap_95_ci": interval(boot[:, a] - boot[:, b]),
                "exact_two_sided_p": p,
                "wins_ties_losses_a": [
                    int(sum(differences > 1e-12)),
                    int(sum(abs(differences) <= 1e-12)),
                    int(sum(differences < -1e-12)),
                ],
            }
        )
    adjusted = 0.0
    for rank, item in enumerate(
        sorted(comparisons, key=lambda x: x["exact_two_sided_p"])
    ):
        adjusted = max(
            adjusted, min(1.0, (len(comparisons) - rank) * item["exact_two_sided_p"])
        )
        item["holm_p"] = adjusted
    save(
        HERE / "results.json",
        {
            "weeks": weeks,
            "weekly_scores": scores.tolist(),
            "systems": systems,
            "summaries": summaries,
            "comparisons": comparisons,
            "label_counts": dict(Counter(labels.values())),
        },
    )
    with (HERE / "weekly-scores.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(weekly[0]))
        writer.writeheader()
        writer.writerows(weekly)
    lines = [
        "Weekly digest evaluation",
        "",
        "Primary outcome: mean relevance@7, using ratings 0 / 0.5 / 1.",
        "Seven paired weekly tasks; one run per agent and week. All 35 digests completed.",
        "",
        "| System | Mean | SD across weeks | Approx. 95% CI | Interesting / potential / no (49 slots) |",
        "|---|---:|---:|---|---|",
    ]
    for system in systems:
        s = summaries[system]
        c = counts[system]
        lo, hi = s["bootstrap_95_ci"]
        lines.append(
            f"| {NAMES[system]} | {s['mean']:.3f} | {s['std_across_weeks']:.3f} | [{lo:.3f}, {hi:.3f}] | {c[1]} / {c[.5]} / {c[0]} |"
        )
    lines += ["", "Paired effects (first system minus second):"]
    for c in comparisons:
        lines.append(
            f"{NAMES[c['a']]} - {NAMES[c['b']]}: {c['mean_difference_a_minus_b']:+.3f}; 95% CI {c['paired_bootstrap_95_ci']}; p={c['exact_two_sided_p']:.5f}; Holm p={c['holm_p']:.5f}."
        )
    lines += ["", "Resources per digest (includes screening and reviewer/check calls):"]
    for system in systems:
        if "resources_per_digest" in summaries[system]:
            lines.append(
                f"{NAMES[system]}: {summaries[system]['resources_per_digest']}"
            )
    caveats = (
        "This is exploratory, personal-relevance evaluation for one annotator. The primary metric was in the run manifest; the all-pairs testing family was fixed in this analysis turn, not preregistered before collection. "
        "Weeks are the units, not the 49 slots or 122 distinct papers. Intervals enumerate all 823543 bootstrap resamples of seven whole weeks, percentile 95%; pairwise intervals are pointwise, not simultaneous. "
        "Tests enumerate all 128 paired sign flips, using absolute mean differences; validity requires independent weeks and exchangeability of the signs under the null. Adjacent weeks may be dependent. "
        "No normality assumption or normality pretest is used. Holm correction covers all ten comparisons. With seven weeks the minimum two-sided p is 0.015625, so this ten-comparison Holm procedure cannot reject any pair at 0.05. "
        "Non-significance does not imply equivalence. Small-sample bootstrap intervals may be optimistic. One model run per task cannot estimate model randomness separately. "
        "Dates overlap development exposure and two pooled papers had prior labels, so this is not a pristine holdout. Only selected papers were labeled: full-corpus recall cannot be calculated. "
        "The intervals generalize to similar weeks only under the sampling assumptions, not to other researchers. No verified gpt-6-sol pricing is recorded, so dollar cost remains unknown; token use is measured. "
        "Agent wall times exclude shared corpus collection/ranking and are affected by concurrency. Baseline timing was not recorded, and is not set to zero."
    )
    lines += ["", caveats]
    (HERE / "analysis-report.txt").write_text("\n".join(lines) + "\n")
    (HERE / "stats-appendix.txt").write_text(
        caveats + "\n\nExact numeric results: results.json and weekly-scores.csv.\n"
    )
    # Standalone vector figures without additional dependencies.
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="390" viewBox="0 0 850 390"><rect width="850" height="390" fill="white"/><g font-family="sans-serif" fill="#162735"><text x="25" y="30" font-size="20">Mean relevance@7 and approximate 95% bootstrap intervals</text>'
    ]
    for tick in range(11):
        x = 160 + tick * 60
        svg.append(
            f'<path d="M{x} 55V320" stroke="#e3e6ea"/><text x="{x-8}" y="345">{tick/10:.1f}</text>'
        )
    for j, system in enumerate(systems):
        y = 80 + j * 50
        s = summaries[system]
        lo, hi = s["bootstrap_95_ci"]
        svg.append(
            f'<text x="25" y="{y+5}">{NAMES[system]}</text><path d="M{160+600*lo} {y}H{160+600*hi}" stroke="#126b8c" stroke-width="4"/><circle cx="{160+600*s["mean"]}" cy="{y}" r="6" fill="#126b8c"/>'
        )
    svg.append(
        '<text x="25" y="380" font-size="13">Seven weekly tasks; intervals assume independent, representative weeks.</text></g></svg>'
    )
    (HERE / "comparison.svg").write_text("".join(svg))
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="400"><rect width="850" height="400" fill="white"/><g font-family="sans-serif"><text x="25" y="30" font-size="20">Weekly mean relevance@7</text>'
    ]
    for j, week in enumerate(weeks):
        svg.append(f'<text x="{180+j*90}" y="70">{week}</text>')
    for i, system in enumerate(systems):
        y = 90 + i * 55
        svg.append(f'<text x="25" y="{y+28}">{NAMES[system]}</text>')
        for j in range(7):
            v = scores[j, i]
            shade = int(250 - 120 * v)
            svg.append(
                f'<rect x="{170+j*90}" y="{y}" width="85" height="48" fill="rgb({shade},{min(255,shade+20)},245)"/><text x="{190+j*90}" y="{y+28}">{v:.2f}</text>'
            )
    svg.append("</g></svg>")
    (HERE / "weekly-breakdown.svg").write_text("".join(svg))
    best = max(systems, key=lambda s: summaries[s]["mean"])
    (HERE / "figure-catalog.txt").write_text(
        f"comparison.svg: Compare mean relevance and pointwise approximate bootstrap uncertainty. {NAMES[best]} has the highest observed mean. Overlap alone is not a paired test. This identifies a provisional leader, not a proven winner.\nweekly-breakdown.svg: Display all seven paired tasks; each cell is the exact weekly mean rounded to two decimals. Inspect consistency and reversals rather than relying on an aggregate alone. No error bars because each cell is one run. Data: weekly-scores.csv.\n"
    )
    save(
        HERE / "hashes.json",
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                sample_path,
                labels_path,
                ROOT / "private-selection-map.json",
                HERE / "analyze.py",
                HERE / "results.json",
            ]
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
