"""Generate a pyplot figure and pairwise counts from frozen weekly scores."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "weekly-scores.csv").open()))
# Integer half-points give exact comparisons without floating-point ties.
scores = {
    (r["week"], r["system"]): 2 * int(r["interesting"])
    + int(r["potentially_interesting"])
    for r in rows
}
weeks = [f"week-{i}" for i in range(1, 8)]
pairs = [
    ("react-explore", "react", "Agent 2 vs Agent 1"),
    ("debate", "react", "Agent 3 vs Agent 1"),
    ("debate", "react-explore", "Agent 3 vs Agent 2"),
]
results = []
for a, b, label in pairs:
    diff = [scores[w, a] - scores[w, b] for w in weeks]
    results.append(
        dict(
            comparison=label,
            wins=sum(d > 0 for d in diff),
            ties=diff.count(0),
            losses=sum(d < 0 for d in diff),
            weekly_differences=[d / 14 for d in diff],
        )
    )
(HERE / "weekly-comparisons.json").write_text(json.dumps(results, indent=2) + "\n")
differences = results[1]["weekly_differences"]

with plt.rc_context(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
):
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    x = list(range(1, 8))
    ax.bar(x, differences, width=0.6, color="0.55", edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    mean = sum(differences) / len(differences)
    ax.axhline(
        mean,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Mean difference: {mean:+.3f}",
    )
    for week, value in zip(x, differences):
        ax.annotate(
            f"{value:+.3f}" if value else "0.000",
            (week, value),
            xytext=(0, 5 if value >= 0 else -5),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    ax.set(
        xlabel="Week",
        ylabel="Mean relevance@7 difference\n(Agent 3 − Agent 1)",
        xticks=x,
        ylim=(-0.12, 0.34),
        yticks=[-0.1, 0, 0.1, 0.2, 0.3],
    )
    ax.legend(frameon=False, loc="upper left")
    destination = ROOT / "reports/figures/weekly-agent3-vs-agent1"
    for extension in ("svg", "png", "pdf"):
        fig.savefig(destination.with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)
print(json.dumps(results, indent=2))
