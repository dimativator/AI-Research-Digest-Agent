"""Frozen 1,000-paper stress test: 20 independent runs of each digest agent.

python -m eval.run_repeated --prepare-only
python -m eval.run_repeated
python -m eval.run_repeated --analyze-only
"""

import argparse
import hashlib
import itertools
import json
import random
import shutil
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from eval.run_weekly import SharedFixtures, VARIANTS, run_agent, save
from research_digest.profiles.user_profile import UserProfile
from research_digest.retrieval.arxiv_source import Paper
from research_digest.retrieval.e5_reranker import rerank_bm25_head
from research_digest.retrieval.ranking import bm25

SOURCE = Path("eval/runs/weekly-20261118")
OUTPUT = Path("eval/runs/repeated-1000-20261118")
SEED = 2026111801


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path) -> dict:
    if (root / "manifest.json").exists():
        manifest = read(root / "manifest.json")
        for name, digest in manifest["source_sha256"].items():
            if sha(Path(name)) != digest:
                raise ValueError(f"Source changed after freeze: {name}")
        for name, digest in manifest["input_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError(f"Frozen input changed: {name}")
        return manifest
    root.mkdir(parents=True, exist_ok=False)
    labeled = read(SOURCE / "analysis/labeled-sample.json")["papers"]
    human_ids = {p["paper_id"] for p in labeled}
    assert len(labeled) == len(human_ids) == 122
    universe = {}
    source_hashes = {}
    for path in sorted(SOURCE.glob("week-*/corpus.json")):
        source_hashes[str(path)] = sha(path)
        for paper in read(path):
            assert paper["paper_id"] not in universe
            universe[paper["paper_id"]] = paper
    rng = random.Random(SEED)
    random_ids = rng.sample(sorted(set(universe) - human_ids), 878)
    dataset = [
        {
            **{k: p[k] for k in Paper.__dataclass_fields__},
            "label": p["label"],
            "label_source": "human_weekly_122",
            "note": p.get("note", ""),
            "label_timestamp": p.get("updated_at"),
        }
        for p in labeled
    ]
    dataset += [
        {**universe[pid], "label": 0, "label_source": "assumed_negative_random"}
        for pid in random_ids
    ]
    rng.shuffle(dataset)
    assert len({p["paper_id"] for p in dataset}) == 1000
    save(
        root / "sample.json",
        dict(
            seed=SEED,
            papers=dataset,
            source_corpus_sha256=source_hashes,
            human_labels_sha256=sha(SOURCE / "analysis/labeled-sample.json"),
            sampling="878 uniformly sampled without replacement from seven-week frozen arXiv metadata, excluding 122 human-labeled IDs; assumed zero labels, not human judgments",
        ),
    )
    corpus = [{k: p[k] for k in Paper.__dataclass_fields__} for p in dataset]
    save(root / "corpus.json", corpus)
    shutil.copyfile(SOURCE / "profile.json", root / "profile.json")
    profile = UserProfile.from_dict(read(root / "profile.json"))
    ranked = bm25([Paper(**p) for p in corpus], profile.ranking_text())
    reranked, details = rerank_bm25_head(ranked, profile.ranking_text())
    save(root / "e5-details.json", details)
    rankings = {
        name: [dict(paper_id=r.paper.paper_id, bm25_score=r.score) for r in values]
        for name, values in [("bm25", ranked), ("e5", reranked)]
    }
    save(root / "rankings.json", rankings)
    labels = {p["paper_id"]: p["label"] for p in dataset}
    coverage = {}
    for name, ranking in rankings.items():
        ids = [p["paper_id"] for p in ranking[:30]]
        n1 = sum(labels[p] == 1 for p in ids)
        nhalf = sum(labels[p] == 0.5 for p in ids)
        coverage[name] = dict(
            top30_ids=ids,
            interesting=n1,
            potentially_interesting=nhalf,
            interesting_coverage=n1 / 31,
            positive_or_partial_coverage=(n1 + nhalf) / 76,
            assumed_negatives=sum(p not in human_ids for p in ids),
        )
    save(root / "shortlist-coverage.json", coverage)
    shutil.copytree(SOURCE / "fixtures", root / "fixtures")
    paths = list(Path("research_digest").rglob("*.py")) + [
        Path("eval/run_repeated.py"),
        Path("eval/run_weekly.py"),
        Path("eval/run_corpus.py"),
        Path("environment.yml"),
    ]
    for path in paths:
        target = root / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        model="gpt-6-sol",
        repetitions=20,
        digest_size=10,
        shortlist_size=30,
        seed=SEED,
        categories=list(profile.arxiv_categories),
        variants=list(VARIANTS),
        date_from=min(p["published"][:10] for p in corpus),
        date_to=max(p["published"][:10] for p in corpus),
        source_sha256={str(p): sha(p) for p in paths},
        input_sha256={
            name: sha(root / name)
            for name in [
                "sample.json",
                "corpus.json",
                "profile.json",
                "rankings.json",
                "e5-details.json",
            ]
        },
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        codex_version=subprocess.check_output(
            ["codex", "--version"], text=True
        ).strip(),
        protocol="60 runs, fresh databases and screening per run; no labels supplied to models; unlimited budgets; X disabled; three concurrent workers, deterministic shuffled job order; shared cached/first-use tool responses; preserve failures without replacement.",
        interpretation="One enriched fixed corpus with assumed negatives: assesses stochastic repeatability, not generalization across tasks or unbiased relevance. Coverage concerns known positives only.",
    )
    (root / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff"]))
    save(root / "manifest.json", manifest)
    return manifest


def analyze(root: Path, update_report: bool = False) -> dict:
    sample = read(root / "sample.json")["papers"]
    labels = {p["paper_id"]: p["label"] for p in sample}
    human = {p["paper_id"] for p in sample if p["label_source"] == "human_weekly_122"}
    rows = []
    for path in sorted(root.glob("repeat-*/*/outcome.json")):
        outcome = read(path)
        selected = outcome["selected"]
        complete = outcome["status"] == "completed"
        if complete:
            assert (
                len(selected) == len(set(selected)) == 10
                and set(selected) <= labels.keys()
            )
        row = {
            **outcome,
            "relevance_at_10": (
                sum(labels[p] for p in selected) / 10 if complete else None
            ),
            "interesting": sum(labels[p] == 1 for p in selected),
            "potentially_interesting": sum(labels[p] == 0.5 for p in selected),
            "assumed_negative_selections": sum(p not in human for p in selected),
        }
        rows.append(row)
    summaries = {}
    for variant in VARIANTS:
        all_rows = [r for r in rows if r["variant"] == variant]
        ok = [r for r in all_rows if r["status"] == "completed"]
        values = [r["relevance_at_10"] for r in ok]
        overlaps = [
            len(set(a["selected"]) & set(b["selected"])) / 10
            for a, b in itertools.combinations(ok, 2)
        ]
        summaries[variant] = dict(
            finished=len(all_rows),
            completed=len(ok),
            failed=len(all_rows) - len(ok),
            mean=statistics.mean(values) if values else None,
            std=statistics.stdev(values) if len(values) > 1 else None,
            minimum=min(values) if values else None,
            maximum=max(values) if values else None,
            mean_pairwise_selection_overlap=(
                statistics.mean(overlaps) if overlaps else None
            ),
            mean_assumed_negative_selections=(
                statistics.mean(r["assumed_negative_selections"] for r in ok)
                if ok
                else None
            ),
        )
    result = dict(
        runs=rows, summary=summaries, coverage=read(root / "shortlist-coverage.json")
    )
    save(root / "analysis.json", result)
    if update_report:
        assert len(rows) == 60
        lines = [
            "### Repeated runs on a 1,000-paper corpus",
            "",
            "We kept all 122 weekly human-labeled papers (31 interesting, 45 potentially interesting, 46 not interesting) and sampled 878 distinct papers from the saved seven-week arXiv corpus, using seed `2026111801`. The additional papers were assigned zero as assumed negatives, not human judgments. The frozen corpus, profile, source snapshot, and labels were saved. Each agent ran 20 times on the same corpus with `gpt-6-sol`, fresh screening and run state, a 30-paper initial shortlist, and an exact 10-paper digest. X was disabled and budgets were unlimited. Tool responses were shared through cached/first-use fixtures.",
            "",
            "| Agent | Interesting in top 30 (of 31) | Interesting or potential in top 30 (of 76) | Completed | Mean relevance@10 ± SD | Min–max | Mean selection overlap |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: |",
        ]
        for index, variant in enumerate(VARIANTS, 1):
            s = summaries[variant]
            c = result["coverage"]["bm25" if variant == "react" else "e5"]
            stats = (
                f"{s['mean']:.3f} ± {s['std']:.3f}"
                if s["std"] is not None
                else "Unavailable"
            )
            span = (
                f"{s['minimum']:.3f}–{s['maximum']:.3f}"
                if s["mean"] is not None
                else "Unavailable"
            )
            overlap = (
                f"{100*s['mean_pairwise_selection_overlap']:.1f}%"
                if s["mean_pairwise_selection_overlap"] is not None
                else "Unavailable"
            )
            lines.append(
                f"| Agent {index} | {c['interesting']}/31 ({100*c['interesting_coverage']:.1f}%) | {c['interesting']+c['potentially_interesting']}/76 ({100*c['positive_or_partial_coverage']:.1f}%) | {s['completed']}/20 | {stats} | {span} | {overlap} |"
            )
        lines += [
            "",
            "Coverage is measured before LLM screening and is fixed across repetitions: Agent 1 uses BM25, while Agents 2 and 3 use E5 reranking of the BM25 top 200. SD measures run-to-run variation on this one corpus, not uncertainty across independent tasks. Selection overlap is the shared fraction of the top 10, averaged over all pairs of completed runs. Failed runs are reported separately and excluded from relevance summaries. Assumed-negative labels can penalize genuinely relevant discoveries, so the scores and coverage apply to this enriched benchmark and its known positives.",
            "",
            "[Frozen dataset](../eval/runs/repeated-1000-20261118/sample.json) · [Per-run results and coverage](../eval/runs/repeated-1000-20261118/analysis.json) · [Configuration and source hashes](../eval/runs/repeated-1000-20261118/manifest.json). Reproduce with `python -m eval.run_repeated` from the project environment. Existing completed or failed outcomes are preserved on restart.",
            "",
        ]
        report = Path("reports/M3.md")
        text = report.read_text()
        heading = "### Repeated runs on a 1,000-paper corpus"
        if heading in text:
            start = text.index(heading)
            end = text.index("## Reproducibility", start)
            text = text[:start] + text[end:]
        text = text.replace(
            "## Reproducibility and assessment scope",
            "\n".join(lines) + "\n## Reproducibility and assessment scope",
        )
        report.write_text(text)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        result = analyze(args.output)
        if len(result["runs"]) == 60:
            analyze(args.output, True)
        print(json.dumps(result["summary"], indent=2))
        return
    manifest = prepare(args.output)
    if args.prepare_only:
        return
    jobs = []
    for n in range(1, 21):
        task = f"repeat-{n:02d}"
        directory = args.output / task
        directory.mkdir(exist_ok=True)
        for name in ["corpus.json", "rankings.json", "e5-details.json"]:
            if not (directory / name).exists():
                shutil.copyfile(args.output / name, directory / name)
        window = dict(
            task=task, date_from=manifest["date_from"], date_to=manifest["date_to"]
        )
        jobs.extend((window, variant) for variant in VARIANTS)
    random.Random(SEED).shuffle(jobs)
    fixtures = SharedFixtures(args.output / "fixtures")

    def execute(job):
        window, variant = job
        run_agent(args.output, manifest, window, variant, fixtures, digest_size=10)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(execute, jobs))
    result = analyze(args.output, True)
    save(
        args.output / "fixture-hashes.json",
        {p.name: sha(p) for p in (args.output / "fixtures").glob("*.json")},
    )
    print("COMPLETED", json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
