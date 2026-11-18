"""Freeze seven weekly corpora, compare five systems, export a blind labeling pool."""

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from eval.run_corpus import Fixtures, RecordedClient
from research_digest.agent.config import DigestConfig
from research_digest.agent.digest_agent import DigestAgent
from research_digest.profiles.user_profile import ProfileStore, UserProfile
from research_digest.retrieval.arxiv_source import ArxivClient, Paper, collect_metadata
from research_digest.retrieval.e5_reranker import rerank_bm25_head
from research_digest.retrieval.ranking import RankedPaper, bm25
from research_digest.storage.paper_store import PaperStore

VARIANTS = ("react", "react-explore", "debate")
SYSTEMS = ("bm25", "e5", *VARIANTS)
SEED = 20261118


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def weekly_windows(as_of: date) -> list[dict]:
    last_sunday = as_of - timedelta(days=as_of.weekday() + 1)
    return [
        dict(
            task=f"week-{i + 1}",
            date_from=(last_sunday - timedelta(days=48 - 7 * i)).isoformat(),
            date_to=(last_sunday - timedelta(days=42 - 7 * i)).isoformat(),
        )
        for i in range(7)
    ]


class SharedFixtures(Fixtures):
    """Serialize first-use fetches; all systems replay the same saved response."""

    def __init__(self, directory: Path):
        super().__init__(directory)
        self.lock = threading.Lock()

    def get(self, kind, paper, fetch):
        with self.lock:
            return super().get(kind, paper, fetch)


def initialize(root: Path, args: argparse.Namespace) -> dict:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError(
                "Output exists; use --resume to continue saved configuration"
            )
        manifest = json.loads(manifest_path.read_text())
        for name, digest in manifest["source_sha256"].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed since freeze: {name}")
        return manifest
    root.mkdir(parents=True)
    profile = ProfileStore(args.db).load("dmitrii").profile
    save(root / "profile.json", asdict(profile))
    files = list(Path("research_digest").rglob("*.py")) + [
        Path("eval/run_weekly.py"),
        Path("eval/run_corpus.py"),
        Path("environment.yml"),
    ]
    hashes = {}
    for source in files:
        target = root / "source" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = dict(
        created_at=datetime.now(timezone.utc).isoformat(),
        as_of=args.as_of,
        timezone="UTC",
        windows=weekly_windows(date.fromisoformat(args.as_of)),
        model="gpt-6-sol",
        digest_size=digest_size,
        repetitions=1,
        systems=SYSTEMS,
        shuffle_seed=SEED,
        categories=list(profile.arxiv_categories),
        profile_sha256=hashlib.sha256((root / "profile.json").read_bytes()).hexdigest(),
        source_sha256=hashes,
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        codex_version=subprocess.check_output(
            ["codex", "--version"], text=True
        ).strip(),
        protocol={
            "primary_metric": "mean relevance@7 with labels 0/0.5/1",
            "comparison_unit": "weekly task; paired systems",
            "budgets": "unlimited",
            "x_enabled": False,
            "metadata": "UTC initial submission dates; freeze metadata and versions on collection; cached daily API pages may predate launch",
            "tools": "First-use capture with shared replay including unavailable responses; external world was not frozen before initial capture",
            "labels": "Blind pooled judgments; no old labels copied",
            "failures": "Preserve failures, never replace with a fresh successful attempt; completed jobs skipped on resume",
            "exposure": "Requested latest seven weeks overlap development/live runs. This is not an untouched holdout.",
            "randomness": "Model identifier pinned; hosted model output is not guaranteed bitwise reproducible; shuffle and inputs are reproducible",
        },
    )
    save(manifest_path, manifest)
    (root / "working-tree.patch").write_bytes(subprocess.check_output(["git", "diff"]))
    (root / "python-packages.txt").write_bytes(
        subprocess.check_output([__import__("sys").executable, "-m", "pip", "freeze"])
    )
    return manifest


def prepare(
    root: Path, manifest: dict, cache: PaperStore, profile: UserProfile
) -> None:
    client = ArxivClient()
    for window in manifest["windows"]:
        directory = root / window["task"]
        directory.mkdir(exist_ok=True)
        corpus = directory / "corpus.json"
        if not corpus.exists():
            print("FETCH", window, flush=True)
            papers = collect_metadata(
                client,
                cache,
                tuple(manifest["categories"]),
                window["date_from"],
                window["date_to"],
            )
            save(corpus, [asdict(p) for p in papers])
        papers = [Paper(**p) for p in json.loads(corpus.read_text())]
        if len(papers) < 7:
            raise ValueError(f'{window["task"]}: fewer than seven papers')
        if not (directory / "rankings.json").exists():
            ranked = bm25(papers, profile.ranking_text())
            e5, details = rerank_bm25_head(ranked, profile.ranking_text())
            save(directory / "e5-details.json", details)
            save(
                directory / "rankings.json",
                {
                    name: [
                        {"paper_id": r.paper.paper_id, "bm25_score": r.score}
                        for r in rows
                    ]
                    for name, rows in [("bm25", ranked), ("e5", e5)]
                },
            )
        rankings = json.loads((directory / "rankings.json").read_text())
        for name in ("bm25", "e5"):
            save(
                directory / f"{name}.json",
                {
                    "status": "completed",
                    "selected": [p["paper_id"] for p in rankings[name][:7]],
                },
            )
        print("READY", window["task"], len(papers), "papers", flush=True)


def run_agent(
    root: Path,
    manifest: dict,
    window: dict,
    variant: str,
    fixtures: SharedFixtures,
    digest_size: int = 7,
) -> None:
    directory = root / window["task"] / variant
    directory.mkdir(exist_ok=True)
    if (directory / "outcome.json").exists():
        return
    profile = UserProfile.from_dict(json.loads((root / "profile.json").read_text()))
    corpus = {
        p["paper_id"]: Paper(**p)
        for p in json.loads((directory.parent / "corpus.json").read_text())
    }
    ranking_name = "bm25" if variant == "react" else "e5"
    ranks = json.loads((directory.parent / "rankings.json").read_text())[ranking_name]
    ranked = [RankedPaper(corpus[r["paper_id"]], r["bm25_score"]) for r in ranks]
    config = DigestConfig(
        categories=tuple(manifest["categories"]),
        date_from=window["date_from"],
        date_to=window["date_to"],
        model=manifest["model"],
        variant=variant,
        digest_size=digest_size,
        shortlist_size=30,
        max_candidates=-1,
        max_steps=-1,
        max_llm_calls=-1,
        max_inspections=-1,
        max_expansions=-1,
        max_context_chars=-1,
        x_enabled=False,
    )
    save(directory / "config.json", asdict(config))

    def event(value):
        value = {**value, "variant": variant, "task": window["task"]}
        with (directory / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        print(
            window["task"], variant, value["type"], value.get("purpose", ""), flush=True
        )

    store = PaperStore(directory / "run.sqlite3")
    agent = DigestAgent(
        store,
        client=RecordedClient(directory / "calls.jsonl", variant),
        arxiv=fixtures,
        code_finder=fixtures,
        on_event=event,
    )
    if (directory / "run-id.txt").exists():
        run_id = (directory / "run-id.txt").read_text().strip()
    else:
        run_id = agent.start(profile, config)
        (directory / "run-id.txt").write_text(run_id)
        store.freeze_candidates(run_id, ranked)
        agent.state.update(
            phase="screen", fetched_count=len(corpus), candidate_count=len(corpus)
        )
        if variant != "react":
            agent.state["retrieval_reranking"] = json.loads(
                (directory.parent / "e5-details.json").read_text()
            )
        agent._persist_run()
    started = time.perf_counter()
    try:
        result = agent.run(run_id)
        error = None
    except Exception as exc:
        result = {"status": "failed", "papers": []}
        error = repr(exc)
    save(directory / "result.json", result)
    save(directory / "state.json", agent.state)
    save(
        directory / "outcome.json",
        dict(
            task=window["task"],
            variant=variant,
            run_id=run_id,
            status=result["status"],
            error=error,
            selected=[p["paper_id"] for p in result["papers"]],
            latency_this_invocation_seconds=time.perf_counter() - started,
        ),
    )


def export_pool(root: Path, manifest: dict) -> dict:
    pool = {}
    mapping = []
    prior_path = Path("labeling/labeled_sample.json")
    prior = (
        {p["paper_id"] for p in json.loads(prior_path.read_text())["papers"]}
        if prior_path.exists()
        else set()
    )
    for window in manifest["windows"]:
        directory = root / window["task"]
        papers = {
            p["paper_id"]: p
            for p in json.loads((directory / "corpus.json").read_text())
        }
        for system in SYSTEMS:
            path = (
                directory / f"{system}.json"
                if system in ("bm25", "e5")
                else directory / system / "outcome.json"
            )
            outcome = json.loads(path.read_text())
            mapping.append({"task": window["task"], "system": system, **outcome})
            for pid in outcome["selected"]:
                if pid not in papers:
                    raise ValueError(f"Selection outside frozen corpus: {pid}")
                if pid in pool and pool[pid] != papers[pid]:
                    raise ValueError(f"Inconsistent versions for pooled paper: {pid}")
                pool[pid] = papers[pid]
    rows = [pool[pid] for pid in sorted(pool)]
    random.Random(manifest["shuffle_seed"]).shuffle(rows)
    sample = {
        "schema_version": 1,
        "sample_id": "weekly-blind-20261118",
        "label_scale": {
            "0": "not interesting",
            "0.5": "potentially interesting",
            "1": "interesting",
        },
        "papers": rows,
    }
    save(root / "blind-sample.json", sample)
    private = {
        "selections": mapping,
        "unique_papers": len(rows),
        "previously_labeled_ids": sorted(set(pool) & prior),
    }
    save(root / "private-selection-map.json", private)
    save(
        root / "artifact-hashes.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*.json")
            if p.name != "artifact-hashes.json"
        },
    )
    print(
        "BLIND SAMPLE READY:",
        len(rows),
        "papers;",
        len(private["previously_labeled_ids"]),
        "previously labeled",
        flush=True,
    )
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("data/digest.sqlite3"))
    parser.add_argument("--as-of", default="2026-11-18")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = initialize(args.output, args)
    profile = UserProfile.from_dict(
        json.loads((args.output / "profile.json").read_text())
    )
    prepare(args.output, manifest, PaperStore(args.db), profile)
    fixtures = SharedFixtures(args.output / "fixtures")
    jobs = [(w, v) for w in manifest["windows"] for v in VARIANTS]
    random.Random(SEED).shuffle(jobs)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(run_agent, args.output, manifest, w, v, fixtures)
            for w, v in jobs
        ]
        for future in futures:
            future.result()
    export_pool(args.output, manifest)


if __name__ == "__main__":
    main()
