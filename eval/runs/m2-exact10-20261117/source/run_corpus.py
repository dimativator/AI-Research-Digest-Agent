"""Diagnose the production ReAct agent on a fixed, privately labeled corpus."""

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from research_digest.agent.config import DigestConfig
from research_digest.agent.digest_agent import DigestAgent
from research_digest.llm.codex_llm import CodexOpenAI
from research_digest.llm.usage import DEFAULT_MODEL, summarize_usage
from research_digest.profiles.user_profile import UserProfile
from research_digest.retrieval.arxiv_source import ArxivClient, Paper
from research_digest.retrieval.code_source import CodeFinder
from research_digest.retrieval.ranking import bm25
from research_digest.storage.paper_store import PaperStore


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class RecordedClient:
    def __init__(self, path):
        self.path = path
        self.client = CodexOpenAI()
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        record = {"request": kwargs, "variant": "react"}
        try:
            response = self.client.responses.create(**kwargs)
            record["response"] = response.output_text
            usage = getattr(response, "usage", None)
            record["usage"] = vars(usage) if usage else None
            return response
        except Exception as error:
            record["error"] = repr(error)
            raise
        finally:
            with self.path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class Fixtures:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir(exist_ok=True)
        self.arxiv = ArxivClient()
        self.code = CodeFinder()
        self.capture_seconds = 0.0

    def get(self, kind, paper, fetch):
        path = self.directory / f"{kind}-{paper.fingerprint}.json"
        if not path.exists():
            started = time.perf_counter()
            result = fetch()
            self.capture_seconds += time.perf_counter() - started
            save(path, result)
        return json.loads(path.read_text())

    def document(self, paper):
        return self.get("document", paper, lambda: self.arxiv.document(paper))

    def find(self, paper, document):
        return self.get("code", paper, lambda: self.code.find(paper, document))


def score(selected, labels, k=5):
    gains = [labels[pid] for pid in selected[:k]]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal = sum(
        gain / math.log2(i + 2)
        for i, gain in enumerate(sorted(labels.values(), reverse=True)[:k])
    )
    return {
        f"ndcg_at_{k}": dcg / ideal if ideal else 0.0,
        "selected_labels": gains,
        "interesting": gains.count(1),
        "potentially_interesting": gains.count(0.5),
        "not_interesting": gains.count(0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sample", type=Path, default=Path("labeling/labeled_sample.json")
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixtures-from", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--digest-size", type=int, default=5)
    args = parser.parse_args()
    if args.digest_size < 1:
        parser.error("--digest-size must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    sample = json.loads(args.sample.read_text())
    profile_data = json.loads(Path("eval/sample.json").read_text())["profile"]
    profile = UserProfile.from_dict(profile_data)
    papers = [
        Paper(**{k: row[k] for k in Paper.__dataclass_fields__})
        for row in sample["papers"]
    ]
    labels = {row["paper_id"]: row["label"] for row in sample["papers"]}
    assert len(labels) == len(papers) == 100
    assert all(label in (0, 0.5, 1) for label in labels.values())
    ranked = bm25(papers, profile.ranking_text())
    config = DigestConfig(
        categories=tuple(sorted({cat for p in papers for cat in p.categories})),
        date_from=min(p.published[:10] for p in papers),
        date_to=max(p.published[:10] for p in papers),
        model=args.model,
        shortlist_size=30,
        digest_size=args.digest_size,
        max_candidates=100,
        max_steps=-1,
        max_llm_calls=-1,
        max_inspections=-1,
        max_expansions=-1,
        max_context_chars=-1,
        x_enabled=False,
    )
    save(args.output / "sample.json", sample)
    save(args.output / "profile.json", profile_data)
    save(
        args.output / "manifest.json",
        {
            "config": asdict(config),
            "variant": "react",
            "repetitions": args.repetitions,
            "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "codex_version": subprocess.check_output(
                ["codex", "--version"], text=True
            ).strip(),
            "protocol": "Fixed corpus bypasses live metadata harvesting. First-use tool responses captured and replayed; labels never sent to model. Unlimited CLI-style budgets.",
        },
    )
    (args.output / "working_changes.patch").write_bytes(
        subprocess.check_output(["git", "diff"])
    )
    source_dir = args.output / "source"
    source_dir.mkdir()
    import shutil

    shutil.copytree(
        "research_digest",
        source_dir / "research_digest",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(__file__, source_dir / "run_corpus.py")
    save(
        args.output / "bm25.json",
        [
            {
                "paper_id": r.paper.paper_id,
                "score": r.score,
                "gold": labels[r.paper.paper_id],
            }
            for r in ranked
        ],
    )
    if args.fixtures_from:
        shutil.copytree(args.fixtures_from, args.output / "fixtures")
    fixtures = Fixtures(args.output / "fixtures")
    summaries = []
    for repetition in range(1, args.repetitions + 1):
        directory = args.output / f"run-{repetition}"
        directory.mkdir()

        def event(value):
            value = {"variant": "react", "repetition": repetition, **value}
            with (directory / "trace.jsonl").open("a") as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
            print(repetition, value["type"], value.get("purpose", ""), flush=True)

        store = PaperStore(directory / "run.sqlite3")
        agent = DigestAgent(
            store,
            client=RecordedClient(directory / "calls.jsonl"),
            arxiv=fixtures,
            code_finder=fixtures,
            on_event=event,
        )
        run_id = agent.start(profile, config)
        store.freeze_candidates(run_id, ranked)
        agent.state.update(
            phase="screen", fetched_count=len(papers), candidate_count=len(papers)
        )
        store.update_run(run_id, agent.state)
        started = time.perf_counter()
        capture_before = fixtures.capture_seconds
        try:
            result = agent.run(run_id)
            error = None
        except Exception as exc:
            result = {"status": "failed", "papers": []}
            error = repr(exc)
        elapsed = time.perf_counter() - started
        save(directory / "result.json", result)
        save(directory / "state.json", agent.state)
        selected = [p["paper_id"] for p in result["papers"]]
        summary = {
            "repetition": repetition,
            "run_id": run_id,
            "status": result["status"],
            "error": error,
            "latency_seconds": elapsed,
            "fixture_capture_seconds": fixtures.capture_seconds - capture_before,
            "selected": selected,
            "metrics": (
                score(selected, labels, args.digest_size)
                if result["status"] == "completed"
                else None
            ),
            "stats": result.get("stats")
            or {
                "llm_calls": agent.state.get("llm_calls", 0),
                "steps": agent.state.get("steps", 0),
                "tool_calls": agent.state.get("tool_calls", 0),
                "screened": len(agent.state.get("screened", {})),
                "usage": summarize_usage(agent.state.get("llm_usage", [])),
            },
        }
        summaries.append(summary)
        save(args.output / "summary.json", summaries)
        print(json.dumps(summary), flush=True)
    save(
        args.output / "fixture_hashes.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (args.output / "fixtures").glob("*.json")
        },
    )


if __name__ == "__main__":
    main()
