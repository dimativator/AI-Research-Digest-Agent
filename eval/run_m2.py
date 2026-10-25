"""Compare one-call relevance judgments with the production ReAct harness."""

import argparse
import json
import statistics
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from research_digest.agent.config import DigestConfig
from research_digest.agent.digest_agent import DigestAgent
from research_digest.llm.codex_llm import CodexOpenAI
from research_digest.profiles.user_profile import UserProfile
from research_digest.retrieval.arxiv_source import Page, Paper
from research_digest.storage.paper_store import PaperStore


BASELINE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"relevant": {"type": "boolean"}},
    "required": ["relevant"],
}


class FrozenArxiv:
    def __init__(self, paper: Paper) -> None:
        self.paper = paper

    def page(self, query: str, start: int, size: int) -> Page:
        date_range = query.partition("submittedDate:[")[2].partition("]")[0]
        start_date, end_date = date_range.split(" TO ")
        published = datetime.fromisoformat(
            self.paper.published.replace("Z", "+00:00")
        ).strftime("%Y%m%d%H%M")
        matches = start_date <= published <= end_date
        return Page(int(matches), start, [self.paper] if matches and start == 0 else [])

    def document(self, paper: Paper) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "source": paper.url,
            "text": "",
            "chunks": [],
            "links": [],
            "reason": "No full-text fixture is included in this frozen sample.",
        }


class DisabledX:
    def close(self) -> None:
        pass


def metrics(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    true_positive = sum(x["prediction"] and x["gold"] for x in records)
    false_positive = sum(x["prediction"] and not x["gold"] for x in records)
    false_negative = sum(not x["prediction"] and x["gold"] for x in records)
    correct = sum(x["prediction"] == x["gold"] for x in records)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "tasks": len(records),
        "accuracy": correct / len(records) if records else None,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_latency_seconds": statistics.mean(x["latency_seconds"] for x in records) if records else None,
        "mean_steps_per_task": statistics.mean(x["steps"] for x in records),
        "mean_llm_calls_per_task": statistics.mean(x["llm_calls"] for x in records),
        "mean_tool_calls_per_task": statistics.mean(x["tool_calls"] for x in records),
        "cost_usd": None,
    }


def baseline_one_call(client: CodexOpenAI, profile: dict[str, Any], task: dict[str, Any], model: str | None) -> bool:
    response = client.responses.create(
        model=model,
        instructions=(
            "Judge whether this paper is relevant to the researcher's stated interests. "
            "Use only the supplied profile and abstract. Return the boolean judgment."
        ),
        input=json.dumps(
            {
                "profile": {
                    key: profile[key]
                    for key in ("summary", "topics", "goals", "exclusions")
                },
                "paper": {
                    "paper_id": task["paper_id"],
                    "title": task["title"],
                    "abstract": task["abstract"],
                },
            },
            ensure_ascii=False,
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "relevance_judgment",
                "strict": True,
                "schema": BASELINE_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)["relevant"]


def run_agent_task(
    client: CodexOpenAI,
    profile_data: dict[str, Any],
    task: dict[str, Any],
    sample: dict[str, Any],
    model: str | None,
) -> tuple[bool, dict[str, Any], str | None]:
    paper = Paper(**{key: task[key] for key in Paper.__dataclass_fields__})
    profile = UserProfile.from_dict(profile_data)
    config = DigestConfig(
        categories=tuple(sample["categories"]),
        date_from=sample["date_from"],
        date_to=sample["date_to"],
        model=model,
        shortlist_size=1,
        batch_size=1,
        expansion_size=1,
        max_candidates=1,
        digest_size=1,
        max_steps=4,
        max_llm_calls=8,
        max_inspections=1,
        max_expansions=0,
        x_enabled=False,
    )
    with tempfile.TemporaryDirectory(prefix="digest-m2-eval-") as directory:
        store = PaperStore(Path(directory) / "eval.sqlite3")
        agent = DigestAgent(
            store,
            client=client,
            arxiv=FrozenArxiv(paper),
            x_client=DisabledX(),
        )
        try:
            result = agent.run(agent.start(profile, config))
            status = result["status"]
            prediction = bool(result["papers"])
            error = None
        except Exception as exc:
            status = "failed"
            prediction = False
            error = f"{type(exc).__name__}: {exc}"
        events = store.events(agent.run_id)
        return prediction, {
            "status": status,
            "steps": agent.state.get("steps", 0),
            "llm_calls": agent.state.get("llm_calls", 0),
            "tool_calls": sum(event["type"] == "observation" for event in events),
            "actions": [
                event["decision"]["action"]
                for event in events
                if event["type"] == "action"
            ],
        }, error


def evaluate_method(
    method: str,
    client: CodexOpenAI,
    sample: dict[str, Any],
    model: str | None,
) -> tuple[list[dict[str, Any]], dict[str, float | int | None]]:
    records = []
    profile = sample["profile"]
    for task in sample["tasks"]:
        started = time.perf_counter()
        if method == "baseline":
            try:
                prediction = baseline_one_call(client, profile, task, model)
                usage = {"steps": 0, "llm_calls": 1, "tool_calls": 0}
                error = None
            except Exception as exc:
                prediction = False
                usage = {"steps": 0, "llm_calls": 1, "tool_calls": 0}
                error = f"{type(exc).__name__}: {exc}"
        else:
            prediction, usage, error = run_agent_task(
                client, profile, task, sample, model
            )
        records.append(
            {
                "task_id": task["task_id"],
                "gold": task["relevant"],
                "prediction": prediction,
                "correct": prediction == task["relevant"],
                "latency_seconds": round(time.perf_counter() - started, 3),
                **usage,
                "error": error,
            }
        )
    return records, metrics(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("eval/sample.json"))
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, default=Path("eval/results.json"))
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    if len(sample["tasks"]) < 20:
        parser.error("Evaluation requires at least 20 frozen tasks")

    client = CodexOpenAI()
    report = {
        "sample": str(args.sample),
        "sample_tasks": len(sample["tasks"]),
        "model": args.model or "Codex CLI default",
        "cost_note": "CodexOpenAI does not expose token usage or price; cost is unknown.",
        "methods": {},
    }
    for method in ("baseline", "agent"):
        records, summary = evaluate_method(method, client, sample, args.model)
        report["methods"][method] = {"summary": summary, "tasks": records}
        print(f"{method}: {json.dumps(summary, sort_keys=True)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved detailed results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())