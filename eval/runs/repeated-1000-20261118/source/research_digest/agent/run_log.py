"""A readable timeline with references to separately stored full records."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def brief(value: object, limit: int = 300) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def describe(event: dict[str, Any]) -> str:
    kind = event["type"]
    if kind == "export_note":
        return event["note"]
    if kind == "reranking_started":
        return f"E5 reranking: scoring {event['pool_size']} BM25 candidates"
    if kind == "reranking_complete":
        return f"E5 reranking complete: {event['pool_size']} candidates; {event['latency_seconds']:.2f}s; profile truncated={event.get('profile_truncated', False)}"
    if kind == "debate_review":
        items = event["review"]["assessments"]
        return f"Debate {event['role']} review: " + "; ".join(
            f"{item['paper_id']}={item['recommendation']} ({brief(item['reason'], 140)})"
            for item in items
        )
    if kind == "debate_disagreements":
        return "Debate disagreements: " + ", ".join(
            item["paper_id"] for item in event["disagreements"]
        )
    if kind == "debate_reconsideration":
        return f"Debate {event['role']} reconsideration: {event['result']['status']}"
    if kind == "selection_check_result":
        review = event["review"]
        if not isinstance(review, dict):
            return "Selection check: invalid response structure"
        return f"Selection check: {brief(review.get('verdict'))}; {brief(review.get('reason'))}"
    if kind == "selection_check_applied":
        return f"Selection check applied: {event['verdict']}; error={brief(event.get('error'))}"
    if kind == "llm_request":
        size = event.get("context", {})
        return f"Model call {event['call']}: {event['purpose']} ({event.get('model')})" + (
            f"; context {size['total_chars']:,} chars (instructions {size['instructions_chars']:,}, payload {size['payload_chars']:,}, schema {size['schema_chars']:,})"
            if size
            else ""
        )
    if kind == "model_input":
        return f"Model call {event['call']}: full input saved ({event['purpose']})"
    if kind == "llm_response":
        return f"Model call {event['call']}: response received"
    if kind == "action":
        decision = event["decision"]
        arguments = {
            key: decision[key]
            for key in ("paper_id", "question", "query", "limit", "reviewer")
            if decision.get(key)
        }
        selected = [item["paper_id"] for item in decision.get("selections", [])]
        return (
            f"Decision: {decision['action']} {brief(json.dumps(arguments))}"
            + (f"; selected {', '.join(selected)}" if selected else "")
            + f"\n  Reason: {brief(decision.get('reason', ''))}"
        )
    if kind == "observation":
        result = event["result"]
        if event["action"] == "get_candidate_overview":
            return (
                f"Candidate overview: {result['corpus_count']} total, "
                f"{result['screened_count']} screened, {result['unseen_count']} unseen; "
                f"screening labels: {result['screening_counts']}"
            )
        return (
            f"Tool {event['action']}: {result.get('status', 'returned')}"
            f"; {len(result.get('passages', []))} passages"
            f"; new papers: {brief(result.get('new_ids', []))}"
            + (f"; {brief(result['reason'])}" if result.get("reason") else "")
        )
    if kind == "run_finished":
        result = event["result"]
        return f"Run {result['status']}: {len(result['papers'])} papers selected"
    if kind == "llm_usage":
        return f"Model usage: {brief(json.dumps({k: v for k, v in event.items() if k in ('input_tokens', 'output_tokens', 'latency_seconds', 'estimated_api_cost_usd', 'status')}))}"
    if "error" in event:
        return f"{kind}: {brief(event['error'])}"
    if kind == "screening_result":
        return "Screened: " + "; ".join(
            f"{item['paper_id']}={item['label']} ({brief(item['reason'], 140)})"
            for item in event["assessments"].values()
        )
    if kind == "screening_progress":
        return f"Screening progress: {event['screened']}/{event['target']}"
    if kind == "selection_stage":
        response = event["response"]
        return f"Selection {event['stage']}: {response.get('verdict', response.get('action', 'saved'))}; {brief(response.get('reason', ''))}"
    return kind.replace("_", " ").capitalize()


class RunLog:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def record(self, event: dict[str, Any]) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        record_id = uuid.uuid4().hex
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record = {"id": record_id, "timestamp": timestamp, "event": event}
        # Persist details first so every timeline reference resolves.
        with (self.directory / "details.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (self.directory / "activity.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{timestamp} {describe(event)} [ref={record_id}]\n")
        return record_id
