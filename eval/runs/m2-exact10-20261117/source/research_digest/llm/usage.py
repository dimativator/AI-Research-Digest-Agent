"""Usage accounting and estimated API-equivalent costs, not subscription billing."""

from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable


DEFAULT_MODEL = "gpt-5.6-luna"
PRICE_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
COST_NOTE = (
    "API-equivalent estimate using standard GPT-5.6-Luna rates checked 2026-11-17: "
    "$0.20/M input, $1.20/M output; above 272K input, $0.40/M and $1.80/M. "
    "Assumes all input is uncached; excludes cache writes and external tools. "
    "Not actual Codex subscription billing. Unknown usage/model yields null cost."
)


def measure_call(create: Callable[..., Any], records: list[dict], **kwargs: Any) -> Any:
    started = perf_counter()
    record = {"model": kwargs.get("model"), "status": "failed"}
    try:
        response = create(**kwargs)
        usage = getattr(response, "usage", None)
        record.update(
            status="completed",
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
        )
        return response
    finally:
        record["latency_seconds"] = perf_counter() - started
        incoming = record.get("input_tokens")
        outgoing = record.get("output_tokens")
        record["estimated_api_cost_usd"] = None
        if (
            incoming is not None
            and outgoing is not None
            and record["model"] == DEFAULT_MODEL
        ):
            input_rate, output_rate = (
                (0.40, 1.80) if incoming > 272_000 else (0.20, 1.20)
            )
            record["estimated_api_cost_usd"] = (
                incoming * input_rate + outgoing * output_rate
            ) / 1_000_000
        records.append(record)


def summarize_usage(records: list[dict]) -> dict[str, Any]:
    result = {
        "recorded_calls": len(records),
        "llm_latency_seconds": sum(r["latency_seconds"] for r in records),
    }
    for key in ("input_tokens", "output_tokens", "estimated_api_cost_usd"):
        values = [r.get(key) for r in records]
        result[key] = sum(values) if all(v is not None for v in values) else None
    result["calls_without_usage"] = sum(
        r.get("input_tokens") is None or r.get("output_tokens") is None for r in records
    )
    result["cost_note"] = COST_NOTE
    result["price_source"] = PRICE_SOURCE
    return result


class MeteredClient:
    """Wrap Responses calls without changing prompts or response objects."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.records: list[dict] = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs: Any) -> Any:
        return measure_call(self.client.responses.create, self.records, **kwargs)
