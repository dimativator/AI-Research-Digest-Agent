"""Match X posts to papers and aggregate observed attention without an LLM."""

import re
import unicodedata
from datetime import datetime, timezone
from typing import Any

from .arxiv_source import Paper


def normalized_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def parse_count(value: str | None) -> int | None:
    """Read English compact UI counts; missing metrics remain unknown."""
    if value is None:
        return None
    match = re.fullmatch(r"([\d,.]+)\s*([KMB]?)", value.strip(), re.I)
    if not match:
        return None
    number = float(match[1].replace(",", ""))
    return int(
        number
        * {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[match[2].upper()]
    )


def match_post(post: dict[str, Any], paper: Paper) -> str | None:
    text = " ".join([post["text"], *post.get("links", [])])
    # Accept abs/pdf/html links and versioned IDs, without matching longer IDs.
    if re.search(
        rf"(?<![\w.]){re.escape(paper.paper_id)}(?:v\d+)?(?:\.pdf)?(?![\w.])", text
    ):
        return "arxiv_id"
    title = normalized_text(paper.title)
    if len(title.split()) >= 4 and title in normalized_text(post["text"]):
        return "exact_title"
    return None


def aggregate_attention(
    paper: Paper,
    posts: list[dict[str, Any]],
    date_from: str,
    date_to: str,
    *,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Deduplicate verified matches, retaining collection and measurement scope."""
    unique = {}
    for post in posts:
        match = match_post(post, paper)
        try:
            posted_date = (
                datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
                .astimezone(timezone.utc)
                .date()
                .isoformat()
            )
        except (KeyError, ValueError):
            continue
        if not match or not date_from <= posted_date <= date_to:
            continue
        unique[post["id"]] = {**post, "match": match}

    matched = sorted(unique.values(), key=post_attention_key, reverse=True)
    accounts = {post["author"].casefold() for post in matched}
    errors = errors or []
    status = "ok" if matched else ("unavailable" if errors else "no_matches")
    total = {}
    missing = {}
    for name in ("likes", "reposts", "replies", "quotes"):
        observed = [post["metrics"].get(name) for post in matched]
        values = [value for value in observed if value is not None]
        total[name] = (
            sum(values) if values else (0 if not matched and not errors else None)
        )
        missing[name] = len(observed) - len(values)

    count = len(accounts)
    level = (
        "unavailable"
        if status == "unavailable"
        else (
            "none"
            if not count
            else "low" if count < 3 else "moderate" if count < 10 else "high"
        )
    )
    return {
        "paper_id": paper.paper_id,
        "status": status,
        "level": level,
        "window": {"from": date_from, "to": date_to},
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "coverage": "partial",
        "note": "Sampled X Latest searches. Engagement counts are snapshots at collection time, not counts restricted to the post-date window. Attention is not scientific quality.",
        "matched_posts": len(matched),
        "unique_accounts": count,
        "engagement": total,
        "missing_metrics": missing,
        "posts": matched,
        "errors": errors,
    }


def post_attention_key(post: dict[str, Any]) -> tuple[int, int, int, str]:
    metrics = post["metrics"]
    counts = [
        metrics.get(name) or 0 for name in ("likes", "reposts", "replies", "quotes")
    ]
    return (
        sum(counts),
        metrics.get("reposts") or 0,
        metrics.get("likes") or 0,
        post["id"],
    )


def attention_summary(attention: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in attention.items()
        if key not in {"posts", "queries"}
    }


def featured_paper(attentions: dict[str, dict[str, Any]]) -> str | None:
    eligible = [
        item
        for item in attentions.values()
        if item["status"] == "ok" and item["matched_posts"] > 0
    ]
    if not eligible:
        return None
    winner = max(
        eligible,
        key=lambda item: (
            item["unique_accounts"],
            item["engagement"].get("reposts") or 0,
            item["engagement"].get("likes") or 0,
            item["paper_id"],
        ),
    )
    return winner["paper_id"]
