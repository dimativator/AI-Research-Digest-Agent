"""Render a validated digest as readable terminal text."""

from typing import Any


def format_digest(result: dict[str, Any]) -> str:
    papers = result["papers"]
    lines = [
        f"Research digest for {result['user_id']}",
        f"{result['date_from']} to {result['date_to']} (inclusive, UTC)",
        f"Categories: {', '.join(result['categories'])}",
        "",
    ]

    if result["status"] != "completed":
        lines.append("The digest could not be completed: execution budget exhausted.")
    elif not papers:
        lines.append("No papers were selected for this digest.")
    else:
        count = len(papers)
        lines.append(f"{count} recommended {'paper' if count == 1 else 'papers'}")

        for index, paper in enumerate(papers, start=1):
            feature = result.get("x_feature")
            featured = feature is not None and feature["paper_id"] == paper["paper_id"]
            lines.extend(
                [
                    "",
                    f"{index}. {paper['title']}",
                    *(
                        ["Most discussed on X among the papers checked"]
                        if featured
                        else []
                    ),
                    paper["url"],
                    "",
                    paper["summary"],
                    "",
                    f"Why it matters to you: {paper['relevance']}",
                ]
            )
            code = paper.get("code", {})
            if code.get("repositories"):
                lines.append("Code candidates (not executed):")
                for repo in code["repositories"]:
                    lines.append(f"- {repo['url']} ({repo['relationship']})")
            if featured:
                activity = feature["attention"]
                lines.extend(
                    [
                        "",
                        f"X attention: {activity['unique_accounts']} distinct accounts, "
                        f"{activity['matched_posts']} matching posts in {activity['window']['from']} "
                        f"to {activity['window']['to']}.",
                        f"Collected: {activity['collected_at']}. Search coverage is partial.",
                        "Posts with the most observed engagement:",
                    ]
                )
                for post in feature["top_posts"]:
                    metrics = post["metrics"]
                    counts = ", ".join(
                        f"{metrics.get(name) if metrics.get(name) is not None else 'unknown'} {name}"
                        for name in ("likes", "reposts", "replies")
                    )
                    excerpt = " ".join(post["text"].split())
                    if len(excerpt) > 280:
                        excerpt = excerpt[:277] + "..."
                    lines.extend(
                        [
                            f"- @{post['author']}: {excerpt}",
                            f"  {post['url']}",
                            f"  {counts}",
                        ]
                    )

    if result["limitations"]:
        lines.extend(["", "Limitations:"])
        lines.extend(f"- {note}" for note in result["limitations"])

    stats = result["stats"]
    lines.extend(
        [
            "",
            f"Fetched {stats['fetched_count']} papers; screened {stats['screened']}.",
        ]
    )
    return "\n".join(lines)
