"""Discover paper-associated GitHub repositories without cloning or executing code."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from .arxiv_source import Paper
from .x_attention import normalized_text


CODE_FINDER_VERSION = "code-v1"


def github_repository(url: str) -> str | None:
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.scheme != "https" or parsed.hostname != "github.com" or len(parts) < 2:
        return None
    owner, repo = parts[:2]
    repo = repo.removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", repo
    ):
        return None
    if owner.lower() in {"search", "topics", "orgs", "users", "settings", "login"}:
        return None
    return owner + "/" + repo


class CodeFinder:
    def __init__(self, *, timeout: float = 20, token: str | None = None) -> None:
        self.timeout = timeout
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self._last_search = 0.0

    def _request(self, path: str, *, raw: bool = False) -> Any:
        if not path.startswith("/") or "\r" in path or "\n" in path:
            raise ValueError("Invalid GitHub API path")
        if path.startswith("/search/"):
            interval = 2.1 if self.token else 6.1
            time.sleep(max(0, interval - (time.monotonic() - self._last_search)))
            self._last_search = time.monotonic()
        headers = {
            "Accept": (
                "application/vnd.github.raw+json"
                if raw
                else "application/vnd.github+json"
            ),
            "User-Agent": "AI-Research-Digest-Agent",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = Request("https://api.github.com" + path, headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            body = response.read(1_000_001)
        if len(body) > 1_000_000:
            raise ValueError("GitHub response exceeds size limit")
        text = body.decode("utf-8")
        return text if raw else json.loads(text)

    def find(self, paper: Paper, document: dict[str, Any]) -> dict[str, Any]:
        repositories: dict[str, dict[str, Any]] = {}
        errors = []
        for link in document.get("links", []):
            repo = github_repository(link["url"])
            if repo is None or any(
                term in link["section"].casefold()
                for term in ("references", "bibliography")
            ):
                continue
            # A bibliography link or a baseline implementation is not author-provided code.
            context = (link.get("context", "") + " " + link.get("text", "")).strip()
            declared = bool(
                re.search(
                    r"\b(our code|code (?:is )?available|our implementation|project page|source code)\b",
                    context,
                    re.I,
                )
            )
            repositories[repo.casefold()] = {
                "url": "https://github.com/" + repo,
                "relationship": "linked_from_paper" if declared else "unverified",
                "evidence": {
                    "source": document.get("source", paper.url),
                    "quote": context,
                },
                "license": None,
            }

        queries = [
            f'"{paper.paper_id}" in:readme',
            f'"{paper.title.replace(chr(34), " ")[:180]}" in:readme',
        ]
        if not any(
            item["relationship"] == "linked_from_paper"
            for item in repositories.values()
        ):
            for query in queries:
                try:
                    result = self._request(
                        "/search/repositories?" + urlencode({"q": query, "per_page": 3})
                    )
                    if result.get("incomplete_results"):
                        errors.append("GitHub reported incomplete search results.")
                    for item in result.get("items", []):
                        repo = github_repository(item.get("html_url", ""))
                        if repo is None:
                            continue
                        repositories.setdefault(
                            repo.casefold(),
                            {
                                "url": "https://github.com/" + repo,
                                "relationship": "unverified",
                                "evidence": None,
                                "license": (item.get("license") or {}).get("spdx_id"),
                            },
                        )
                    if repositories:
                        break
                except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                    errors.append("GitHub search unavailable or rate limited.")
                    break

        candidates = sorted(
            repositories.values(),
            key=lambda item: item["relationship"] != "linked_from_paper",
        )[:3]
        for item in candidates:
            repo = github_repository(item["url"])
            if item["relationship"] == "linked_from_paper":
                try:
                    metadata = self._request(f"/repos/{repo}")
                    item["license"] = (metadata.get("license") or {}).get("spdx_id")
                    item["availability"] = "reachable"
                except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                    item["availability"] = "unavailable"
                    item["relationship"] = "unavailable_link"
                    errors.append(f"Repository unavailable for {repo}.")
                continue
            try:
                readme = self._request(f"/repos/{repo}/readme", raw=True)
                match = re.search(
                    r"(?<![\w.])" + re.escape(paper.paper_id) + r"(?:v\d+)?(?!\d)",
                    readme,
                )
                title_match = re.search(
                    r"\W+".join(
                        re.escape(term) for term in normalized_text(paper.title).split()
                    ),
                    readme,
                    re.I,
                )
                if match:
                    quote = readme[max(0, match.start() - 100) : match.end() + 150]
                elif title_match:
                    quote = readme[
                        max(0, title_match.start() - 100) : title_match.end() + 150
                    ]
                    item["relationship"] = "title_match_unverified"
                else:
                    continue
                if match:
                    item["relationship"] = "mentions_paper"
                item["evidence"] = {"source": item["url"] + "#readme", "quote": quote}
            except (HTTPError, URLError, TimeoutError, OSError, ValueError):
                errors.append(f"README unavailable for {repo}.")
        associated = any(
            item["relationship"] in {"linked_from_paper", "mentions_paper"}
            for item in candidates
        )
        return {
            "status": (
                "found"
                if associated
                else (
                    "unavailable"
                    if errors
                    else "unverified" if candidates else "not_found"
                )
            ),
            "paper_id": paper.paper_id,
            "repositories": candidates,
            "errors": errors,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "note": "Paper/README association only. Author ownership, implementation completeness, runnable code, and reproducibility were not verified. Repositories were not cloned or executed.",
        }
