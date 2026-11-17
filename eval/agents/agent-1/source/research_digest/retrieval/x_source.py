"""Collect a bounded sample from X's rendered Latest search in the user's Arc.

No private API endpoints, credentials in prompts, or anti-bot bypasses. Search
failures are unavailable observations, never evidence of zero attention.
"""

from __future__ import annotations

import re
import json
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlencode

from .arxiv_source import Paper
from .x_attention import aggregate_attention, parse_count
from .arc_browser import ArcTab


COLLECTOR_VERSION = "arc-v2"
POST_SELECTOR = 'article[data-testid="tweet"]'
ACCOUNT_SELECTOR = '[data-testid="SideNav_AccountSwitcher_Button"]'

# Read only rendered post elements. All extracted strings remain untrusted data.
POST_SCRIPT = """articles => articles.map(article => {
    const time = article.querySelector('time');
    const link = time?.closest('a');
    const count = name => {
        const button = article.querySelector(`[data-testid="${name}"]`);
        if (!button) return null;
        const text = button.innerText.trim();
        if (text) return text;
        const label = button.getAttribute('aria-label') || '';
        const match = label.match(/^([\\d,.]+\\s*[KMB]?)\\s/i);
        return match ? match[1] : null;
    };
    return {
        url: link?.href || '', created_at: time?.getAttribute('datetime') || '',
        text: article.querySelector('[data-testid="tweetText"]')?.innerText || '',
        links: Array.from(article.querySelectorAll('a')).flatMap(a =>
            [a.href, a.getAttribute('title') || '', a.innerText || '']),
        counts: {likes: count('like') ?? count('unlike'),
                 reposts: count('retweet') ?? count('unretweet'),
                 replies: count('reply'), quotes: null}
    };
})"""


def search_queries(paper: Paper, date_from: str, date_to: str) -> list[str]:
    until = (date.fromisoformat(date_to) + timedelta(days=1)).isoformat()
    period = f"since:{date_from} until:{until} -filter:retweets"
    title = paper.title.replace('"', " ").replace("\n", " ")
    return [
        f'(url:arxiv.org/abs/{paper.paper_id} OR url:arxiv.org/pdf/{paper.paper_id} OR url:arxiv.org/html/{paper.paper_id} OR "{paper.paper_id}") {period}',
        f'"{title}" {period}',
    ]


def parse_rendered_post(raw: dict[str, Any]) -> dict[str, Any] | None:
    match = re.fullmatch(
        r"https://(?:www\.)?(?:x|twitter)\.com/([A-Za-z0-9_]+)/status/(\d+)(?:\?.*)?",
        raw["url"],
    )
    if not match or not raw["created_at"]:
        return None
    author, post_id = match.groups()
    return {
        "id": post_id,
        "author": author,
        "url": f"https://x.com/{author}/status/{post_id}",
        "text": raw["text"],
        "created_at": raw["created_at"],
        "links": raw["links"],
        "metrics": {
            name: parse_count(raw["counts"].get(name))
            for name in ("likes", "reposts", "replies", "quotes")
        },
    }


class XAccessError(RuntimeError):
    """Confirmed X access failure, rather than a transient readiness timeout."""


class XBrowserClient:
    def __init__(
        self,
        *,
        max_posts: int = 50,
        max_scrolls: int = 5,
    ) -> None:
        self.max_posts = max_posts
        self.max_scrolls = max_scrolls
        self._page = None
        self._failure: str | None = None

    def _open(self) -> None:
        if self._failure:
            raise RuntimeError(self._failure)
        if self._page is not None:
            return
        self._page = ArcTab.open("https://x.com/home")

    def collect(self, paper: Paper, date_from: str, date_to: str) -> dict[str, Any]:
        posts = []
        errors = []
        queries = search_queries(paper, date_from, date_to)
        try:
            self._open()
        except (ImportError, RuntimeError) as error:
            self._failure = str(error)
            return aggregate_attention(
                paper, [], date_from, date_to, errors=[str(error)]
            )
        except Exception:
            self._failure = (
                "Arc could not open the X scraper tab. Run python setup_x.py."
            )
            self.close()
            return aggregate_attention(
                paper, [], date_from, date_to, errors=[self._failure]
            )

        for query in queries:
            try:
                posts.extend(self._search(query))
            except Exception as error:
                message = (
                    str(error)
                    if isinstance(error, RuntimeError)
                    else "X search failed or was blocked; results are incomplete."
                )
                errors.append(message)
                # Only confirmed access failures disable the rest of the session.
                if isinstance(error, XAccessError):
                    self._failure = message
                break
        result = aggregate_attention(paper, posts, date_from, date_to, errors=errors)
        result["queries"] = queries
        return result

    def _search(self, query: str) -> list[dict[str, Any]]:
        page = self._page
        page.goto(
            "https://x.com/search?"
            + urlencode({"q": query, "src": "typed_query", "f": "live"}),
            wait_until="domcontentloaded",
            timeout=30_000,
        )
        page.wait_for_function(
            """() => document.querySelector('article[data-testid="tweet"]') ||
            /No results for/i.test(document.body.innerText) ||
            /Something went wrong|Rate limit exceeded|Verify you are human/i.test(document.body.innerText) ||
            location.pathname.includes('/login')""",
            timeout=20_000,
        )
        page.wait_for_function(
            f"() => !!document.querySelector({json.dumps(ACCOUNT_SELECTOR)})",
            timeout=10_000,
        )
        body = page.evaluate("() => document.body.innerText")
        if re.search(
            r"Something went wrong|Rate limit exceeded|Verify you are human", body, re.I
        ):
            raise XAccessError(
                "X search blocked: an error, rate limit, or challenge was displayed."
            )
        if re.search(r"No results for", body, re.I) and not page.evaluate(
            f"() => document.querySelectorAll({json.dumps(POST_SELECTOR)}).length"
        ):
            return []

        found = {}
        for _ in range(self.max_scrolls):
            raw_posts = page.evaluate(
                f"() => ({POST_SCRIPT})(Array.from(document.querySelectorAll({json.dumps(POST_SELECTOR)})))"
            )
            for raw in raw_posts:
                post = parse_rendered_post(raw)
                if post is not None:
                    found[post["id"]] = post
            if len(found) >= self.max_posts:
                break
            page.evaluate("() => window.scrollBy(0, 1800)")
            page.wait_for_timeout(1200)
        return list(found.values())[: self.max_posts]

    def close(self) -> None:
        if self._page is not None:
            self._page.close()
            self._page = None


def login() -> None:
    """Check the active Arc profile's X login in a new, owned tab."""
    page = ArcTab.open("https://x.com/home")
    try:
        print(
            "Using your Arc profile. Sign in to X in the new tab if needed.", flush=True
        )
        page.wait_for_function(
            f"() => !!document.querySelector({json.dumps(ACCOUNT_SELECTOR)})",
            timeout=600_000,
        )
        print("Arc's X session is ready. You can now run your digest normally.")
    finally:
        page.close()
