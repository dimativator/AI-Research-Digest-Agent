"""Sequential, rate-limited arXiv metadata and HTML inspection (standard library)."""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from research_digest.storage.paper_store import PaperStore


MAX_WINDOW_RESULTS = 10_000
REQUEST_INTERVAL_SECONDS = 3.0
NAMESPACES = {
    "a": "http://www.w3.org/2005/Atom",
    "o": "http://a9.com/-/spec/opensearch/1.1/",
}


class RetrievalError(RuntimeError):
    pass


def arxiv_id(value: str) -> tuple[str, int]:
    value = re.sub(r"^https?://(?:export\.)?arxiv\.org/(?:abs|pdf|html)/", "", value)
    value = value.removesuffix(".pdf")
    match = re.fullmatch(
        r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v([1-9]\d*))?", value
    )

    if not match:
        raise ValueError(f"Invalid arXiv identifier: {value!r}")

    return match[1], int(match[2] or 1)


def categories(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(sorted(set(values)))

    if not result or any(
        not re.fullmatch(r"[a-z][a-z0-9-]*(?:\.[A-Za-z][A-Za-z0-9-]*)?", x)
        for x in result
    ):
        raise ValueError("Supply arXiv categories such as cs.CL or cs.AI")

    return result


@dataclass(frozen=True)
class Paper:
    paper_id: str
    version: int
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.paper_id}v{self.version}"


@dataclass(frozen=True)
class Page:
    total: int
    start: int
    papers: list[Paper]


def parse_feed(content: str) -> Page:
    """Parse feed metadata and paper entries, distinguishing API errors from no results."""

    try:
        root = ET.fromstring(content)
        entries = root.findall("a:entry", NAMESPACES)

        for entry in entries:
            if "/api/errors" in entry.findtext("a:id", "", NAMESPACES):
                raise RetrievalError(
                    entry.findtext("a:summary", "arXiv API error", NAMESPACES)
                )

        total = int(root.findtext("o:totalResults", namespaces=NAMESPACES))
        start = int(root.findtext("o:startIndex", namespaces=NAMESPACES))
        papers = [_parse_entry(entry) for entry in entries]

        if total < 0 or start < 0:
            raise ValueError("Negative pagination metadata")

        return Page(total=total, start=start, papers=papers)
    except (ET.ParseError, ValueError, TypeError, KeyError) as error:
        raise RetrievalError(f"Malformed arXiv feed: {error}") from error


def _parse_entry(entry: ET.Element) -> Paper:
    """Normalize one Atom entry while preserving its submission/version metadata."""
    paper_id, version = arxiv_id(entry.findtext("a:id", "", NAMESPACES))
    title = " ".join(entry.findtext("a:title", "", NAMESPACES).split())
    abstract = " ".join(entry.findtext("a:summary", "", NAMESPACES).split())
    published = entry.findtext("a:published", "", NAMESPACES)
    updated = entry.findtext("a:updated", "", NAMESPACES)

    if not title or not abstract:
        raise ValueError("Missing title or abstract")

    for timestamp in (published, updated):
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        if parsed.utcoffset() != timedelta(0):
            raise ValueError("Expected UTC timestamps")

    return Paper(
        paper_id=paper_id,
        version=version,
        title=title,
        abstract=abstract,
        authors=[
            author.findtext("a:name", "", NAMESPACES)
            for author in entry.findall("a:author", NAMESPACES)
        ],
        categories=[
            category.attrib["term"]
            for category in entry.findall("a:category", NAMESPACES)
        ],
        published=published,
        updated=updated,
    )


class _ArticleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.article_depth = 0
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "article":
            self.article_depth += 1

        if self.article_depth and tag in {"script", "style", "nav", "aside", "footer"}:
            self.skip += 1

    def handle_endtag(self, tag: str) -> None:
        if self.article_depth and tag in {"script", "style", "nav", "aside", "footer"}:
            self.skip = max(0, self.skip - 1)

        if tag == "article":
            self.article_depth = max(0, self.article_depth - 1)

    def handle_data(self, data: str) -> None:
        if self.article_depth and not self.skip:
            self.parts.append(data)


class ArxivClient:
    def __init__(self, *, timeout: float = 30, retries: int = 2) -> None:
        self.timeout = timeout
        self.retries = retries
        self._last_request = 0.0

    def _get(self, url: str, *, max_bytes: int = 4_000_000) -> str:
        for attempt in range(self.retries + 1):
            time.sleep(
                max(
                    0,
                    REQUEST_INTERVAL_SECONDS - (time.monotonic() - self._last_request),
                )
            )
            self._last_request = time.monotonic()

            try:
                request = Request(
                    url,
                    headers={
                        "User-Agent": "AI-Research-Digest-Agent/0.1 (research prototype)"
                    },
                )

                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read(max_bytes + 1)

                    if len(body) > max_bytes:
                        raise RetrievalError(
                            "arXiv response exceeds the download size limit"
                        )

                    return body.decode("utf-8")
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504}:
                    raise RetrievalError(f"arXiv HTTP {exc.code}") from exc

                if attempt == self.retries:
                    raise RetrievalError(
                        f"arXiv HTTP {exc.code}; retries exhausted"
                    ) from exc

                try:
                    retry_after = min(
                        60, max(3, int(exc.headers.get("Retry-After", "3")))
                    )
                except (TypeError, ValueError):
                    retry_after = 3

                time.sleep(max(retry_after, 3 * 2**attempt))
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == self.retries:
                    raise RetrievalError(f"arXiv request failed: {exc}") from exc

                time.sleep(3 * 2**attempt)

        raise RetrievalError("arXiv request failed")

    def page(self, query: str, start: int, size: int) -> Page:
        params = urlencode(
            {
                "search_query": query,
                "start": start,
                "max_results": size,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
        )
        return parse_feed(self._get("https://export.arxiv.org/api/query?" + params))

    def inspect(self, paper: Paper, *, max_chars: int = 12_000) -> dict[str, Any]:
        url = f"https://arxiv.org/html/{paper.paper_id}v{paper.version}"

        try:
            html = self._get(url)
            parser = _ArticleText()
            parser.feed(html)
            text = " ".join(" ".join(parser.parts).split())

            if len(text) < 100:
                raise RetrievalError("No readable article body in the HTML response")

            return {
                "status": "ok",
                "source": url,
                "text": text[:max_chars],
                "truncated": len(text) > max_chars,
                "available_chars": len(text),
                "note": "HTML text excerpt; equations and formatting may be imperfect.",
            }
        except (RetrievalError, UnicodeError, ValueError) as exc:
            return {
                "status": "unavailable",
                "source": url,
                "text": "",
                "reason": str(exc),
                "note": "Only abstract evidence is available. No PDF extraction was attempted.",
            }

    def document(self, paper: Paper) -> dict[str, Any]:
        """Fetch the exact version as structured sections and preserved hyperlinks."""
        from .paper_content import extract_document

        source = f"https://arxiv.org/html/{paper.paper_id}v{paper.version}"
        try:
            return extract_document(self._get(source), source)
        except (RetrievalError, UnicodeError, ValueError) as error:
            return {
                "status": "unavailable",
                "source": source,
                "reason": str(error),
                "note": "HTML article unavailable. No PDF extraction was attempted.",
            }


def collect_metadata(
    client: ArxivClient,
    store: PaperStore,
    selected_categories: tuple[str, ...],
    date_from: str,
    date_to: str,
    *,
    page_size: int = 100,
    refresh: bool = False,
) -> list[Paper]:
    """Fetch complete UTC date windows. Cache each page for restart after failure.

    Completeness is relative to the API's current index, not unannounced papers.
    Large windows are split recursively before crossing the API result limit.
    """
    normalized_categories = categories(selected_categories)
    start_date, end_date = date.fromisoformat(date_from), date.fromisoformat(date_to)

    if start_date > end_date or not 1 <= page_size <= 1000:
        raise ValueError("Invalid date range or page size (1..1000)")

    range_start = datetime.combine(start_date, datetime.min.time(), timezone.utc)
    range_end = datetime.combine(
        end_date, datetime.min.time(), timezone.utc
    ) + timedelta(hours=23, minutes=59)
    papers_by_id: dict[str, Paper] = {}

    def collect_window(start_time: datetime, end_time: datetime) -> None:
        query = _build_query(normalized_categories, start_time, end_time)
        offset = 0
        expected_total = None
        seen_paper_ids: set[str] = set()

        while True:
            page = None if refresh else store.cached_page(query, offset, page_size)

            if page is None:
                page = client.page(query, offset, page_size)

            if page.start != offset:
                raise RetrievalError("arXiv returned an unexpected page offset")

            if page.total > MAX_WINDOW_RESULTS:
                minutes = int((end_time - start_time).total_seconds() // 60)

                if minutes == 0:
                    raise RetrievalError(
                        "Too many results in one minute; bulk harvesting is required"
                    )

                middle = start_time + timedelta(minutes=minutes // 2)
                before = set(papers_by_id)
                collect_window(start_time, middle)
                collect_window(middle + timedelta(minutes=1), end_time)

                if len(set(papers_by_id) - before) != page.total:
                    raise RetrievalError(
                        "Split-window counts do not match the parent query; refresh required"
                    )

                return

            if expected_total is None:
                expected_total = page.total

            if page.total != expected_total:
                raise RetrievalError(
                    "arXiv result count changed during pagination; retry with refresh"
                )

            if (
                len(page.papers) > page_size
                or offset + len(page.papers) > expected_total
            ):
                raise RetrievalError("arXiv page exceeds its expected result count")

            for paper in page.papers:
                submitted = datetime.fromisoformat(
                    paper.published.replace("Z", "+00:00")
                )

                if not start_time <= submitted < end_time + timedelta(minutes=1):
                    raise RetrievalError(
                        "arXiv returned a paper outside the requested submission window"
                    )

                if not set(normalized_categories).intersection(paper.categories):
                    raise RetrievalError(
                        "arXiv returned a paper outside the requested categories"
                    )

                seen_paper_ids.add(paper.paper_id)
                previous = papers_by_id.get(paper.paper_id)

                if previous is None or (paper.version, paper.updated) > (
                    previous.version,
                    previous.updated,
                ):
                    papers_by_id[paper.paper_id] = paper

            store.save_page(query, offset, page_size, page)
            offset += len(page.papers)

            if offset == expected_total:
                if len(seen_paper_ids) != expected_total:
                    raise RetrievalError(
                        "Duplicate or missing metadata across arXiv pages; refresh required"
                    )

                return

            if not page.papers:
                raise RetrievalError(
                    "arXiv returned an empty page before collection was complete"
                )

    window_start = range_start

    while window_start <= range_end:
        end = min(window_start + timedelta(hours=23, minutes=59), range_end)
        collect_window(window_start, end)
        window_start = end + timedelta(minutes=1)

    return sorted(papers_by_id.values(), key=lambda p: p.paper_id)


def _build_query(
    selected_categories: tuple[str, ...], start: datetime, end: datetime
) -> str:
    category_query = " OR ".join(f"cat:{category}" for category in selected_categories)
    first_minute = start.strftime("%Y%m%d%H%M")
    last_minute = end.strftime("%Y%m%d%H%M")
    return f"({category_query}) AND submittedDate:[{first_minute} TO {last_minute}]"
