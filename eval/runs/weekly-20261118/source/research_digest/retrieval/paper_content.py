"""Section-aware HTML extraction and question-driven passage retrieval."""

from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

from .ranking import bm25_scores, tokens


EXTRACTOR_VERSION = "article-v1"
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 200
QUESTION_STOPWORDS = set(
    "a an the is are was were be been being does do did which what how why when whether this that these those of in on at to for and or with by as it its paper from can".split()
)


class ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.article_depth = 0
        self.skip_tags: list[str] = []
        self.heading: str | None = None
        self.heading_parts: list[str] = []
        self.section = "Article"
        self.parts: list[str] = []
        self.blocks: list[tuple[str, str]] = []
        self.links: list[dict[str, str]] = []
        self.anchor: dict[str, Any] | None = None

    def _flush(self) -> None:
        text = " ".join(" ".join(self.parts).split())
        if text:
            self.blocks.append((self.section, text))
        self.parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "article":
            self.article_depth += 1
        if not self.article_depth:
            return
        if tag in {"script", "style", "nav", "aside", "footer"}:
            self.skip_tags.append(tag)
        if self.skip_tags:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
            self.heading = tag
            self.heading_parts = []
        if tag == "a" and dict(attrs).get("href"):
            self.anchor = {"href": dict(attrs)["href"], "parts": []}

    def handle_endtag(self, tag: str) -> None:
        if self.skip_tags:
            if tag == self.skip_tags[-1]:
                self.skip_tags.pop()
            return
        if not self.article_depth:
            return
        if tag == self.heading:
            self.section = (
                " ".join(" ".join(self.heading_parts).split()) or self.section
            )
            self.heading = None
        if tag == "a" and self.anchor is not None:
            self.links.append(
                {
                    "href": self.anchor["href"],
                    "text": " ".join(self.anchor["parts"]),
                    "section": self.section,
                    "context": " ".join(" ".join(self.parts).split())[-600:],
                }
            )
            self.anchor = None
        if tag in {"p", "li", "table", "figcaption", "article"}:
            self._flush()
        if tag == "article":
            self.article_depth = max(0, self.article_depth - 1)

    def handle_data(self, data: str) -> None:
        if not self.article_depth or self.skip_tags:
            return
        if self.heading:
            self.heading_parts.append(data)
        else:
            self.parts.append(data)
        if self.anchor is not None:
            self.anchor["parts"].append(data)


def extract_document(html: str, source: str) -> dict[str, Any]:
    parser = ArticleParser()
    parser.feed(html)
    parser._flush()
    chunks = []
    # Merge neighboring short blocks so a heading, table, or sentence has context.
    merged: list[tuple[str, str]] = []
    for section, text in parser.blocks:
        if (
            merged
            and merged[-1][0] == section
            and len(merged[-1][1]) + len(text) < CHUNK_CHARS
        ):
            merged[-1] = (section, merged[-1][1] + " " + text)
        else:
            merged.append((section, text))
    for section, text in merged:
        offset = 0
        while offset < len(text):
            end = min(len(text), offset + CHUNK_CHARS)
            if end < len(text):
                space = text.rfind(" ", offset + CHUNK_CHARS // 2, end)
                if space > offset:
                    end = space
            chunks.append(
                {
                    "chunk_id": f"chunk-{len(chunks) + 1:04d}",
                    "section": section,
                    "text": text[offset:end],
                    "source": source,
                    "offset": offset,
                }
            )
            if end == len(text):
                break
            offset = max(offset + 1, end - CHUNK_OVERLAP)
    links = [{**link, "url": urljoin(source, link["href"])} for link in parser.links]
    if sum(len(chunk["text"]) for chunk in chunks) < 100:
        raise ValueError("No readable article body in the HTML response")
    return {
        "status": "ok",
        "source": source,
        "extractor_version": EXTRACTOR_VERSION,
        "chunks": chunks,
        "links": links,
        "note": "Extracted HTML text; equations and table formatting may be imperfect. No PDF extraction was attempted.",
    }


def retrieve_passages(
    document: dict[str, Any], question: str, *, max_chars: int = 12_000
) -> dict[str, Any]:
    if document["status"] != "ok":
        return {**document, "question": question, "passages": [], "text": ""}
    chunks = document["chunks"]
    query = " ".join(
        term for term in tokens(question) if term not in QUESTION_STOPWORDS
    )
    scores = (
        bm25_scores([chunk["section"] + " " + chunk["text"] for chunk in chunks], query)
        if query
        else [0.0] * len(chunks)
    )
    ranked = sorted(range(len(chunks)), key=lambda index: (-scores[index], index))
    chosen: list[int] = []
    for index in ranked:
        if scores[index] <= 0 or len(chosen) >= 6:
            break
        # Bibliography matches should not substitute for the paper's own experiments.
        if any(
            term in chunks[index]["section"].casefold()
            for term in ("references", "bibliography")
        ):
            continue
        for neighbor in (index, index - 1, index + 1):
            if (
                0 <= neighbor < len(chunks)
                and neighbor not in chosen
                and chunks[neighbor]["section"] == chunks[index]["section"]
                and len(chosen) < 6
            ):
                chosen.append(neighbor)
    passages = []
    remaining = max_chars
    for index in sorted(chosen):
        chunk = chunks[index]
        text = chunk["text"][:remaining]
        if not text:
            break
        passages.append({**chunk, "text": text, "score": scores[index]})
        remaining -= len(text)
    return {
        "status": "ok" if passages else "no_relevant_passages",
        "question": question,
        "source": document["source"],
        "passages": passages,
        "text": "\n\n".join(passage["text"] for passage in passages),
        "truncated": len(passages) < len(chunks),
        "available_chars": sum(len(chunk["text"]) for chunk in chunks),
        "note": "Retrieved passages, not an answer. Missing matches do not establish that the paper lacks the requested experiment. "
        + document.get("note", ""),
    }
