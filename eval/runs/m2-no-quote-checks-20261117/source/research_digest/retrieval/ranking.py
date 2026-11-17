"""Deterministic BM25 over titles and abstracts. No embedding dependency."""

import math
import re
from collections import Counter
from dataclasses import dataclass

from research_digest.retrieval.arxiv_source import Paper


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


@dataclass(frozen=True)
class RankedPaper:
    paper: Paper
    score: float


def bm25(
    papers: list[Paper], query: str, *, k1: float = 1.5, b: float = 0.75
) -> list[RankedPaper]:
    scores = bm25_scores(
        [p.title + " " + p.abstract for p in papers], query, k1=k1, b=b
    )
    ranked = [RankedPaper(paper, score) for paper, score in zip(papers, scores)]
    return sorted(ranked, key=lambda item: (-item.score, item.paper.paper_id))


def bm25_scores(
    texts: list[str], query: str, *, k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Score arbitrary documents in input order, including article passages."""
    if not texts:
        return []

    terms = set(tokens(query))

    if not terms:
        raise ValueError("The query has no searchable keywords")

    documents = [Counter(tokens(text)) for text in texts]
    lengths = [sum(document.values()) for document in documents]
    average = sum(lengths) / len(lengths) or 1
    frequency = Counter(
        term for document in documents for term in document if term in terms
    )
    ranked = []

    for document, length in zip(documents, lengths):
        score = 0.0

        for term in sorted(terms):
            tf = document[term]

            if tf:
                idf = math.log(
                    1 + (len(texts) - frequency[term] + 0.5) / (frequency[term] + 0.5)
                )
                score += (
                    idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
                )

        ranked.append(score)

    return ranked
