"""Local E5 cosine reranking of the BM25 head; preserve the searchable tail.

Model usage: https://huggingface.co/intfloat/e5-small-v2 (MIT).
"""

from time import perf_counter
from typing import Any

from .ranking import RankedPaper

MODEL_ID = "intfloat/e5-small-v2"
MODEL_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
RERANK_POOL_SIZE = 200
MAX_TOKENS = 512


class E5Reranker:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "E5 reranking requires torch and transformers in the project environment."
            ) from error
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        self.model = AutoModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=True
        )
        self.model.eval()

    def score(
        self, profile_text: str, candidates: list[RankedPaper]
    ) -> tuple[list[float], dict[str, Any]]:
        import torch
        import torch.nn.functional as functional

        self._load()
        texts = ["query: " + profile_text] + [
            "passage: " + item.paper.title + "\n" + item.paper.abstract
            for item in candidates
        ]
        lengths = [
            len(ids) for ids in self.tokenizer(texts, truncation=False)["input_ids"]
        ]
        embeddings = []
        with torch.inference_mode():
            for offset in range(0, len(texts), 16):
                batch = self.tokenizer(
                    texts[offset : offset + 16],
                    padding=True,
                    truncation=True,
                    max_length=MAX_TOKENS,
                    return_tensors="pt",
                )
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"]
                pooled = (
                    hidden.masked_fill(~mask[..., None].bool(), 0.0).sum(dim=1)
                    / mask.sum(dim=1)[..., None]
                )
                embeddings.append(functional.normalize(pooled, p=2, dim=1))
        vectors = torch.cat(embeddings)
        scores = (vectors[1:] @ vectors[0]).tolist()
        return scores, {
            "profile_tokens_before_truncation": lengths[0],
            "profile_truncated": lengths[0] > MAX_TOKENS,
            "truncated_paper_ids": [
                item.paper.paper_id
                for item, length in zip(candidates, lengths[1:])
                if length > MAX_TOKENS
            ],
        }


def rerank_bm25_head(
    ranked: list[RankedPaper], profile_text: str, reranker: E5Reranker | None = None
) -> tuple[list[RankedPaper], dict[str, Any]]:
    """Keep BM25 scores intact; record E5 scores separately from ordering."""
    started = perf_counter()
    head = ranked[:RERANK_POOL_SIZE]
    scores, truncation = (
        (reranker or E5Reranker()).score(profile_text, head) if head else ([], {})
    )
    if len(scores) != len(head):
        raise ValueError("E5 returned the wrong number of scores")
    order = sorted(range(len(head)), key=lambda index: (-scores[index], index))
    details = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": "cpu",
        "pool_size": len(head),
        "max_tokens": MAX_TOKENS,
        "profile_source": "positive topic names and keywords (profile.ranking_text)",
        "latency_seconds": perf_counter() - started,
        **truncation,
        "ranking": [
            {
                "paper_id": head[index].paper.paper_id,
                "bm25_rank": index + 1,
                "bm25_score": head[index].score,
                "e5_rank": rank,
                "e5_score": scores[index],
            }
            for rank, index in enumerate(order, 1)
        ],
    }
    return [head[index] for index in order] + ranked[RERANK_POOL_SIZE:], details
