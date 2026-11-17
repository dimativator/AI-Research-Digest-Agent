import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from research_digest.agent.digest_agent import DigestAgent
from research_digest.retrieval.e5_reranker import rerank_bm25_head
from research_digest.retrieval.ranking import RankedPaper
from research_digest.storage.paper_store import PaperStore
from test_digest_pipeline import FakeArxiv, FakeLLM, config, paper, profile


class E5RerankerTests(unittest.TestCase):
    def test_only_top200_are_reordered_and_bm25_scores_preserved(self):
        ranked = [RankedPaper(paper(i), 500 - i) for i in range(205)]
        encoder = Mock()
        encoder.score.return_value = (list(range(200)), {"profile_truncated": False})
        result, details = rerank_bm25_head(ranked, "low rank", encoder)
        self.assertEqual(result[:200], list(reversed(ranked[:200])))
        self.assertEqual(result[200:], ranked[200:])
        self.assertEqual(len(encoder.score.call_args.args[1]), 200)
        self.assertEqual(details["ranking"][0]["bm25_score"], ranked[199].score)
        self.assertEqual(details["ranking"][0]["e5_score"], 199)

    def test_empty_pool_and_stable_ties(self):
        encoder = Mock()
        result, _ = rerank_bm25_head([], "profile", encoder)
        self.assertEqual(result, [])
        encoder.score.assert_not_called()
        ranked = [RankedPaper(paper(i), 3 - i) for i in range(3)]
        encoder.score.return_value = ([0.5] * 3, {})
        self.assertEqual(rerank_bm25_head(ranked, "profile", encoder)[0], ranked)

    def test_agent2_screens_reranked_head_and_resume_keeps_order(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = DigestAgent(
                PaperStore(Path(directory) / "test.sqlite3"),
                client=FakeLLM(),
                arxiv=FakeArxiv(),
                x_client=Mock(),
            )
            run_id = agent.start(
                profile(), config(variant="react-explore", x_enabled=False)
            )
            agent._restore_run(run_id)

            def rerank(ranked, text):
                return list(reversed(ranked)), {
                    "pool_size": len(ranked),
                    "latency_seconds": 0,
                }

            with patch(
                "research_digest.retrieval.e5_reranker.rerank_bm25_head",
                side_effect=rerank,
            ) as mocked:
                agent._prepare_candidates()
                self.assertEqual(agent.state["admitted_ids"], [paper(3).paper_id])
                agent._restore_run(run_id)
                agent._prepare_candidates()
                mocked.assert_called_once()

    def test_agent1_never_loads_e5(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = DigestAgent(
                PaperStore(Path(directory) / "test.sqlite3"),
                client=FakeLLM(),
                arxiv=FakeArxiv(),
                x_client=Mock(),
            )
            run_id = agent.start(profile(), config(x_enabled=False))
            agent._restore_run(run_id)
            with patch(
                "research_digest.retrieval.e5_reranker.rerank_bm25_head"
            ) as mocked:
                agent._prepare_candidates()
                mocked.assert_not_called()
