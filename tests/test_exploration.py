import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from research_digest.agent.digest_agent import DigestAgent
from research_digest.agent.protocol import validate_decision
from research_digest.storage.paper_store import PaperStore
from test_digest_pipeline import FakeArxiv, FakeLLM, config, decision, finish, profile


class ExplorationTests(unittest.TestCase):
    def agent(self, **settings):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        agent = DigestAgent(
            PaperStore(Path(directory.name) / "run.sqlite3"),
            client=FakeLLM(),
            arxiv=FakeArxiv(),
            x_client=Mock(),
        )
        run_id = agent.start(
            profile(), config(variant="react-explore", x_enabled=False, **settings)
        )
        agent._restore_run(run_id)
        with patch(
            "research_digest.retrieval.e5_reranker.rerank_bm25_head",
            side_effect=lambda ranked, text: (
                ranked,
                {"pool_size": len(ranked), "latency_seconds": 0},
            ),
        ):
            agent._prepare_candidates()
        return agent

    def test_tools_are_variant_specific(self):
        agent = self.agent()
        names = {t["name"] for t in agent._build_decision_context()["tools"]}
        self.assertTrue({"get_candidate_overview", "search_similar"} <= names)
        for action in ("get_candidate_overview", "search_similar"):
            value = decision(
                action,
                agent.state["admitted_ids"][0] if action == "search_similar" else "",
            )
            validate_decision(value, agent.state["screened"], exploration=True)
            with self.assertRaisesRegex(ValueError, "Unknown"):
                validate_decision(value, agent.state["screened"])
        with self.assertRaisesRegex(ValueError, "screened"):
            validate_decision(
                decision("search_similar", "unknown"),
                agent.state["screened"],
                exploration=True,
            )

    def test_overview_is_current_and_does_not_spend_budget(self):
        agent = self.agent()
        before = agent.state["llm_calls"]
        result = agent._execute_tool(decision("get_candidate_overview"))
        self.assertEqual(result["corpus_count"], 3)
        self.assertEqual(result["screened_count"], 1)
        self.assertEqual(result["unseen_count"], 2)
        self.assertEqual(result["screening_counts"], {"relevant": 1})
        self.assertEqual(agent.state["expansions"], 0)
        self.assertEqual(agent.state["llm_calls"], before)

    def test_similar_screens_unseen_caches_and_resumes(self):
        agent = self.agent()
        seed = agent.state["admitted_ids"][0]
        action = decision("search_similar", seed)
        result = agent._execute_tool(action)
        self.assertEqual(len(result["new_ids"]), 1)
        self.assertNotIn(seed, [hit["paper_id"] for hit in result["matches"]])
        self.assertTrue(all(hit["screened"] for hit in result["matches"]))
        agent._persist_run()
        calls = agent.state["llm_calls"]
        agent._restore_run(agent.run_id)
        self.assertEqual(agent._execute_tool(action), result)
        self.assertEqual(agent.state["expansions"], 1)
        self.assertEqual(agent.state["llm_calls"], calls)
        overview = agent._execute_tool(decision("get_candidate_overview"))
        self.assertEqual(overview["screened_count"], 2)
        self.assertEqual(overview["unseen_count"], 1)

    def test_similarity_respects_candidate_cap(self):
        agent = self.agent(max_candidates=1)
        seed = agent.state["admitted_ids"][0]
        result = agent._execute_tool(decision("search_similar", seed))
        self.assertEqual(result["status"], "exhausted")
        self.assertEqual(len(agent.state["screened"]), 1)

    def test_search_budget_is_shared(self):
        agent = self.agent(max_expansions=0)
        seed = agent.state["admitted_ids"][0]
        result = agent._execute_tool(decision("search_similar", seed))
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(len(agent.state["screened"]), 1)

    def test_loop_uses_both_tools_then_finishes(self):
        agent = self.agent()
        actions = iter(["get_candidate_overview", "search_similar", "finish"])

        def policy(context):
            if "proposed" in context:
                return {
                    "verdict": "accept",
                    "weak_selection_ids": [],
                    "overlooked_candidate_ids": [],
                    "missing_topics": [],
                    "reason": "Good selection",
                    "search": None,
                }
            action = next(actions)
            if action == "finish":
                return finish(context)
            return decision(
                action,
                (
                    context["candidates"][0]["paper_id"]
                    if action == "search_similar"
                    else ""
                ),
            )

        agent.client = FakeLLM(policy)
        result = agent.run(agent.run_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stats"]["tool_calls"], 2)
        self.assertEqual(result["stats"]["screened"], 2)
