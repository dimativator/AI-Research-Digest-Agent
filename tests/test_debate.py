import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from research_digest.agent.debate import DebateAgent
from research_digest.agent.digest_agent import DigestAgent
from research_digest.storage.paper_store import PaperStore
from test_digest_pipeline import FakeArxiv, FakeLLM, config, decision, finish, profile


def review(context):
    return {
        "assessments": [
            {
                "paper_id": candidate["paper_id"],
                "recommendation": (
                    "include" if context["role"] == "precision" else "uncertain"
                ),
                "reason": "Matches the profile's evaluation interest",
                "uncertainty": "",
            }
            for candidate in context["candidates"]
        ]
    }


def editor_finish(context):
    return {**finish(context), "reviewer": ""}


class DebateTests(unittest.TestCase):
    def agent(self, policy):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        agent = DigestAgent(
            PaperStore(Path(directory.name) / "test.sqlite3"),
            client=FakeLLM(policy),
            arxiv=FakeArxiv(),
            x_client=Mock(),
        )
        agent.start(
            profile(),
            config(variant="debate", x_enabled=False, max_steps=-1, max_llm_calls=-1),
        )
        agent._restore_run(agent.run_id)
        with patch(
            "research_digest.retrieval.e5_reranker.rerank_bm25_head",
            side_effect=lambda ranked, text: (
                ranked,
                {"pool_size": len(ranked), "latency_seconds": 0},
            ),
        ):
            agent._prepare_candidates()
        return agent

    def test_independent_reviews_disagreement_and_finish(self):
        inputs = []

        def policy(context):
            if "role" in context:
                inputs.append(context)
                return review(context)
            self.assertEqual(len(context["debate"]["disagreements"]), 1)
            return editor_finish(context)

        agent = self.agent(policy)
        result = agent.run(agent.run_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["papers"]), 1)
        self.assertEqual(inputs[0]["candidates"], inputs[1]["candidates"])
        self.assertEqual(
            set(inputs[1]),
            {"role", "responsibility", "profile", "candidates", "bounds"},
        )
        self.assertNotIn("selection_check", agent.state)
        calls = len(agent.client.calls)
        agent.run(agent.run_id)
        self.assertEqual(len(agent.client.calls), calls)

    def test_editor_tool_result_then_finish(self):
        def policy(context):
            if "role" in context:
                return review(context)
            if not context["last_observation"]:
                return {
                    **decision("inspect_paper", context["candidates"][0]["paper_id"]),
                    "reviewer": "",
                }
            self.assertEqual(context["last_observation"]["action"], "inspect_paper")
            return editor_finish(context)

        agent = self.agent(policy)
        result = agent.run(agent.run_id)
        self.assertEqual(result["stats"]["tool_calls"], 1)
        self.assertEqual(len(agent.arxiv.inspect_calls), 1)

    def test_reconsideration_once_and_resume_saved_reviews(self):
        def policy(context):
            if "role" in context:
                if "question" in context:
                    self.assertIn("prior_assessments", context)
                return review(context)
            if not context["debate"]["reconsiderations"]:
                return {
                    **decision(
                        "ask_reviewer",
                        context["candidates"][0]["paper_id"],
                        question="Is this a core profile match?",
                    ),
                    "reviewer": "precision",
                    "question": "Is this a core profile match?",
                }
            return editor_finish(context)

        agent = self.agent(policy)
        controller = DebateAgent(agent)
        controller._initial_reviews()
        calls = len(agent.client.calls)
        agent._restore_run(agent.run_id)
        DebateAgent(agent)._initial_reviews()
        self.assertEqual(len(agent.client.calls), calls)
        result = agent.run(agent.run_id)
        self.assertEqual(result["status"], "completed")
        exchange = agent.state["debate"]["reconsiderations"]["precision"]
        self.assertEqual(exchange["status"], "completed")
        with self.assertRaisesRegex(ValueError, "already used"):
            DebateAgent(agent)._reconsider(exchange["request"])

    def test_invalid_finish_count_returns_feedback(self):
        attempts = 0

        def policy(context):
            nonlocal attempts
            if "role" in context:
                return review(context)
            attempts += 1
            response = editor_finish(context)
            if attempts == 1:
                response["selections"] = []
            else:
                self.assertIn("exactly", context["last_error"])
            return response

        agent = self.agent(policy)
        self.assertEqual(len(agent.run(agent.run_id)["papers"]), 1)
        self.assertEqual(attempts, 2)

    def test_interrupted_reconsideration_is_not_repeated(self):
        def policy(context):
            if "role" in context:
                if "question" in context:
                    raise RuntimeError("interrupted inference")
                return review(context)
            return editor_finish(context)

        agent = self.agent(policy)
        controller = DebateAgent(agent)
        controller._initial_reviews()
        action = {
            **decision(
                "ask_reviewer",
                agent.state["admitted_ids"][0],
                question="Clarify relevance",
            ),
            "reviewer": "coverage",
        }
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            controller._reconsider(action)
        agent._restore_run(agent.run_id)
        result = agent.run(agent.run_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            agent.state["debate"]["reconsiderations"]["coverage"]["status"],
            "interrupted",
        )
