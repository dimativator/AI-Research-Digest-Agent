import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from test_digest_pipeline import FakeArxiv, FakeLLM, config, finish, profile
from research_digest.agent.digest_agent import DigestAgent
from research_digest.storage.paper_store import PaperStore


class SelectionCriticTests(unittest.TestCase):
    def run_agent(self, policy, **settings):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        client = FakeLLM(policy)
        x_client = Mock()
        agent = DigestAgent(
            PaperStore(Path(directory.name) / "test.sqlite3"),
            client=client,
            arxiv=FakeArxiv(),
            x_client=x_client,
        )
        run_id = agent.start(profile(), config(variant="selection-critic", **settings))
        result = agent.run(run_id)
        x_client.collect.assert_not_called()
        return agent, client, result

    def test_accept_keeps_draft_without_revision_or_tools(self):
        def policy(context):
            if "draft" in context:
                return {"verdict": "accept", "reason": "Fits the profile", "issues": []}
            return finish(context)

        agent, client, result = self.run_agent(policy)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            [purpose for purpose, _ in client.calls],
            ["screen_papers", "selection_draft", "selection_critique"],
        )
        self.assertNotIn("revision", agent.state["selection_critic"])
        self.assertEqual(result["stats"]["tool_calls"], 0)
        count = len(client.calls)
        agent.run(agent.run_id)
        self.assertEqual(len(client.calls), count)

    def test_revise_runs_once_and_can_remove_selection(self):
        def policy(context):
            if "critique" in context:
                decision = finish(context)
                decision["selections"] = []
                return decision
            if "draft" in context:
                return {
                    "verdict": "revise",
                    "reason": "Weak fit",
                    "issues": [
                        {
                            "paper_id": context["candidates"][0]["paper_id"],
                            "reason": "Reconsider profile fit",
                        }
                    ],
                }
            return finish(context)

        agent, client, result = self.run_agent(policy, digest_size=-1)
        self.assertEqual(result["papers"], [])
        self.assertEqual(client.calls[-1][0], "selection_revision")
        self.assertEqual(result["stats"]["steps"], 3)
        self.assertEqual(
            len(
                [
                    e
                    for e in agent.store.events(agent.run_id)
                    if e["type"] == "selection_stage"
                ]
            ),
            3,
        )

    def test_invalid_critic_id_gets_feedback(self):
        attempts = 0

        def policy(context):
            nonlocal attempts
            if "draft" not in context:
                return finish(context)
            attempts += 1
            if attempts == 1:
                return {
                    "verdict": "revise",
                    "reason": "Check candidate",
                    "issues": [{"paper_id": "unknown", "reason": "Unsupported"}],
                }
            self.assertIn("validation_error", context)
            return {"verdict": "accept", "reason": "Fits profile", "issues": []}

        _, _, result = self.run_agent(policy)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(attempts, 2)
