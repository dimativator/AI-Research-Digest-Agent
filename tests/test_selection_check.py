import json
import unittest
from unittest.mock import patch

import test_exploration
from test_digest_pipeline import FakeLLM, finish
from research_digest.agent.selection_check import SelectionCheck


def review(verdict="accept", search=None):
    return {
        "verdict": verdict,
        "weak_selection_ids": [],
        "overlooked_candidate_ids": [],
        "missing_topics": [],
        "reason": "Check profile fit",
        "search": search,
    }


class SelectionCheckTests(unittest.TestCase):
    def agent(self, **settings):
        # Reuse the isolated fake corpus without downloading E5 or calling a model.
        helper = test_exploration.ExplorationTests()
        self.addCleanup(helper.doCleanups)
        return helper.agent(**settings)

    def test_accept_one_call_and_no_inspection_payload(self):
        agent = self.agent()

        def policy(context):
            if "proposed" in context:
                self.assertNotIn("observations", context)
                self.assertNotIn("evidence", context["proposed"][0])
                self.assertLessEqual(len(context["candidates"]), 200)
                return review()
            return finish(context)

        agent.client = FakeLLM(policy)
        result = agent.run(agent.run_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([p for p, _ in agent.client.calls].count("selection_check"), 1)
        calls = len(agent.client.calls)
        agent.run(agent.run_id)
        self.assertEqual(len(agent.client.calls), calls)
        self.assertTrue(agent.state["selection_check"]["applied"])

    def test_revise_feedback_then_next_finish_accepted(self):
        agent = self.agent()
        seen = []

        def policy(context):
            if "proposed" in context:
                return review("revise")
            seen.append(context.get("selection_feedback"))
            return finish(context)

        agent.client = FakeLLM(policy)
        self.assertEqual(agent.run(agent.run_id)["status"], "completed")
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1]["review"]["verdict"], "revise")
        self.assertEqual([p for p, _ in agent.client.calls].count("selection_check"), 1)

    def test_search_is_executed_and_logged(self):
        for tool, args in (
            ("expand_shortlist", {}),
            ("search_candidates", {"query": "LLM benchmarks"}),
            ("search_similar", {"paper_id": "2610.00001"}),
        ):
            with self.subTest(tool=tool):
                agent = self.agent()
                agent.client = FakeLLM(
                    lambda c: (
                        review(
                            "search",
                            {"tool": tool, "query": None, "paper_id": None, **args},
                        )
                        if "proposed" in c
                        else finish(c)
                    )
                )
                result = agent.run(agent.run_id)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["stats"]["tool_calls"], 1)
                self.assertGreater(result["stats"]["screened"], 1)
                self.assertTrue(
                    any(
                        e.get("source") == "selection_check"
                        for e in agent.store.events(agent.run_id)
                    )
                )

    def test_invalid_ids_or_exhausted_budget_never_execute_search(self):
        for bad, settings in (
            ({**review(), "weak_selection_ids": ["unknown"]}, {}),
            (
                review(
                    "search",
                    {"tool": "expand_shortlist", "query": None, "paper_id": None},
                ),
                {"max_expansions": 0},
            ),
        ):
            agent = self.agent(**settings)
            agent.client = FakeLLM(lambda c: bad if "proposed" in c else finish(c))
            result = agent.run(agent.run_id)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["stats"]["tool_calls"], 0)
            self.assertIsNotNone(agent.state["selection_check"]["validation_error"])
            self.assertEqual(
                [p for p, _ in agent.client.calls].count("selection_check"), 1
            )

    def test_resume_after_saved_review_does_not_repeat_model_call(self):
        agent = self.agent()
        agent.client = FakeLLM(
            lambda c: (
                review(
                    "search",
                    {"tool": "expand_shortlist", "query": None, "paper_id": None},
                )
                if "proposed" in c
                else finish(c)
            )
        )
        original = agent._persist_run

        def interrupt(event=None, **kwargs):
            original(event, **kwargs)
            if event and event["type"] == "selection_check_result":
                raise KeyboardInterrupt()

        with patch.object(agent, "_persist_run", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                agent.run(agent.run_id)
        result = agent.run(agent.run_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([p for p, _ in agent.client.calls].count("selection_check"), 1)
        self.assertEqual(result["stats"]["tool_calls"], 1)

    def test_resume_after_dispatch_keeps_pending_retrieval(self):
        agent = self.agent()
        agent.client = FakeLLM(
            lambda c: (
                review(
                    "search",
                    {"tool": "expand_shortlist", "query": None, "paper_id": None},
                )
                if "proposed" in c
                else finish(c)
            )
        )
        original = agent._persist_run

        def interrupt(event=None, **kwargs):
            original(event, **kwargs)
            if event and event["type"] == "selection_check_applied":
                raise KeyboardInterrupt()

        with patch.object(agent, "_persist_run", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                agent.run(agent.run_id)
        result = agent.run(agent.run_id)
        self.assertEqual(result["stats"]["tool_calls"], 1)
        self.assertEqual([p for p, _ in agent.client.calls].count("selection_check"), 1)
