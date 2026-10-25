import hashlib
import io
import json
import math
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from research_digest.retrieval.arxiv_source import (
    ArxivClient,
    Page,
    Paper,
    RetrievalError,
    arxiv_id,
    collect_metadata,
    parse_feed,
)
from research_digest.llm.codex_llm import CodexError
from research_digest.agent.digest_agent import (
    DigestAgent,
    DigestConfig,
    PIPELINE_VERSION,
)
from research_digest.agent.config import UNLIMITED_LIMIT_FIELDS
from research_digest.cli import run_digest
from research_digest.cli.digest_output import format_digest
from research_digest.storage.paper_store import PaperStore
from research_digest.retrieval.ranking import bm25
from research_digest.retrieval.x_attention import aggregate_attention
from research_digest.profiles.user_profile import ProfileStore, UserProfile


def paper(index=1, abstract=None):
    return Paper(
        f"2610.{index:05d}",
        1,
        f"LLM evaluation study {index}",
        abstract
        or f"We evaluate LLM agents with reproducible benchmarks. Study {index}.",
        ["Example Author"],
        ["cs.CL", "cs.AI"],
        "2026-10-01T12:00:00Z",
        "2026-10-01T12:00:00Z",
    )


def profile():
    return UserProfile.from_dict(
        {
            "user_id": "test-user",
            "summary": "LLM evaluation",
            "topics": [
                {
                    "name": "LLM evaluation",
                    "keywords": ["agent", "benchmarks"],
                    "source_id": "description",
                    "evidence_quote": "LLM evaluation",
                }
            ],
            "goals": ["Find reproducible methods"],
            "exclusions": ["Medical applications"],
            "arxiv_categories": [],
            "open_questions": [],
            "sources": {"description": "I study LLM evaluation."},
            "model": "test",
            "schema_version": 1,
        }
    )


def config(**kwargs):
    return replace(
        DigestConfig(
            ("cs.CL",),
            "2026-10-01",
            "2026-10-01",
            shortlist_size=1,
            batch_size=1,
            expansion_size=1,
            max_candidates=3,
            digest_size=2,
            model="test-model",
        ),
        **kwargs,
    )


class FakeArxiv:
    def __init__(self, papers=None):
        self.papers = [paper(i) for i in range(1, 4)] if papers is None else papers
        self.page_calls = []
        self.inspect_calls = []

    def page(self, query, start, size):
        self.page_calls.append(start)
        return Page(len(self.papers), start, self.papers[start : start + size])

    def document(self, candidate):
        self.inspect_calls.append(candidate.paper_id)
        return {
            "status": "ok",
            "source": candidate.url,
            "text": "The experiment compares three agent benchmarks.",
            "chunks": [
                {
                    "chunk_id": "chunk-0001",
                    "section": "Experiments",
                    "text": "The experiment compares three agent benchmarks.",
                    "source": candidate.url,
                    "offset": 0,
                }
            ],
            "links": [],
            "truncated": False,
            "available_chars": 54,
        }


def decision(action, pid="", selections=None, *, question=None, query="", limit=10):
    return {
        "action": action,
        "paper_id": pid,
        "reason": "Collect enough evidence for the digest.",
        "question": (
            (
                question
                if question is not None
                else "Which experiment compares agent benchmarks?"
            )
            if action == "inspect_paper"
            else ""
        ),
        "query": query,
        "limit": limit,
        "selections": selections or [],
        "limitations": [],
    }


def finish(context):
    candidate = context["candidates"][-1]
    return decision(
        "finish",
        selections=[
            {
                "paper_id": candidate["paper_id"],
                "summary": "The paper evaluates LLM agents.",
                "relevance": "It concerns agent evaluation.",
                "evidence": [
                    {
                        "source": "abstract",
                        "quote": "We evaluate LLM agents",
                    }
                ],
            }
        ],
    )


class FakeLLM:
    def __init__(self, policy=finish):
        self.policy = policy
        self.calls = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        context = json.loads(kwargs["input"])
        purpose = kwargs["text"]["format"]["name"]
        self.calls.append((purpose, context))
        if purpose == "screen_papers":
            output = {
                "assessments": [
                    {
                        "paper_id": item["paper_id"],
                        "label": "relevant",
                        "reason": "Matches agent evaluation.",
                        "evidence_quote": "We evaluate LLM agents",
                    }
                    for item in context["papers"]
                ]
            }
        else:
            output = self.policy(context)
        return SimpleNamespace(output_text=json.dumps(output))


FEED = """<feed xmlns="http://www.w3.org/2005/Atom"
 xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
 <opensearch:totalResults>1</opensearch:totalResults><opensearch:startIndex>0</opensearch:startIndex>
 <entry><id>http://arxiv.org/abs/2610.00001v2</id><title> LLM\n evaluation </title>
 <summary>We evaluate LLM agents.</summary><published>2026-10-01T12:00:00Z</published>
 <updated>2026-10-02T12:00:00Z</updated><author><name>A. Researcher</name></author>
 <category term="cs.CL"/></entry></feed>"""


class ArxivTests(unittest.TestCase):
    def test_identifiers_and_atom_parser(self):
        self.assertEqual(
            arxiv_id("https://arxiv.org/pdf/2301.00001v3.pdf"), ("2301.00001", 3)
        )
        self.assertEqual(arxiv_id("hep-th/9901001v2"), ("hep-th/9901001", 2))
        for value in (
            "../../etc/passwd",
            "https://example.com/abs/2301.00001",
            "2301.00001v0",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                arxiv_id(value)
        page = parse_feed(FEED)
        self.assertEqual(page.total, 1)
        self.assertEqual(page.papers[0].version, 2)
        self.assertEqual(page.papers[0].title, "LLM evaluation")

    def test_bad_feeds_are_errors_not_empty_results(self):
        for data in (
            "bad XML",
            "<feed/>",
            FEED.replace("<summary>We evaluate LLM agents.</summary>", ""),
            FEED.replace("2026-10-01T12:00:00Z", "bad date"),
        ):
            with self.subTest(data=data[:20]), self.assertRaises(RetrievalError):
                parse_feed(data)

    def test_inspection_is_bounded_and_excludes_scripts(self):
        client = ArxivClient()
        html = (
            "<html><nav>Ignore</nav><article><h1>Results</h1><script>evil()</script><p>"
            + "Evidence. " * 50
            + "</p></article></html>"
        )
        with patch.object(client, "_get", return_value=html):
            result = client.inspect(paper(), max_chars=100)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["text"]), 100)
        self.assertNotIn("evil", result["text"])
        with patch.object(client, "_get", side_effect=RetrievalError("HTTP 404")):
            result = client.inspect(paper())
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["text"], "")

    def test_http_retry_and_download_limit(self):
        client = ArxivClient(retries=1)
        error = HTTPError("https://export.arxiv.org", 503, "Busy", {}, None)
        with (
            patch(
                "research_digest.retrieval.arxiv_source.urlopen",
                side_effect=[error, io.BytesIO(FEED.encode())],
            ),
            patch("research_digest.retrieval.arxiv_source.time.sleep") as sleep,
        ):
            self.assertEqual(client._get("https://export.arxiv.org"), FEED)
            self.assertTrue(any(call.args[0] >= 3 for call in sleep.call_args_list))
        with (
            patch(
                "research_digest.retrieval.arxiv_source.urlopen",
                return_value=io.BytesIO(b"12345"),
            ),
            patch("research_digest.retrieval.arxiv_source.time.sleep"),
            self.assertRaises(RetrievalError),
        ):
            client._get("https://export.arxiv.org", max_bytes=3)


class StorageAndHarvestTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = PaperStore(Path(directory.name) / "digest.sqlite3")

    def test_pagination_cache_and_version_storage(self):
        arxiv = FakeArxiv()
        result = collect_metadata(
            arxiv,
            self.store,
            ("cs.CL", "cs.AI"),
            "2026-10-01",
            "2026-10-01",
            page_size=2,
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(arxiv.page_calls, [0, 2])
        second = FakeArxiv()
        self.assertEqual(
            collect_metadata(
                second,
                self.store,
                ("cs.AI", "cs.CL"),
                "2026-10-01",
                "2026-10-01",
                page_size=2,
            ),
            result,
        )
        self.assertEqual(second.page_calls, [])
        self.store.save_page(
            "another query", 0, 2, Page(1, 0, [replace(paper(), version=2)])
        )
        with self.store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM papers").fetchone()[0], 4)

    def test_interrupted_harvest_resumes_after_cached_page(self):
        arxiv = FakeArxiv()
        original = arxiv.page

        def fail_second(query, start, size):
            if start:
                raise RetrievalError("Temporary error")
            return original(query, start, size)

        with (
            patch.object(arxiv, "page", side_effect=fail_second),
            self.assertRaises(RetrievalError),
        ):
            collect_metadata(
                arxiv, self.store, ("cs.CL",), "2026-10-01", "2026-10-01", page_size=2
            )
        next_client = FakeArxiv()
        result = collect_metadata(
            next_client, self.store, ("cs.CL",), "2026-10-01", "2026-10-01", page_size=2
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(next_client.page_calls, [2])

    def test_incomplete_duplicate_and_wrong_scope_results_fail(self):
        cases = [
            Page(2, 0, [paper(), paper()]),
            Page(1, 0, [replace(paper(), categories=["math.OC"])]),
            Page(1, 0, [replace(paper(), published="2026-09-30T00:00:00Z")]),
            Page(2, 0, []),
        ]
        for index, response in enumerate(cases):
            store = PaperStore(self.store.path.parent / f"case{index}.sqlite3")
            client = SimpleNamespace(page=Mock(return_value=response))
            with self.subTest(index=index), self.assertRaises(RetrievalError):
                collect_metadata(
                    client, store, ("cs.CL",), "2026-10-01", "2026-10-01", page_size=2
                )

    def test_bm25_formula_and_tie_order(self):
        doc = paper(abstract="llm")
        doc = replace(doc, title="llm")
        self.assertAlmostEqual(
            bm25([doc], "llm")[0].score, math.log(1 + 0.5 / 1.5) * 2 * 2.5 / 3.5
        )
        relevant, unrelated = paper(1, "llm evaluation"), replace(
            paper(2), title="Plants", abstract="botany"
        )
        self.assertEqual(bm25([unrelated, relevant], "llm")[0].paper, relevant)
        self.assertEqual(
            [x.paper.paper_id for x in bm25([paper(2), paper(1)], "absent")],
            [paper(1).paper_id, paper(2).paper_id],
        )

    def test_large_window_splits_without_silently_truncating(self):
        calls = []

        def page(query, offset, size):
            calls.append(query)
            start, end = re.findall(r"\d{12}", query)
            if start.endswith("0000") and end.endswith("2359"):
                return Page(2, offset, [])
            suffix = "0000" if start.endswith("0000") else "1200"
            candidate = replace(
                paper(len(calls)), published=f"2026-10-01T{suffix[:2]}:{suffix[2:]}:00Z"
            )
            return Page(1, offset, [candidate])

        with patch("research_digest.retrieval.arxiv_source.MAX_WINDOW_RESULTS", 1):
            result = collect_metadata(
                SimpleNamespace(page=page),
                self.store,
                ("cs.CL",),
                "2026-10-01",
                "2026-10-01",
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result), 2)

    def test_two_finishing_runs_cannot_commit_the_same_recommendation(self):
        state = {"profile": profile().to_dict(), "config": {"new_only": True}}
        first = self.store.create_run("test-user", state)
        second = self.store.create_run("test-user", state)
        result = {"status": "completed", "papers": [{"paper_id": paper().paper_id}]}
        self.store.finish(first, state, result)
        with self.assertRaisesRegex(ValueError, "already recommended"):
            self.store.finish(second, state, result)
        self.assertIsNone(self.store.load_run(second)[2])

    def test_two_finishing_runs_allow_repeated_recommendations_by_default(self):
        state = {"profile": profile().to_dict(), "config": {"new_only": False}}
        first = self.store.create_run("test-user", state)
        second = self.store.create_run("test-user", state)
        result = {"status": "completed", "papers": [{"paper_id": paper().paper_id}]}
        self.store.finish(first, state, result)
        self.store.finish(second, state, result)
        self.assertEqual(self.store.load_run(second)[2], result)
        self.assertEqual(self.store.recommended("test-user"), {paper().paper_id})


class AgentTests(unittest.TestCase):
    def setUp(self):
        x_patch = patch("research_digest.agent.digest_agent.XBrowserClient")
        x_class = x_patch.start()
        self.addCleanup(x_patch.stop)
        x_class.return_value.collect.side_effect = (
            lambda p, start, end: aggregate_attention(
                p, [], start, end, errors=["X login is not configured in tests"]
            )
        )
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = PaperStore(Path(directory.name) / "digest.sqlite3")
        self.arxiv = FakeArxiv()

    def run_agent(self, client=None, settings=None):
        client = FakeLLM() if client is None else client
        agent = DigestAgent(self.store, client=client, arxiv=self.arxiv)
        run_id = agent.start(profile(), config() if settings is None else settings)
        return agent, agent.run(run_id)

    def test_disabled_x_skips_collection_and_removes_tool_and_feature(self):
        client = FakeLLM()
        agent, result = self.run_agent(client, config(x_enabled=False))

        self.assertEqual(result["status"], "completed")
        agent.x_client.collect.assert_not_called()
        context = client.calls[-1][1]
        self.assertFalse(context["bounds"]["x_enabled"])
        self.assertNotIn("analyze_twitter", [tool["name"] for tool in context["tools"]])
        self.assertEqual(context["candidates"][0]["x_attention"]["status"], "disabled")
        self.assertIsNone(result["x_feature"])
        self.assertFalse(self.store.load_run(agent.run_id)[1]["config"]["x_enabled"])
        self.assertFalse(any("X attention" in note for note in result["limitations"]))
        with self.assertRaisesRegex(ValueError, "X analysis is disabled"):
            agent._analyze_twitter(paper().paper_id)
        agent.x_client.collect.assert_not_called()

    def test_cli_prints_readable_digest_and_preserves_json_export_and_trace(self):
        agent, result = self.run_agent()
        export = self.store.path.parent / "digest.json"
        argv = [
            "run_digest.py",
            "--resume",
            agent.run_id,
            "--db",
            str(self.store.path),
            "--output",
            str(export),
        ]
        output = io.StringIO()
        with (
            patch("sys.argv", argv),
            redirect_stdout(output),
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(run_digest.main(), 0)
        rendered = output.getvalue()
        selected = result["papers"][0]
        for value in (
            selected["title"],
            selected["url"],
            selected["summary"],
            selected["relevance"],
            result["limitations"][0],
        ):
            self.assertIn(value, rendered)
        self.assertTrue(rendered.startswith("Research digest for test-user"))
        self.assertEqual(json.loads(export.read_text()), json.loads(json.dumps(result)))

        for args, expected in (
            (argv + ["--format", "json"], json.loads(json.dumps(result))),
            (
                [
                    "run_digest.py",
                    "--trace",
                    agent.run_id,
                    "--db",
                    str(self.store.path),
                ],
                self.store.events(agent.run_id),
            ),
        ):
            output = io.StringIO()
            with (
                patch("sys.argv", args),
                redirect_stdout(output),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(run_digest.main(), 0)
            self.assertEqual(json.loads(output.getvalue()), expected)

    def test_readable_output_distinguishes_empty_digest_from_budget_exhaustion(self):
        _, result = self.run_agent(settings=config(max_steps=0))
        self.assertIn("could not be completed", format_digest(result))
        self.assertNotIn("recommended paper", format_digest(result))
        self.assertIn(
            "No papers were selected", format_digest({**result, "status": "completed"})
        )

    def test_screening_retries_bad_quote_with_feedback_and_charges_calls(self):
        client = FakeLLM()
        create = client.responses.create
        screening_payloads = []

        def invalid_once(**kwargs):
            response = create(**kwargs)
            if kwargs["text"]["format"]["name"] == "screen_papers":
                screening_payloads.append(json.loads(kwargs["input"]))
                if len(screening_payloads) == 1:
                    data = json.loads(response.output_text)
                    data["assessments"][0]["evidence_quote"] = "Invented quotation"
                    return SimpleNamespace(output_text=json.dumps(data))
            return response

        client.responses.create = invalid_once
        agent, result = self.run_agent(client)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stats"]["llm_calls"], 3)
        self.assertEqual(len(screening_payloads), 2)
        self.assertIn("Invented quotation", screening_payloads[1]["validation_error"])
        self.assertEqual(
            screening_payloads[0]["papers"], screening_payloads[1]["papers"]
        )
        rejected = [
            e
            for e in self.store.events(agent.run_id)
            if e["type"] == "invalid_screening"
        ]
        self.assertEqual(len(rejected), 1)

    def test_permanently_bad_screening_is_bounded_and_never_cached(self):
        client = FakeLLM()
        create = client.responses.create

        def always_invalid(**kwargs):
            response = create(**kwargs)
            data = json.loads(response.output_text)
            data["assessments"][0]["evidence_quote"] = "Invented quotation"
            return SimpleNamespace(output_text=json.dumps(data))

        client.responses.create = always_invalid
        agent = DigestAgent(self.store, client=client, arxiv=self.arxiv)
        run_id = agent.start(profile(), config())
        with self.assertRaisesRegex(ValueError, "Invented quotation"):
            agent.run(run_id)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(self.store.load_run(run_id)[1]["screened"], {})
        self.assertIsNone(
            self.store.cached_screening(agent._screening_cache_key(paper()))
        )

    def test_screening_retry_respects_total_call_budget(self):
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=Mock(
                    return_value=SimpleNamespace(output_text='{"assessments": []}')
                )
            )
        )
        agent, result = self.run_agent(client, config(max_llm_calls=1))
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(result["stats"]["llm_calls"], 1)
        self.assertEqual(agent.state["screened"], {})

    def test_live_trace_is_emitted_after_persistence_and_before_inference(self):
        emitted = []
        output = io.StringIO()

        def on_event(event):
            self.assertEqual(self.store.events(agent.run_id)[-1], event)
            emitted.append(event)
            run_digest._print_event(event)

        def policy(context):
            pid = context["candidates"][0]["paper_id"]
            if "inspect:" + pid not in context["observations"]:
                return decision("inspect_paper", pid)
            return finish(context)

        client = FakeLLM(policy)
        create = client.responses.create

        def checked_create(**kwargs):
            self.assertEqual(emitted[-1]["type"], "llm_request")
            self.assertIn('"llm_request"', output.getvalue())
            return create(**kwargs)

        client.responses.create = checked_create
        agent = DigestAgent(
            self.store, client=client, arxiv=self.arxiv, on_event=on_event
        )
        run_id = agent.start(profile(), config())

        with redirect_stderr(output):
            result = agent.run(run_id)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(emitted, self.store.events(run_id))
        for event_type in ("screening_result", "action", "observation"):
            self.assertIn(f'"type": "{event_type}"', output.getvalue())

    def test_complete_react_loop_inspect_twitter_expand_finish(self):
        def policy(context):
            pid = context["candidates"][0]["paper_id"]
            observations = context["observations"]
            if "inspect:" + pid not in observations:
                return decision("inspect_paper", pid)
            if "twitter:" + pid not in observations:
                return decision("analyze_twitter", pid)
            if not any(key.startswith("expand:") for key in observations):
                return decision("expand_shortlist")
            return finish(context)

        client = FakeLLM(policy)
        agent, result = self.run_agent(client)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stats"]["steps"], 4)
        self.assertEqual(result["stats"]["screened"], 2)
        self.assertEqual(result["stats"]["llm_calls"], 6)
        events = self.store.events(agent.run_id)
        self.assertEqual(
            [x["decision"]["action"] for x in events if x["type"] == "action"],
            ["inspect_paper", "analyze_twitter", "expand_shortlist", "finish"],
        )
        self.assertEqual(len(self.arxiv.inspect_calls), 1)
        self.assertEqual(
            self.store.recommended("test-user"), {result["papers"][0]["paper_id"]}
        )
        calls = len(client.calls)
        self.assertEqual(agent.run(agent.run_id), result)
        self.assertEqual(len(client.calls), calls)

    def test_new_mode_excludes_previously_recommended_ids(self):
        _, first = self.run_agent()
        _, second = self.run_agent(settings=config(new_only=True))
        self.assertNotEqual(
            first["papers"][0]["paper_id"], second["papers"][0]["paper_id"]
        )
        self.assertEqual(second["stats"]["fetched_count"], 3)
        self.assertEqual(second["stats"]["candidate_count"], 2)

    def test_default_run_keeps_previously_recommended_papers_eligible(self):
        _, first = self.run_agent()
        agent, second = self.run_agent()
        self.assertEqual(
            first["papers"][0]["paper_id"], second["papers"][0]["paper_id"]
        )
        self.assertEqual(second["stats"]["candidate_count"], 3)
        self.assertFalse(self.store.load_run(agent.run_id)[1]["config"]["new_only"])

    def test_resuming_legacy_run_preserves_its_history_exclusion(self):
        agent = DigestAgent(self.store, client=FakeLLM(), arxiv=self.arxiv)
        run_id = agent.start(profile(), config())
        agent.state["config"].pop("new_only")
        agent._persist_run()
        agent._restore_run(run_id)
        self.assertTrue(agent.config.new_only)

    def test_budgets_stop_without_committing_recommendations(self):
        for settings in (config(max_llm_calls=0), config(max_steps=0)):
            with self.subTest(settings=settings):
                _, result = self.run_agent(settings=settings)
                self.assertEqual(result["status"], "budget_exhausted")
                self.assertEqual(result["papers"], [])
                self.assertEqual(self.store.recommended("test-user"), set())

    def test_empty_corpus_needs_no_llm(self):
        self.arxiv.papers = []
        client = FakeLLM()
        _, result = self.run_agent(client)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["papers"], [])
        self.assertEqual(client.calls, [])

    def test_unknown_tool_and_unscreened_id_are_rejected_then_corrected(self):
        actions = iter(
            [decision("shell"), decision("inspect_paper", "../../etc/passwd")]
        )

        def policy(context):
            return next(actions, None) or finish(context)

        agent, result = self.run_agent(FakeLLM(policy))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.arxiv.inspect_calls, [])
        invalid_events = [
            event
            for event in self.store.events(agent.run_id)
            if event["type"] == "invalid_action"
        ]
        self.assertEqual(len(invalid_events), 2)
        self.assertEqual(invalid_events[0]["decision"]["action"], "shell")
        self.assertEqual(
            invalid_events[1]["decision"]["paper_id"], "../../etc/passwd"
        )

    def test_fabricated_final_evidence_is_rejected(self):
        count = 0

        def policy(context):
            nonlocal count
            count += 1
            output = finish(context)
            if count == 1:
                output["selections"][0]["evidence"][0][
                    "quote"
                ] = "This quote is invented."
            return output

        agent, result = self.run_agent(FakeLLM(policy))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stats"]["steps"], 2)
        self.assertTrue(
            any(x["type"] == "invalid_action" for x in self.store.events(agent.run_id))
        )

    def test_screening_requires_every_id_and_exact_evidence(self):
        for assessments in (
            [],
            [
                {
                    "paper_id": "unknown",
                    "label": "relevant",
                    "reason": "Yes",
                    "evidence_quote": "invented",
                }
            ],
        ):
            client = SimpleNamespace(
                responses=SimpleNamespace(
                    create=Mock(
                        return_value=SimpleNamespace(
                            output_text=json.dumps({"assessments": assessments})
                        )
                    )
                )
            )
            with self.subTest(assessments=assessments), self.assertRaises(ValueError):
                self.run_agent(client)
            self.assertEqual(self.store.recommended("test-user"), set())

    def test_failed_expansion_resumes_without_double_charging_or_skipping(self):
        def policy(context):
            if len(context["candidates"]) == 1:
                return decision("expand_shortlist")
            return finish(context)

        client = FakeLLM(policy)
        original = client.responses.create
        failed = False

        def create(**kwargs):
            nonlocal failed
            payload = json.loads(kwargs["input"])
            if (
                kwargs["text"]["format"]["name"] == "screen_papers"
                and payload["papers"][0]["paper_id"] == paper(2).paper_id
                and not failed
            ):
                failed = True
                raise CodexError("Temporary CLI failure")
            return original(**kwargs)

        client.responses.create = create
        agent = DigestAgent(self.store, client=client, arxiv=self.arxiv)
        run_id = agent.start(profile(), config())
        with self.assertRaises(CodexError):
            agent.run(run_id)
        status, state, _ = self.store.load_run(run_id)
        self.assertEqual(status, "failed")
        self.assertEqual(state["expansions"], 1)
        result = agent.run(run_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stats"]["expansions"], 1)
        self.assertEqual(result["stats"]["screened"], 2)
        self.assertEqual(result["stats"]["llm_calls"], 5)

    def test_cached_screenings_do_not_leave_holes_after_failure(self):
        cached = {
            "paper_id": paper(2).paper_id,
            "label": "relevant",
            "reason": "Known result",
            "evidence_quote": "We evaluate LLM agents",
        }
        key = hashlib.sha256(
            json.dumps(
                [
                    PIPELINE_VERSION,
                    profile().fingerprint,
                    "test-model",
                    paper(2).fingerprint,
                ]
            ).encode()
        ).hexdigest()
        self.store.save_screening(key, cached)
        bad = SimpleNamespace(
            responses=SimpleNamespace(create=Mock(side_effect=CodexError("Failure")))
        )
        agent = DigestAgent(self.store, client=bad, arxiv=self.arxiv)
        run_id = agent.start(profile(), config(shortlist_size=3, batch_size=3))
        with self.assertRaises(CodexError):
            agent.run(run_id)
        self.assertEqual(
            set(self.store.load_run(run_id)[1]["screened"]), {paper(2).paper_id}
        )
        agent.client = FakeLLM()
        result = agent.run(run_id)
        self.assertEqual(result["stats"]["screened"], 3)
        screened_ids = {
            item["paper_id"]
            for purpose, ctx in agent.client.calls
            if purpose == "screen_papers"
            for item in ctx["papers"]
        }
        self.assertEqual(screened_ids, {paper(1).paper_id, paper(3).paper_id})

    def test_inspection_failure_is_an_observation_not_a_fake_full_text(self):
        self.arxiv.document = Mock(
            return_value={"status": "unavailable", "text": "", "reason": "404"}
        )

        def policy(context):
            pid = context["candidates"][0]["paper_id"]
            if "inspect:" + pid not in context["observations"]:
                return decision("inspect_paper", pid)
            return finish(context)

        _, result = self.run_agent(FakeLLM(policy))
        self.assertEqual(result["status"], "completed")
        self.assertTrue(any("unavailable" in note for note in result["limitations"]))

    def test_tool_budget_and_corpus_exhaustion_are_visible_observations(self):
        for action, settings, expected in (
            ("inspect_paper", config(max_inspections=0), "budget_exhausted"),
            ("expand_shortlist", config(max_candidates=1), "exhausted"),
        ):
            observations = []

            def policy(context):
                if context["last_observation"] is None:
                    pid = (
                        context["candidates"][0]["paper_id"]
                        if action == "inspect_paper"
                        else ""
                    )
                    return decision(action, pid)
                observations.append(context["last_observation"]["result"]["status"])
                return decision("finish")

            with self.subTest(action=action):
                _, result = self.run_agent(FakeLLM(policy), settings)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(observations, [expected])
        self.assertEqual(self.arxiv.inspect_calls, [])

    def test_screening_cache_is_scoped_to_explicit_model_and_profile(self):
        _, first = self.run_agent(settings=config(max_steps=0))
        _, cached = self.run_agent(settings=config(max_steps=0))
        _, different_model = self.run_agent(
            settings=config(max_steps=0, model="other-model")
        )
        self.assertEqual(first["stats"]["llm_calls"], 1)
        self.assertEqual(cached["stats"]["llm_calls"], 0)
        self.assertEqual(different_model["stats"]["llm_calls"], 1)
        agent = DigestAgent(self.store, client=FakeLLM(), arxiv=self.arxiv)
        run_id = agent.start(
            replace(profile(), exclusions=("New exclusion",)), config(max_steps=0)
        )
        self.assertEqual(agent.run(run_id)["stats"]["llm_calls"], 1)

    def test_unknown_default_model_cache_is_reused_only_within_a_run(self):
        for _ in range(2):
            _, result = self.run_agent(settings=config(max_steps=0, model=None))
            self.assertEqual(result["stats"]["llm_calls"], 1)

    def test_unlimited_run_exceeds_finite_defaults_and_finishes_at_corpus_end(self):
        self.arxiv.papers = [paper(index) for index in range(1, 7)]
        settings = config(**{field: -1 for field in UNLIMITED_LIMIT_FIELDS})

        def policy(context):
            for budget in ("llm_calls", "steps", "inspections", "expansions"):
                self.assertIsNone(context["remaining"][budget])
            self.assertIsNone(context["bounds"]["digest_size"])

            for candidate in context["candidates"]:
                paper_id = candidate["paper_id"]
                if f"twitter:{paper_id}" not in context["observations"]:
                    return decision("analyze_twitter", paper_id)
                if f"inspect:{paper_id}" not in context["observations"]:
                    return decision("inspect_paper", paper_id)

            if context["remaining"]["candidates"]:
                return decision("expand_shortlist")

            selections = [
                {
                    "paper_id": candidate["paper_id"],
                    "summary": "The paper evaluates LLM agents.",
                    "relevance": "It concerns agent evaluation.",
                    "evidence": [
                        {"source": "abstract", "quote": "We evaluate LLM agents"}
                    ],
                }
                for candidate in context["candidates"]
            ]
            return decision("finish", selections=selections)

        agent, result = self.run_agent(FakeLLM(policy), settings)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["papers"]), 6)
        self.assertEqual(result["stats"]["screened"], 6)
        self.assertEqual(result["stats"]["steps"], 18)
        self.assertEqual(result["stats"]["llm_calls"], 24)
        self.assertEqual(result["stats"]["inspections"], 6)
        self.assertEqual(result["stats"]["expansions"], 5)

        saved_config = self.store.load_run(agent.run_id)[1]["config"]
        for field in UNLIMITED_LIMIT_FIELDS:
            self.assertIsNone(saved_config[field])

    def test_unlimited_context_disables_only_the_local_size_check(self):
        with self.assertRaisesRegex(ValueError, "max_context_chars"):
            self.run_agent(settings=config(max_context_chars=1))

        _, result = self.run_agent(settings=config(max_context_chars=-1))
        self.assertEqual(result["status"], "completed")

    def test_unlimited_configuration_can_resume_after_inference_failure(self):
        broken_client = SimpleNamespace(
            responses=SimpleNamespace(
                create=Mock(side_effect=CodexError("Temporary failure"))
            )
        )
        agent = DigestAgent(self.store, client=broken_client, arxiv=self.arxiv)
        settings = config(**{field: -1 for field in UNLIMITED_LIMIT_FIELDS})
        run_id = agent.start(profile(), settings)

        with self.assertRaises(CodexError):
            agent.run(run_id)

        resumed_agent = DigestAgent(self.store, client=FakeLLM(), arxiv=self.arxiv)
        result = resumed_agent.run(run_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stats"]["llm_calls"], 3)


class UnlimitedConfigTests(unittest.TestCase):
    def test_minus_one_normalizes_and_round_trips_as_json_null(self):
        settings = config(**{field: -1 for field in UNLIMITED_LIMIT_FIELDS})
        settings.validate()
        serialized = json.loads(json.dumps(asdict(settings)))
        restored = DigestConfig(**serialized)
        restored.validate()
        self.assertEqual(asdict(restored), serialized)

        for field in UNLIMITED_LIMIT_FIELDS:
            self.assertIsNone(getattr(settings, field))

    def test_invalid_limits_and_nonpositive_sizes_still_fail(self):
        for field in UNLIMITED_LIMIT_FIELDS:
            for value in (-2, True, "unlimited"):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    config(**{field: value}).validate()

        for field in ("shortlist_size", "batch_size", "expansion_size", "page_size"):
            for value in (-1, 0, None):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    config(**{field: value}).validate()

        for field in ("max_candidates", "digest_size", "max_context_chars"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                config(**{field: 0}).validate()

    def test_zero_call_budgets_and_finite_candidate_constraints_are_preserved(self):
        config(
            max_steps=0, max_llm_calls=0, max_inspections=0, max_expansions=0
        ).validate()
        config(shortlist_size=100, max_candidates=-1).validate()

        with self.assertRaises(ValueError):
            config(shortlist_size=4, max_candidates=3).validate()
        with self.assertRaises(ValueError):
            config(page_size=1001).validate()

    def test_cli_accepts_all_unlimited_flags(self):
        arguments = [
            "run_digest.py",
            "--user-id",
            "test-user",
            "--category",
            "cs.CL",
            "--date-from",
            "2026-10-01",
            "--date-to",
            "2026-10-01",
        ]
        for field in UNLIMITED_LIMIT_FIELDS:
            arguments.extend(["--" + field.replace("_", "-"), "-1"])

        with patch("sys.argv", arguments):
            settings = run_digest._build_config(run_digest._parse_arguments())

        for field in UNLIMITED_LIMIT_FIELDS:
            self.assertIsNone(getattr(settings, field))


class ProfileCategoryCLITests(unittest.TestCase):
    def test_no_x_flag_disables_x_in_config(self):
        arguments = [
            "run_digest.py",
            "--user-id",
            "test-user",
            "--category",
            "cs.CL",
            "--date-from",
            "2026-10-20",
            "--date-to",
            "2026-10-24",
        ]
        for extra, expected in (([], True), (["--no-X"], False)):
            with self.subTest(extra=extra), patch("sys.argv", arguments + extra):
                settings = run_digest._build_config(run_digest._parse_arguments())
                self.assertEqual(settings.x_enabled, expected)

    def test_new_flag_is_optional_and_persisted_in_config(self):
        arguments = [
            "run_digest.py",
            "--user-id",
            "test-user",
            "--category",
            "cs.CL",
            "--date-from",
            "2026-10-20",
            "--date-to",
            "2026-10-24",
        ]
        for extra, expected in (([], False), (["--new"], True)):
            with self.subTest(extra=extra), patch("sys.argv", arguments + extra):
                self.assertEqual(
                    run_digest._build_config(run_digest._parse_arguments()).new_only,
                    expected,
                )

    def test_verbose_is_accepted_for_new_and_resumed_runs(self):
        for options in (
            [
                "--user-id",
                "test-user",
                "--date-from",
                "2026-10-20",
                "--date-to",
                "2026-10-24",
            ],
            ["--resume", "saved-run"],
        ):
            with (
                self.subTest(options=options),
                patch("sys.argv", ["run_digest.py", *options, "--verbose"]),
            ):
                self.assertTrue(run_digest._parse_arguments().verbose)

    def arguments(self, db):
        with patch(
            "sys.argv",
            [
                "run_digest.py",
                "--user-id",
                "test-user",
                "--date-from",
                "2026-10-20",
                "--date-to",
                "2026-10-24",
                "--db",
                str(db),
            ],
        ):
            return run_digest._parse_arguments()

    def test_missing_categories_are_inferred_saved_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "digest.sqlite3"
            store = ProfileStore(db)
            original = profile()
            store.save(original)
            args = self.arguments(db)

            with (
                patch(
                    "research_digest.cli.run_digest.infer_arxiv_categories",
                    return_value=("cs.CL", "cs.LG"),
                ) as infer,
                redirect_stderr(io.StringIO()),
            ):
                loaded = run_digest._load_profile(args)
                settings = run_digest._build_config(args, loaded)
                self.assertEqual(settings.categories, ("cs.CL", "cs.LG"))
                self.assertEqual(run_digest._load_profile(args), loaded)
                infer.assert_called_once_with(original, model=None)

            self.assertEqual(store.load("test-user").revision, 2)
            self.assertEqual(store.load("test-user", revision=1).profile, original)
            self.assertEqual(
                loaded, replace(original, arxiv_categories=("cs.CL", "cs.LG"))
            )

    def test_override_is_run_only_and_skips_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "digest.sqlite3"
            store = ProfileStore(db)
            original = replace(profile(), arxiv_categories=("cs.CL",))
            saved = store.save(original)
            args = self.arguments(db)
            args.category = ["cs.AI"]

            with patch(
                "research_digest.cli.run_digest.infer_arxiv_categories"
            ) as infer:
                loaded = run_digest._load_profile(args)
                self.assertEqual(
                    run_digest._build_config(args, loaded).categories, ("cs.AI",)
                )
                infer.assert_not_called()

            self.assertEqual(store.load("test-user"), saved)

    def test_inference_failure_preserves_saved_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "digest.sqlite3"
            store = ProfileStore(db)
            saved = store.save(profile())
            args = self.arguments(db)

            with (
                patch(
                    "research_digest.cli.run_digest.infer_arxiv_categories",
                    side_effect=ValueError("Invalid categories"),
                ),
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(ValueError):
                    run_digest._load_profile(args)

            self.assertEqual(store.load("test-user"), saved)


if __name__ == "__main__":
    unittest.main()
