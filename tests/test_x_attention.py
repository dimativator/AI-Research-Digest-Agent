import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from research_digest.cli.digest_output import format_digest
from research_digest.retrieval.x_attention import (
    aggregate_attention,
    featured_paper,
    match_post,
    parse_count,
)
from research_digest.retrieval.x_source import (
    POST_SCRIPT,
    POST_SELECTOR,
    XBrowserClient,
    XAccessError,
    parse_rendered_post,
    search_queries,
)
from research_digest.storage.paper_store import PaperStore
from test_digest_pipeline import (
    FakeArxiv,
    FakeLLM,
    config,
    decision,
    finish,
    paper,
    profile,
)
from research_digest.agent.digest_agent import DigestAgent


def post(index=1, *, pid="2610.00001", author=None, likes=10):
    return {
        "id": str(index),
        "author": author or f"researcher{index}",
        "url": f"https://x.com/{author or f'researcher{index}'}/status/{index}",
        "text": f"New paper https://arxiv.org/abs/{pid}",
        "links": [f"https://arxiv.org/abs/{pid}"],
        "created_at": "2026-10-01T12:00:00Z",
        "metrics": {"likes": likes, "reposts": 3, "replies": 1, "quotes": None},
    }


class AttentionTests(unittest.TestCase):
    def test_confirmed_access_block_still_stops_further_requests(self):
        client = XBrowserClient()
        client._page = Mock()
        client._search = Mock(side_effect=XAccessError("X rate limit exceeded"))
        self.assertEqual(
            client.collect(paper(), "2026-10-01", "2026-10-01")["status"], "unavailable"
        )
        self.assertEqual(
            client.collect(paper(2), "2026-10-01", "2026-10-01")["status"],
            "unavailable",
        )
        self.assertEqual(client._search.call_count, 1)

    def test_one_search_timeout_does_not_disable_later_papers(self):
        client = XBrowserClient()
        client._page = Mock()
        client._search = Mock(
            side_effect=[
                RuntimeError(
                    "Arc page did not become ready; login or search may be blocked."
                ),
                [],
                [],
            ]
        )
        first = client.collect(paper(), "2026-10-01", "2026-10-01")
        second = client.collect(paper(2), "2026-10-01", "2026-10-01")
        self.assertEqual(first["status"], "unavailable")
        self.assertEqual(second["status"], "no_matches")
        self.assertEqual(client._search.call_count, 3)

    def test_matches_arxiv_variants_and_titles_but_rejects_nearby_ids(self):
        for text in (
            "2610.00001",
            "https://arxiv.org/abs/2610.00001v2",
            "https://arxiv.org/pdf/2610.00001v2.pdf",
            "https://arxiv.org/html/2610.00001",
        ):
            with self.subTest(text=text):
                self.assertEqual(match_post({"text": text}, paper()), "arxiv_id")
        self.assertEqual(
            match_post({"text": paper().title.upper()}, paper()), "exact_title"
        )
        for text in (
            "2610.000010",
            "2610.00002",
            "LLM evaluation unrelated discussion",
        ):
            self.assertIsNone(match_post({"text": text}, paper()))

    def test_count_parsing_preserves_missing_metrics(self):
        for raw, expected in (
            ("1.2K", 1200),
            ("2M", 2_000_000),
            ("1,240", 1240),
            ("0", 0),
            (None, None),
            ("", None),
            ("like", None),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(parse_count(raw), expected)

    def test_deduplicates_filters_dates_and_does_not_treat_errors_as_zero(self):
        first = post()
        outside = {**post(2), "created_at": "2026-09-30T23:59:59Z"}
        unrelated = post(3, pid="2610.99999")
        result = aggregate_attention(
            paper(), [first, first, outside, unrelated], "2026-10-01", "2026-10-01"
        )
        self.assertEqual(result["matched_posts"], 1)
        self.assertEqual(result["unique_accounts"], 1)
        self.assertEqual(result["engagement"]["likes"], 10)
        self.assertIsNone(result["engagement"]["quotes"])
        self.assertEqual(result["coverage"], "partial")
        unavailable = aggregate_attention(
            paper(), [], "2026-10-01", "2026-10-01", errors=["login expired"]
        )
        self.assertEqual(unavailable["level"], "unavailable")
        self.assertIsNone(unavailable["engagement"]["likes"])
        self.assertEqual(
            aggregate_attention(paper(), [], "2026-10-01", "2026-10-01")["status"],
            "no_matches",
        )

    def test_ranks_papers_by_accounts_and_posts_by_engagement(self):
        first = aggregate_attention(
            paper(), [post(likes=100_000)], "2026-10-01", "2026-10-01"
        )
        second = aggregate_attention(
            paper(2),
            [
                post(2, pid=paper(2).paper_id, likes=2),
                post(3, pid=paper(2).paper_id, likes=200),
            ],
            "2026-10-01",
            "2026-10-01",
        )
        self.assertEqual(
            featured_paper({paper().paper_id: first, paper(2).paper_id: second}),
            paper(2).paper_id,
        )
        self.assertEqual(second["posts"][0]["id"], "3")
        self.assertIsNone(featured_paper({}))

    def test_queries_use_inclusive_post_dates(self):
        queries = search_queries(paper(), "2026-10-20", "2026-10-24")
        self.assertTrue(all("until:2026-10-25" in query for query in queries))
        self.assertIn("url:arxiv.org/pdf/2610.00001", queries[0])

    def test_arc_permission_failure_returns_unavailable(self):
        with patch(
            "research_digest.retrieval.x_source.ArcTab.open",
            side_effect=RuntimeError(
                "Arc scripting unavailable. Run python setup_x.py"
            ),
        ):
            client = XBrowserClient()
            result = client.collect(paper(), "2026-10-01", "2026-10-01")
            self.assertEqual(result["status"], "unavailable")
            self.assertIn("setup_x.py", result["errors"][0])
            self.assertIsNone(client._page)
            client.close()

    def test_cache_expires_and_raw_posts_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            store = PaperStore(Path(directory) / "digest.sqlite3")
            result = aggregate_attention(paper(), [post()], "2026-10-01", "2026-10-01")
            store.save_x_attention("key", result)
            self.assertEqual(PaperStore(store.path).cached_x_attention("key"), result)
            result["collected_at"] = (
                datetime.now(timezone.utc) - timedelta(hours=7)
            ).isoformat()
            store.save_x_attention("key", result)
            self.assertIsNone(store.cached_x_attention("key"))


class XPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = PaperStore(Path(self.directory.name) / "digest.sqlite3")
        self.x = Mock()
        self.x.collect.side_effect = lambda p, start, end: aggregate_attention(
            p,
            [post(i, pid=p.paper_id) for i in range(1, int(p.paper_id[-1]) + 1)],
            start,
            end,
        )

    def test_expansion_enriches_all_candidates_and_features_one_entry(self):
        contexts = []

        def policy(context):
            contexts.append(context)
            self.assertTrue(
                all("x_attention" in item for item in context["candidates"])
            )
            if len(context["candidates"]) == 1:
                return decision("expand_shortlist")
            return finish(context)

        agent = DigestAgent(
            self.store, arxiv=FakeArxiv(), client=FakeLLM(policy), x_client=self.x
        )
        run_id = agent.start(profile(), config())
        result = agent.run(run_id)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.x.collect.call_count, 2)

        self.assertEqual(contexts[0]["featured_x_paper_id"], paper().paper_id)
        self.assertEqual(contexts[-1]["featured_x_paper_id"], paper(2).paper_id)
        self.assertEqual(result["x_feature"]["paper_id"], paper(2).paper_id)
        self.assertEqual(sum(item["featured_on_x"] for item in result["papers"]), 1)
        self.assertEqual(len(result["papers"]), 1)
        rendered = format_digest(result)
        self.assertEqual(
            rendered.count("Most discussed on X among the papers checked"), 1
        )
        self.assertIn("https://x.com/researcher1/status/1", rendered)
        self.assertNotIn("posts", contexts[-1]["candidates"][0]["x_attention"])
        self.assertTrue(self.x.close.called)

        # Completed runs render the saved feature without collecting again.
        self.assertEqual(agent.run(run_id), result)
        self.assertEqual(self.x.collect.call_count, 2)

    def test_resuming_old_collector_discards_only_old_x_observations(self):
        agent = DigestAgent(
            self.store, arxiv=FakeArxiv(), client=FakeLLM(), x_client=self.x
        )
        run_id = agent.start(profile(), config())
        agent.state.update(
            x_collector="chromium-v1",
            x_attention={"old": {"status": "unavailable"}},
            observations={
                "twitter:old": {"status": "unavailable"},
                "inspect:old": {"status": "ok"},
            },
        )
        agent._persist_run()
        agent._restore_run(run_id)
        self.assertEqual(agent.state["x_attention"], {})
        self.assertNotIn("twitter:old", agent.state["observations"])
        self.assertIn("inspect:old", agent.state["observations"])

    def test_feature_is_required_but_other_entries_have_no_x_post_links(self):
        def policy(context):
            selections = [
                finish({**context, "candidates": [item]})["selections"][0]
                for item in context["candidates"]
            ]
            return decision("finish", selections=selections)

        agent = DigestAgent(
            self.store, arxiv=FakeArxiv(), client=FakeLLM(policy), x_client=self.x
        )
        run_id = agent.start(profile(), config(shortlist_size=2))
        result = agent.run(run_id)
        self.assertEqual(len(result["papers"]), 2)
        self.assertEqual(sum(p["featured_on_x"] for p in result["papers"]), 1)
        self.assertNotIn("top_posts", result["papers"][0])
        bad = decision("finish", selections=[result["papers"][0]])
        # Feed the selection shape expected by the decision protocol.
        bad["selections"] = [
            {
                key: value
                for key, value in result["papers"][0].items()
                if key in {"paper_id", "summary", "relevance", "evidence"}
            }
        ]
        with self.assertRaisesRegex(ValueError, "Include featured X paper"):
            agent._build_selections(bad)

    def test_unavailable_collection_does_not_prevent_digest(self):
        self.x.collect.side_effect = lambda p, start, end: aggregate_attention(
            p, [], start, end, errors=["login expired"]
        )
        agent = DigestAgent(
            self.store, arxiv=FakeArxiv(), client=FakeLLM(), x_client=self.x
        )
        result = agent.run(agent.start(profile(), config()))
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["x_feature"])
        self.assertFalse(result["papers"][0]["featured_on_x"])


class RenderedPostTests(unittest.TestCase):
    def test_browser_collector_distinguishes_matches_empty_search_and_blocking(self):
        from playwright.sync_api import sync_playwright

        account = (
            '<button data-testid="SideNav_AccountSwitcher_Button">Account</button>'
        )
        article = """<article data-testid="tweet">
          <a href="https://x.com/scientist/status/123"><time datetime="2026-10-01T12:00:00Z"></time></a>
          <div data-testid="tweetText">Paper 2610.00001</div>
          <button data-testid="like">5</button>
        </article>"""
        with sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True, channel="chromium")
            try:
                page = browser.new_page()
                for body, expected in (
                    (article, "ok"),
                    ("No results for this query", "no_matches"),
                    ("Something went wrong", "unavailable"),
                ):
                    with self.subTest(status=expected):
                        page.unroute("https://x.com/**")

                        def respond(route):
                            route.fulfill(content_type="text/html", body=account + body)

                        page.route("https://x.com/**", respond)
                        client = XBrowserClient(max_scrolls=1, max_posts=1)
                        client._page = page
                        result = client.collect(paper(), "2026-10-01", "2026-10-01")
                        self.assertEqual(result["status"], expected)
                        if expected == "ok":
                            self.assertEqual(result["matched_posts"], 1)
                            self.assertEqual(result["engagement"]["likes"], 5)
                        if expected == "unavailable":
                            self.assertIsNone(result["engagement"]["likes"])
            finally:
                browser.close()

    def test_dom_extraction_in_real_chromium(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True, channel="chromium")
            try:
                page = browser.new_page()
                page.set_content(
                    """<article data-testid="tweet">
                  <a href="https://x.com/scientist/status/123"><time datetime="2026-10-01T12:00:00Z"></time></a>
                  <div data-testid="tweetText">Paper 2610.00001</div>
                  <a href="https://t.co/example" title="https://arxiv.org/pdf/2610.00001v2.pdf">arxiv.org</a>
                  <button data-testid="like">1.2K</button>
                  <button data-testid="retweet" aria-label="3 reposts"></button>
                  <button data-testid="reply" aria-label="0 replies"></button>
                </article>"""
                )
                raw = page.locator(POST_SELECTOR).evaluate_all(POST_SCRIPT)[0]
                parsed = parse_rendered_post(raw)
                self.assertEqual(parsed["metrics"]["likes"], 1200)
                self.assertEqual(parsed["metrics"]["reposts"], 3)
                self.assertEqual(parsed["metrics"]["replies"], 0)
                self.assertIsNone(parsed["metrics"]["quotes"])
                self.assertIn("https://arxiv.org/pdf/2610.00001v2.pdf", parsed["links"])
            finally:
                browser.close()


if __name__ == "__main__":
    unittest.main()
