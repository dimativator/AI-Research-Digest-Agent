import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from research_digest.cli import create_profile
from research_digest.profiles.user_profile import (
    MAX_SOURCE_CHARS,
    ProfileStore,
    UserProfile,
    build_profile,
    infer_arxiv_categories,
)


DESCRIPTION = "I study evaluation of LLM agents. I do not want medical applications."


def draft():
    return {
        "summary": "Research on evaluating LLM agents, excluding medical applications.",
        "topics": [
            {
                "name": "LLM agent evaluation",
                "keywords": ["agent benchmarks", "LLM evaluation"],
                "source_id": "description",
                "evidence_quote": "evaluation of LLM agents",
            }
        ],
        "goals": ["Find reliable evaluation methods"],
        "exclusions": ["Medical applications"],
        "open_questions": [],
        "arxiv_categories": ["cs.CL", "cs.AI"],
    }


def fake_client(payload=None):
    return SimpleNamespace(
        responses=SimpleNamespace(
            create=Mock(
                return_value=SimpleNamespace(
                    output_text=json.dumps(draft() if payload is None else payload)
                )
            )
        )
    )


def profile(user_id="alice"):
    return build_profile(user_id, DESCRIPTION, client=fake_client())


class ProfileBuilderTests(unittest.TestCase):
    def test_infers_categories_and_persists_them(self):
        result = profile()
        self.assertEqual(result.arxiv_categories, ("cs.CL", "cs.AI"))

        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / "digest.sqlite3")
            store.save(result)
            self.assertEqual(
                store.load("alice").profile.arxiv_categories, ("cs.CL", "cs.AI")
            )

    def test_rejects_empty_or_malformed_inferred_categories(self):
        for categories in ([], ["cs AI"], "cs.CL", [42]):
            payload = {**draft(), "arxiv_categories": categories}
            with self.subTest(categories=categories), self.assertRaises(ValueError):
                build_profile("alice", DESCRIPTION, client=fake_client(payload))

    def test_legacy_category_inference_uses_profile_and_validates_response(self):
        original = replace(profile(), arxiv_categories=())
        client = fake_client({"arxiv_categories": ["cs.CL", "cs.AI", "cs.CL"]})
        self.assertEqual(
            infer_arxiv_categories(original, client=client), ("cs.CL", "cs.AI")
        )
        self.assertEqual(
            json.loads(client.responses.create.call_args.kwargs["input"]),
            original.to_dict(),
        )

        for payload in (
            {"arxiv_categories": []},
            {"arxiv_categories": ["cs AI"]},
            [],
            {"arxiv_categories": ["cs.CL"], "extra": True},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                infer_arxiv_categories(original, client=fake_client(payload))

    def test_preserves_explicit_constraints_and_categories(self):
        client = fake_client()
        result = build_profile(
            "alice",
            DESCRIPTION,
            goals=["Support my thesis"],
            exclusions=["Survey papers"],
            arxiv_categories=["cs.CL", "cs.AI", "cs.CL"],
            client=client,
            model="test-model",
        )
        self.assertEqual(result.user_id, "alice")
        self.assertEqual(result.model, "test-model")
        self.assertEqual(result.arxiv_categories, ("cs.CL", "cs.AI"))
        self.assertIn("Support my thesis", result.goals)
        self.assertIn("Survey papers", result.exclusions)
        self.assertIn("Medical applications", result.exclusions)
        call = client.responses.create.call_args.kwargs
        self.assertEqual(call["text"]["format"]["type"], "json_schema")
        self.assertEqual(
            json.loads(call["input"])["sources"]["description"], DESCRIPTION
        )

    def test_ranking_text_does_not_include_exclusions_or_sources(self):
        result = profile()
        self.assertIn("LLM evaluation", result.ranking_text())
        self.assertNotIn("medical", result.ranking_text().lower())
        self.assertNotIn("do not want", result.ranking_text())

    def test_evidence_can_reference_supplied_context(self):
        payload = draft()
        payload["topics"][0].update(
            source_id="context_1", evidence_quote="agent benchmarks"
        )
        result = build_profile(
            "alice",
            DESCRIPTION,
            context=["My project studies agent benchmarks."],
            client=fake_client(payload),
        )
        self.assertEqual(result.topics[0].source_id, "context_1")

    def test_rejects_fabricated_quotes_unknown_sources_and_empty_topics(self):
        for change in (
            {"source_id": "unknown"},
            {"evidence_quote": "invented evidence"},
            {"evidence_quote": ""},
            {"keywords": []},
        ):
            with self.subTest(change=change):
                payload = draft()
                payload["topics"][0].update(change)
                with self.assertRaises(ValueError):
                    build_profile("alice", DESCRIPTION, client=fake_client(payload))
        payload = draft()
        payload["topics"] = []
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_profile("alice", DESCRIPTION, client=fake_client(payload))

    def test_invalid_inputs_fail_before_llm_call(self):
        client = fake_client()
        for kwargs in (
            {"description": ""},
            {"description": "x" * (MAX_SOURCE_CHARS + 1)},
            {"description": DESCRIPTION, "arxiv_categories": ["cs AI"]},
            {"description": DESCRIPTION, "exclusions": "not a list"},
        ):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                build_profile("alice", client=client, **kwargs)
        client.responses.create.assert_not_called()

    def test_local_validation_does_not_rely_on_llm_schema_enforcement(self):
        for payload in (
            [],
            {**draft(), "extra": "value"},
            {**draft(), "goals": "bad"},
            {**draft(), "exclusions": [3]},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                build_profile("alice", DESCRIPTION, client=fake_client(payload))

    def test_serialization_round_trip_and_stable_fingerprint(self):
        result = profile()
        loaded = UserProfile.from_dict(result.to_dict())
        self.assertEqual(result, loaded)
        self.assertEqual(result.fingerprint, loaded.fingerprint)
        self.assertNotEqual(
            result.fingerprint,
            replace(result, exclusions=("New exclusion",)).fingerprint,
        )


class ProfileStoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "data" / "digest.sqlite3"
        self.store = ProfileStore(self.path)

    def test_persists_across_connections_and_skips_unchanged_latest(self):
        first = self.store.save(profile())
        second = ProfileStore(self.path).save(profile())
        self.assertEqual(first, second)
        self.assertEqual(first.revision, 1)
        self.assertEqual(ProfileStore(self.path).load("alice"), first)

    def test_new_revision_preserves_old_profile_and_allows_reversion(self):
        original = profile()
        self.store.save(original)
        changed = replace(original, exclusions=("Clinical papers",))
        second = self.store.save(changed)
        self.assertEqual(second.revision, 2)
        self.assertEqual(self.store.load("alice").profile, changed)
        self.assertEqual(self.store.load("alice", revision=1).profile, original)
        self.assertEqual(self.store.save(original).revision, 3)

    def test_user_isolation_and_parameterized_ids(self):
        self.store.save(profile())
        odd_id = "bob'; DROP TABLE user_profiles; --"
        self.store.save(profile(odd_id))
        self.assertEqual(self.store.load(odd_id).revision, 1)
        self.assertEqual(self.store.load("alice").profile.user_id, "alice")
        with self.assertRaises(KeyError):
            self.store.load("unknown")
        with self.assertRaises(ValueError):
            self.store.load("alice", revision=0)

    def test_corrupted_fingerprint_is_detected(self):
        import sqlite3

        self.store.save(profile())
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE user_profiles SET fingerprint = 'bad'")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.store.load("alice")


class ProfileCLITests(unittest.TestCase):
    def test_missing_file_does_not_call_llm_or_create_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digest.sqlite3"
            with (
                patch(
                    "sys.argv",
                    [
                        "create_profile.py",
                        str(Path(directory) / "missing.txt"),
                        "--user-id",
                        "alice",
                        "--db",
                        str(path),
                    ],
                ),
                patch("research_digest.cli.create_profile.build_profile") as build,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(create_profile.main(), 1)
                build.assert_not_called()
            self.assertFalse(path.exists())

    def test_file_is_structured_saved_automatically_and_can_be_shown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digest.sqlite3"
            source = Path(directory) / "profile.txt"
            source.write_text(DESCRIPTION, encoding="utf-8")
            argv = ["create_profile.py", "--user-id", "alice", "--db", str(path)]
            with (
                patch("sys.argv", argv + [str(source), "--model", "test-model"]),
                patch(
                    "research_digest.cli.create_profile.build_profile",
                    return_value=profile(),
                ) as build,
                patch(
                    "builtins.input", side_effect=AssertionError("Unexpected prompt")
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(create_profile.main(), 0)
                build.assert_called_once_with("alice", DESCRIPTION, model="test-model")
            output = io.StringIO()
            with (
                patch("sys.argv", argv + ["--show"]),
                patch("research_digest.cli.create_profile.build_profile") as build,
                redirect_stdout(output),
            ):
                self.assertEqual(create_profile.main(), 0)
                build.assert_not_called()
            saved = json.loads(output.getvalue())
            self.assertEqual(saved["profile"]["user_id"], "alice")
            self.assertEqual(saved["revision"], 1)

    def test_extraction_failure_preserves_existing_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digest.sqlite3"
            source = Path(directory) / "profile.txt"
            source.write_text("Unclear interests", encoding="utf-8")
            original = ProfileStore(path).save(profile())
            with (
                patch(
                    "sys.argv",
                    [
                        "create_profile.py",
                        str(source),
                        "--user-id",
                        "alice",
                        "--db",
                        str(path),
                    ],
                ),
                patch(
                    "research_digest.cli.create_profile.build_profile",
                    side_effect=ValueError("Invalid profile"),
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(create_profile.main(), 1)
            self.assertEqual(ProfileStore(path).load("alice"), original)

    def test_requires_input_file_unless_showing_profile(self):
        for extra in ([], ["profile.txt", "--show"]):
            with (
                self.subTest(extra=extra),
                patch("sys.argv", ["create_profile.py", "--user-id", "alice", *extra]),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                create_profile.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
