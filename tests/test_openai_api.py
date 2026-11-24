"""Offline contract checks for the API adapter."""

import io
import json
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from research_digest.llm.openai_api import OpenAIAPI, OpenAIAPIError, create_client
from research_digest.agent.config import DigestConfig


class OpenAIAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "key.txt"
        self.path.write_text("test-secret\n")
        self.client = OpenAIAPI(api_key_file=self.path)

    def response(self, **message):
        return io.BytesIO(
            json.dumps(
                {
                    "id": "test-id",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"ok": true}', **message},
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5},
                }
            ).encode()
        )

    @patch("research_digest.llm.openai_api.urlopen")
    def test_response_schema_usage_and_no_tools(self, send):
        send.return_value = self.response()
        result = self.client.responses.create(
            model="api-model",
            instructions="Be helpful",
            input="hello",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "test",
                    "schema": {"type": "object"},
                }
            },
        )
        self.assertEqual(result.output_text, '{"ok": true}')
        self.assertEqual(result.usage.input_tokens, 20)
        self.assertEqual(result.usage.output_tokens, 5)
        payload = json.loads(send.call_args.args[0].data)
        self.assertNotIn("tools", payload)
        self.assertFalse(payload["store"])
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertNotIn("test-secret", json.dumps(payload))

    @patch("research_digest.llm.openai_api.time.sleep")
    @patch("research_digest.llm.openai_api.urlopen")
    def test_rate_limit_retry(self, send, sleep):
        send.side_effect = [
            HTTPError("url", 429, "limited", {"Retry-After": "3"}, None),
            self.response(),
        ]
        self.client.responses.create(model="api-model", input="hello")
        sleep.assert_called_once_with(3.0)
        self.assertEqual(send.call_count, 2)

    @patch("research_digest.llm.openai_api.urlopen")
    def test_error_does_not_expose_key(self, send):
        send.side_effect = HTTPError(
            "url", 401, "test-secret", {}, io.BytesIO(b"test-secret")
        )
        with self.assertRaisesRegex(OpenAIAPIError, "^OpenAI API returned HTTP 401$"):
            self.client.responses.create(model="api-model", input="hello")
        self.assertEqual(send.call_count, 1)

    @patch("research_digest.llm.openai_api.urlopen")
    def test_refusal_and_native_tool_rejected(self, send):
        for message in ({"refusal": "no"}, {"tool_calls": [{"id": "tool"}]}):
            send.return_value = self.response(**message)
            with self.assertRaises(OpenAIAPIError):
                self.client.responses.create(model="api-model", input="hello")

    def test_resume_accepts_key_file_without_overriding_backend(self):
        from research_digest.cli.run_digest import _parse_arguments

        with patch(
            "sys.argv",
            ["run_digest.py", "--resume", "run-id", "--api-key-file", str(self.path)],
        ):
            args = _parse_arguments()
        self.assertEqual(args.resume, "run-id")
        self.assertEqual(args.api_key_file, self.path)

    def test_backend_separates_screening_cache(self):
        from research_digest.agent.digest_agent import DigestAgent

        agent = DigestAgent(Mock())
        agent.profile = SimpleNamespace(fingerprint="profile")
        agent.config = DigestConfig(
            categories=("cs.LG",),
            date_from="2026-01-01",
            date_to="2026-01-02",
            model="same-model",
        )
        paper = SimpleNamespace(fingerprint="paper")
        codex_key = agent._screening_cache_key(paper)
        agent.config = replace(agent.config, llm_backend="openai")
        self.assertNotEqual(codex_key, agent._screening_cache_key(paper))

    def test_configuration(self):
        with self.assertRaises(ValueError):
            create_client("openai")
        with self.assertRaises(ValueError):
            self.client.responses.create(input="hello")
        config = DigestConfig(
            categories=("cs.LG",), date_from="2026-01-01", date_to="2026-01-02"
        )
        self.assertEqual(config.llm_backend, "codex")
        config.validate()


if __name__ == "__main__":
    unittest.main()
