import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from research_digest.llm.codex_llm import (
    CodexError,
    CodexOpenAI,
    CodexTimeoutError,
    CodexToolUseError,
)


def events(answer="OK"):
    return "\n".join(
        json.dumps(event)
        for event in [
            {"type": "thread.started", "thread_id": "test-thread"},
            {"type": "turn.started"},
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": answer,
                },
            },
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 10,
                    "cached_input_tokens": 2,
                    "output_tokens": 3,
                },
            },
        ]
    )


class CodexClientTests(unittest.TestCase):
    def setUp(self):
        self.client = CodexOpenAI(timeout=2)
        self.messages = [{"role": "user", "content": "Hello"}]
        self.run = patch("research_digest.llm.codex_llm.subprocess.run").start()
        self.addCleanup(patch.stopall)
        self.run.return_value = subprocess.CompletedProcess([], 0, events(), "")

    def test_chat_shape_and_usage(self):
        result = self.client.chat.completions.create(
            messages=self.messages, model="test"
        )
        self.assertEqual(result.choices[0].message.content, "OK")
        self.assertEqual(result.usage.total_tokens, 13)
        self.assertEqual(result.model, "test")

    def test_prompt_stays_on_stdin_and_config_is_isolated(self):
        payload = "$(touch /tmp/never-run); `echo nope`\nUnicode: α"
        self.client.responses.create(input=payload, instructions="Be concise")
        args, kwargs = self.run.call_args
        command = args[0]
        self.assertNotIn(payload, command)
        self.assertEqual(
            json.loads(kwargs["input"].split("\n\n", 1)[1]),
            [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": payload},
            ],
        )
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--ephemeral", command)
        self.assertIn("shell_tool", command)
        self.assertIn('web_search="disabled"', command)
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
        self.assertNotIn("CODEX_API_KEY", kwargs["env"])
        self.assertNotIn("shell", kwargs)
        self.assertFalse(Path(kwargs["cwd"]).exists())

    def test_responses_shape(self):
        result = self.client.responses.create(input="Hello")
        self.assertEqual(result.output_text, "OK")
        self.assertEqual(result.output[0].content[0].text, "OK")
        self.assertEqual(result.usage.input_tokens, 10)

    def test_schema_forwarded_to_cli(self):
        schema = {"type": "object", "properties": {}, "additionalProperties": False}

        def run(command, **kwargs):
            path = command[command.index("--output-schema") + 1]
            self.assertEqual(json.loads(Path(path).read_text()), schema)
            return subprocess.CompletedProcess(command, 0, events("{}"), "")

        self.run.side_effect = run
        result = self.client.responses.create(
            input="Return an object",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": schema,
                }
            },
        )
        self.assertEqual(result.output_text, "{}")

    def test_tool_activity_is_rejected(self):
        for kind in ("command_execution", "mcp_tool_call", "web_search", "file_change"):
            with self.subTest(kind=kind):
                self.run.return_value.stdout = json.dumps(
                    {
                        "type": "item.started",
                        "item": {"type": kind},
                    }
                )
                with self.assertRaises(CodexToolUseError):
                    self.client.responses.create(input="Hello")

    def test_nonfatal_cli_diagnostic_is_surfaced(self):
        self.run.return_value.stdout = (
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "type": "error",
                        "message": "Startup diagnostic",
                    },
                }
            )
            + "\n"
            + events()
        )
        with self.assertWarnsRegex(RuntimeWarning, "Startup diagnostic"):
            result = self.client.responses.create(input="Hello")
        self.assertEqual(result.output_text, "OK")

    def test_failed_or_malformed_output_is_rejected(self):
        for output in (
            "not JSON",
            "[]",
            "{}",
            "",
            '{"type":"turn.completed"}',
            '{"type":"turn.failed","error":{"message":"quota"}}',
            '{"type":"future.event"}',
        ):
            with self.subTest(output=output):
                self.run.return_value.stdout = output
                with self.assertRaises(CodexError):
                    self.client.responses.create(input="Hello")

    def test_timeout_and_missing_cli_are_actionable(self):
        self.run.side_effect = subprocess.TimeoutExpired("codex", 2)
        with self.assertRaises(CodexTimeoutError):
            self.client.responses.create(input="Hello")
        self.run.side_effect = FileNotFoundError("codex")
        with self.assertRaisesRegex(CodexError, "Install Codex CLI"):
            self.client.responses.create(input="Hello")

    def test_cli_nonzero_exit(self):
        self.run.return_value = subprocess.CompletedProcess([], 1, "", "Please log in")
        with self.assertRaisesRegex(CodexError, "Please log in"):
            self.client.responses.create(input="Hello")

    def test_schema_requires_json_output(self):
        with self.assertRaisesRegex(CodexError, "invalid JSON"):
            self.client.chat.completions.create(
                messages=self.messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "object"}},
                },
            )

    def test_unsupported_sdk_options_fail_explicitly(self):
        for kwargs in ({"tools": []}, {"temperature": 0}, {"max_tokens": 10}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                self.client.chat.completions.create(messages=self.messages, **kwargs)
        with self.assertRaises(ValueError):
            self.client.responses.create(input="Hello", stream=True)
        self.run.assert_not_called()

    def test_invalid_messages_fail_before_starting_cli(self):
        for messages in (
            [],
            "Hi",
            [{"role": "tool", "content": "Hi"}],
            [{"role": "user", "content": []}],
            [{"role": "user", "content": "Hi", "tool_calls": []}],
        ):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.client.chat.completions.create(messages=messages)
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
