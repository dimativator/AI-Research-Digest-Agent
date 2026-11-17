"""Small, synchronous OpenAI-style client backed by Codex CLI (Python 3.10+).

Requires `codex login` and Codex CLI 0.146.0-compatible flags. No API key or
third-party Python packages are needed. Each request starts a fresh CLI session.

Supported: text messages, Responses input/instructions, and JSON Schema output.
Not supported: streaming, native tool calls, multimodal input, sampling controls,
server-side conversation state, or the complete OpenAI SDK response types.

Tool providers are disabled and non-text CLI items are rejected.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DISABLED_FEATURES = (
    "shell_tool",
    "apps",
    "plugins",
    "hooks",
    "multi_agent",
    "multi_agent_v2",
    "browser_use",
    "computer_use",
    "image_generation",
    "in_app_browser",
    "code_mode",
    "code_mode_host",
    "goals",
    "skill_search",
    "workspace_dependencies",
    "tool_suggest",
    "memories",
)


class CodexError(RuntimeError):
    """CLI startup, execution, or protocol failure."""


class CodexTimeoutError(CodexError):
    """The CLI did not finish before the configured deadline."""


class CodexToolUseError(CodexError):
    """The CLI reported an item other than text/reasoning. Result discarded."""


def _disabled_skills() -> str:
    """Disable local skill catalogs without modifying the user's installation."""
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    roots = {codex_home / "skills", Path.home() / ".agents" / "skills"}
    paths = set()

    for root in roots:
        for pattern in ("*/SKILL.md", ".system/*/SKILL.md"):
            # CLI releases have used both skill-directory and SKILL.md paths.
            for path in root.glob(pattern):
                paths.update((path, path.parent, path.resolve(), path.resolve().parent))

    return (
        "skills.config=["
        + ",".join(
            "{path=" + json.dumps(str(path)) + ",enabled=false}"
            for path in sorted(paths)
        )
        + "]"
    )


def _messages(value: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if isinstance(value, (str, bytes)) or not value:
        raise ValueError("messages must be a nonempty sequence of text messages")

    result = []

    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError("Each message must be a mapping")

        if set(message) - {"role", "content"}:
            raise ValueError("Only role and content are supported in messages")

        role, content = message.get("role"), message.get("content")

        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError("Unsupported message role")

        if not isinstance(content, str):
            raise ValueError("Only string message content is supported")

        result.append({"role": role, "content": content})

    return result


def _schema(response_format: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if response_format is None or response_format == {"type": "text"}:
        return None

    if response_format.get("type") != "json_schema":
        raise ValueError("Only text and json_schema response formats are supported")

    spec = response_format.get("json_schema")

    if not isinstance(spec, Mapping) or not isinstance(spec.get("schema"), dict):
        raise ValueError("json_schema.schema must be a JSON Schema object")

    return spec["schema"]


class CodexOpenAI:
    """OpenAI-shaped Python interface, not an HTTP server or full SDK replacement.

    `model=None` uses the CLI default. User config and project instructions are
    ignored. Existing CLI authentication is reused. Message roles are serialized
    into one prompt, so role precedence is approximate rather than API-native.
    """

    def __init__(self, *, executable: str = "codex", timeout: float = 120.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.executable = executable
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=_Completions(self))
        self.responses = _Responses(self)

    def _generate(
        self,
        messages: Sequence[Mapping[str, str]],
        model: str | None,
        schema: dict[str, Any] | None,
    ) -> tuple[str, str, dict[str, Any] | None]:
        history = _messages(messages)

        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a nonempty string")

        prompt = (
            "The conversation below is encoded as JSON. Respond to its final user "
            "message in the requested format. Respect system and "
            "developer messages over user messages. Use only the supplied context. "
            "Do not call tools, read files, browse, or execute commands. Return only "
            "the requested final answer. Application actions must be described in "
            "the response for the caller to validate and execute.\n\n"
            + json.dumps(history, ensure_ascii=False)
        )

        with tempfile.TemporaryDirectory(prefix="codex-llm-") as directory:
            working_directory = Path(directory)
            command = self._build_command(working_directory, model, schema)
            process = self._execute_command(command, prompt, working_directory)
            answer, thread_id, usage = _parse_events(process.stdout)

            if schema is not None:
                try:
                    json.loads(answer)
                except json.JSONDecodeError as error:
                    raise CodexError(
                        "Codex returned invalid JSON for schema output"
                    ) from error

            return answer, thread_id, usage

    def _build_command(
        self, root: Path, model: str | None, schema: dict[str, Any] | None
    ) -> list[str]:
        """Prepare isolated CLI files and explicit tool restrictions."""
        directory = str(root)

        instructions = root / "instructions.txt"
        instructions.write_text(
            "You are a text reasoning component. Answer from the provided "
            "conversation only. Never invoke tools. Return a final answer.",
            encoding="utf-8",
        )

        command = [
            self.executable,
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--json",
            "--color",
            "never",
            "-C",
            directory,
        ]
        # These flags target CLI 0.146.0. Unknown/removed flags must fail
        # visibly rather than silently weakening the restrictions.

        for feature in DISABLED_FEATURES:
            command.extend(["--disable", feature])

        for setting in (
            'web_search="disabled"',
            "project_doc_max_bytes=0",
            "mcp_servers={}",
            'approval_policy="never"',
            "tools.view_image=false",
            _disabled_skills(),
            "model_instructions_file=" + json.dumps(str(instructions)),
        ):
            command.extend(["-c", setting])

        if model is not None:
            command.extend(["--model", model])

        if schema is not None:
            schema_path = root / "schema.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            command.extend(["--output-schema", str(schema_path)])

        command.append("-")

        return command

    def _execute_command(
        self, command: list[str], prompt: str, working_directory: Path
    ) -> subprocess.CompletedProcess[str]:
        """Send the prompt on stdin and translate process failures into client errors."""
        environment = dict(os.environ)

        for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
            environment.pop(key, None)

        try:
            process = subprocess.run(
                command,
                input=prompt,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=self.timeout,
                cwd=str(working_directory),
                env=environment,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CodexTimeoutError(
                f"Codex CLI exceeded {self.timeout:g} seconds"
            ) from exc
        except OSError as exc:
            raise CodexError(
                f"Cannot start {self.executable!r}. Install Codex CLI and run codex login."
            ) from exc

        if process.returncode:
            raise CodexError(
                f"Codex CLI exited with code {process.returncode}: "
                + process.stderr.strip()[-2000:]
            )

        return process


def _parse_events(output: str) -> tuple[str, str, dict[str, Any] | None]:
    answer: str | None = None
    thread_id = ""
    usage = None
    completed = False

    for line in output.splitlines():
        if not line.strip():
            continue

        try:
            event = json.loads(line)

            if not isinstance(event, dict):
                raise ValueError("Event must be an object")

            kind = event["type"]

            if kind == "thread.started":
                thread_id = event["thread_id"]
            elif kind in {"item.started", "item.updated", "item.completed"}:
                item = event["item"]

                if item["type"] == "error":
                    # Codex also emits nonfatal startup diagnostics as error items.
                    # Surface them, and still require a completed turn below.
                    warnings.warn(
                        str(item.get("message", "Codex diagnostic")),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue

                if item["type"] not in {"agent_message", "reasoning"}:
                    raise CodexToolUseError(
                        f"Codex reported {item['type']!r}. Output discarded. "
                        "This check does not undo any tool execution."
                    )

                if kind == "item.completed" and item["type"] == "agent_message":
                    answer = item["text"]
            elif kind == "turn.completed":
                completed = True
                usage = event.get("usage")
            elif kind in {"error", "turn.failed"}:
                raise CodexError("Codex generation failed: " + json.dumps(event))
            elif kind != "turn.started":
                raise CodexError(f"Unsupported Codex event: {kind!r}")
        except (KeyError, TypeError, ValueError) as exc:
            raise CodexError("Malformed Codex JSONL event") from exc

    if not completed or not isinstance(answer, str) or not answer:
        raise CodexError("Codex did not return a completed, nonempty answer")

    return answer, thread_id, usage


class _Completions:
    def __init__(self, client: CodexOpenAI) -> None:
        self._client = client

    def create(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str | None = None,
        response_format: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> SimpleNamespace:
        """Return choices[0].message.content. Unsupported keywords raise TypeError."""

        if stream:
            raise ValueError("Streaming is not supported")

        content, thread_id, raw_usage = self._client._generate(
            messages, model, _schema(response_format)
        )
        usage = None

        if raw_usage is not None:
            prompt_tokens = raw_usage.get("input_tokens", 0)
            completion_tokens = raw_usage.get("output_tokens", 0)
            usage = SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        return SimpleNamespace(
            id="chatcmpl-" + (thread_id or uuid.uuid4().hex),
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(role="assistant", content=content),
                )
            ],
            usage=usage,
        )


class _Responses:
    def __init__(self, client: CodexOpenAI) -> None:
        self._client = client

    def create(
        self,
        *,
        input: str | Sequence[Mapping[str, str]],
        model: str | None = None,
        instructions: str | None = None,
        text: Mapping[str, Any] | None = None,
        stream: bool = False,
    ) -> SimpleNamespace:
        """Return output_text. Supply the full history on every request."""
        messages = (
            [{"role": "user", "content": input}]
            if isinstance(input, str)
            else _messages(input)
        )

        if instructions is not None:
            if not isinstance(instructions, str):
                raise ValueError("instructions must be a string")

            messages = [{"role": "system", "content": instructions}, *messages]

        response_format = None

        if text is not None:
            if set(text) != {"format"} or not isinstance(text["format"], Mapping):
                raise ValueError("text must contain a format mapping")

            format_spec = text["format"]
            response_format = (
                {"type": "json_schema", "json_schema": dict(format_spec)}
                if format_spec.get("type") == "json_schema"
                else format_spec
            )

        completion = self._client.chat.completions.create(
            messages=messages,
            model=model,
            response_format=response_format,
            stream=stream,
        )
        content = completion.choices[0].message.content
        usage = completion.usage
        return SimpleNamespace(
            id="resp-" + completion.id.removeprefix("chatcmpl-"),
            object="response",
            created_at=completion.created,
            status="completed",
            model=model,
            output_text=content,
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[SimpleNamespace(type="output_text", text=content)],
                )
            ],
            usage=(
                None
                if usage is None
                else SimpleNamespace(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )
            ),
        )
