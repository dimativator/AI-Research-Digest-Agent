"""Text-only OpenAI HTTP backend with the same interface as the CLI adapter.

Protocol: https://platform.openai.com/docs/api-reference/chat/create
No native tools are offered. Application actions remain ordinary JSON output.
"""

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .codex_llm import CodexError, CodexOpenAI, _messages


class OpenAIAPIError(CodexError):
    """API transport or response failure, handled by the existing harness."""


class OpenAIAPI(CodexOpenAI):
    """Reuse response adapters while replacing CLI generation with HTTPS."""

    def __init__(self, *, api_key_file: Path, timeout: float = 120.0) -> None:
        super().__init__(timeout=timeout)
        key = Path(api_key_file).expanduser().read_text(encoding="utf-8").strip()
        if not key or any(char.isspace() for char in key):
            raise ValueError("API key file must contain only a single nonempty key")
        self._api_key = key

    def _generate(
        self,
        messages: Sequence[Mapping[str, str]],
        model: str | None,
        schema: dict[str, Any] | None,
    ) -> tuple[str, str, dict[str, Any] | None]:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAI backend requires an explicit --model")
        payload = {"model": model, "messages": _messages(messages), "store": False}
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_response",
                    "strict": True,
                    "schema": schema,
                },
            }
        request = Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                break
            except HTTPError as error:
                status = error.code
                retry_after = error.headers.get("Retry-After", "")
                error.close()
                if (status == 429 or status >= 500) and attempt < 2:
                    try:
                        delay = max(0.0, float(retry_after))
                    except ValueError:
                        delay = 2**attempt
                    time.sleep(min(delay, 75.0))
                    continue
                # Never log response bodies, headers, or credentials.
                raise OpenAIAPIError(f"OpenAI API returned HTTP {status}") from None
            except (URLError, TimeoutError, OSError):
                raise OpenAIAPIError(
                    "OpenAI API connection failed or timed out"
                ) from None
            except (ValueError, UnicodeError):
                raise OpenAIAPIError("OpenAI API returned invalid JSON") from None
        try:
            choice = result["choices"][0]
            message = choice["message"]
            if message.get("tool_calls") or message.get("function_call"):
                raise OpenAIAPIError("Unexpected native tool call in API response")
            if message.get("refusal"):
                raise OpenAIAPIError("OpenAI API refused the request")
            content = message.get("content")
            if (
                choice["finish_reason"] != "stop"
                or not isinstance(content, str)
                or not content
            ):
                raise OpenAIAPIError(
                    "OpenAI API did not return a complete text response"
                )
            if schema is not None:
                json.loads(content)
            usage = result.get("usage")
            normalized = (
                None
                if usage is None
                else {
                    "input_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                }
            )
            return content, result["id"], normalized
        except (KeyError, IndexError, TypeError, ValueError):
            raise OpenAIAPIError("Malformed OpenAI API response") from None


def create_client(
    backend: str = "codex", api_key_file: Path | None = None
) -> CodexOpenAI:
    if backend == "openai":
        if api_key_file is None:
            raise ValueError("OpenAI backend requires --api-key-file")
        return OpenAIAPI(api_key_file=api_key_file)
    if backend != "codex":
        raise ValueError("Unknown LLM backend")
    if api_key_file is not None:
        raise ValueError("--api-key-file requires --llm-backend openai")
    return CodexOpenAI()
