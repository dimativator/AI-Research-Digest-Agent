"""Try the adapter: python3 try_codex.py "Your prompt".

Run `codex login` first if you have not authenticated the CLI.
"""

import argparse
import sys
import time

from research_digest.llm.codex_llm import CodexError, CodexOpenAI


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the Codex CLI LLM adapter.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain what an AI research digest agent does in one sentence.",
    )
    parser.add_argument(
        "--model", default=None, help="Defaults to the Codex CLI model."
    )
    args = parser.parse_args()

    client = CodexOpenAI(timeout=120)
    print("Waiting for Codex...", file=sys.stderr, flush=True)
    started = time.monotonic()

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": "Answer briefly using only the supplied context.",
                },
                {"role": "user", "content": args.prompt},
            ],
        )
    except CodexError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(response.choices[0].message.content)
    print(f"Completed in {time.monotonic() - started:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
