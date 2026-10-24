"""Run from the project root: python3 -m examples.codex_thinking.

Prerequisite: codex login. No OpenAI Python SDK or API key is needed.
This is an OpenAI-style subset, not a drop-in replacement for every SDK option.
"""

from research_digest.llm.codex_llm import CodexOpenAI


def main() -> None:
    client = CodexOpenAI(timeout=120)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You assess AI papers using supplied evidence.",
            },
            {
                "role": "user",
                "content": "What information do you need to rank a paper?",
            },
        ],
    )
    print(response.choices[0].message.content)

    response = client.responses.create(
        instructions="Choose the next action for a research digest harness.",
        input="We have no candidate papers yet. The user studies LLM evaluation.",
        text={
            "format": {
                "type": "json_schema",
                "name": "next_action",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["search_papers", "finish"],
                        },
                        "query": {"type": "string"},
                    },
                    "required": ["action", "query"],
                },
            }
        },
    )
    # Application code must validate the action and enforce its own budgets.
    print(response.output_text)


if __name__ == "__main__":
    main()
