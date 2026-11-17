"""Fetch -> store -> BM25 -> batch screening -> bounded ReAct tool loop.

    python run_digest.py --user-id dmitrii \
        --date-from 2026-10-01 --date-to 2026-10-07
    python run_digest.py --resume RUN_ID
    python run_digest.py --trace RUN_ID

Dates are inclusive UTC first-submission dates, including category cross-lists.
Metadata is cached; use --refresh-metadata to requery delayed/updated API records.
Previously recommended papers remain eligible; --new excludes them.
Inspect uses arXiv HTML only and reports unavailable content explicitly.
X attention uses your active Arc profile (python setup_x.py). This command never fetches profile URLs.
A saved user profile and an authenticated Codex CLI are required.
Categories default to the saved profile; missing categories are inferred and saved.
Set any maximum limit or --digest-size to -1 for unlimited. Batch, page,
shortlist, and expansion sizes must remain positive.
"""

import argparse
import json
import sqlite3
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from research_digest.retrieval.arxiv_source import RetrievalError
from research_digest.cli.digest_output import format_digest
from research_digest.llm.codex_llm import CodexError
from research_digest.agent.digest_agent import DigestAgent, DigestConfig
from research_digest.agent.config import UNLIMITED_LIMIT_FIELDS
from research_digest.storage.paper_store import PaperStore
from research_digest.profiles.user_profile import (
    ProfileStore,
    UserProfile,
    infer_arxiv_categories,
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI research digest pipeline.")
    parser.add_argument("--user-id")
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Optional run-only override; defaults to categories in the saved profile.",
    )
    parser.add_argument(
        "--date-from", help="Inclusive first-submission date, YYYY-MM-DD (UTC)"
    )
    parser.add_argument(
        "--date-to", help="Inclusive first-submission date, YYYY-MM-DD (UTC)"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--variant",
        choices=("react", "react-explore", "selection-critic"),
        default="react",
        help="react-explore is Agent 2 with overview and similarity tools. Selection-critic uses abstracts only and disables X analysis.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "digest.sqlite3",
    )
    parser.add_argument(
        "--output", type=Path, help="Also write the resulting digest as JSON"
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Final terminal output format (default: readable text). Trace stays JSON.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="Resume failed/interrupted run using its saved configuration",
    )
    mode.add_argument(
        "--trace",
        metavar="RUN_ID",
        help="Print saved decisions and tool observations, no LLM calls",
    )
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument(
        "--no-X",
        dest="x_enabled",
        action="store_false",
        help="Disable X analysis and the X featured paper for this run.",
    )
    parser.add_argument(
        "--new",
        dest="new_only",
        action="store_true",
        help="Exclude papers previously recommended to this user (default: allow repeats).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print trace events to stderr as they happen, including on resume.",
    )

    for name, default in (
        ("shortlist-size", 30),
        ("batch-size", 10),
        ("expansion-size", 10),
        ("max-candidates", -1),
        ("digest-size", 5),
        ("max-steps", -1),
        ("max-llm-calls", -1),
        ("max-inspections", -1),
        ("max-expansions", -1),
        ("page-size", 100),
        ("max-context-chars", -1),
    ):
        field_name = name.replace("-", "_")
        help_text = (
            "Use -1 for unlimited."
            if field_name in UNLIMITED_LIMIT_FIELDS
            else "Must be positive."
        )
        if name == "digest-size":
            help_text = "Exact number of papers to select. Use -1 for unlimited."
        parser.add_argument("--" + name, type=int, default=default, help=help_text)

    args = parser.parse_args()

    if not (args.resume or args.trace) and not all(
        (args.user_id, args.date_from, args.date_to)
    ):
        parser.error("New runs require --user-id, --date-from, and --date-to")

    if args.resume or args.trace:
        allowed = {"--resume", "--trace", "--db", "--output", "--verbose", "--format"}

        if any(
            arg.startswith("--") and arg.split("=", 1)[0] not in allowed
            for arg in sys.argv[1:]
        ):
            parser.error(
                "Resume/trace uses the saved configuration; only --db, --output, --verbose, and --format may also be supplied"
            )

    return args


def _build_config(
    args: argparse.Namespace, profile: UserProfile | None = None
) -> DigestConfig:
    config = DigestConfig(
        **{
            field.name: getattr(args, field.name)
            for field in fields(DigestConfig)
            if field.name != "categories"
        },
        categories=tuple(args.category)
        or (profile.arxiv_categories if profile is not None else ()),
    )
    config.validate()
    return config


def _load_profile(args: argparse.Namespace) -> UserProfile:
    store = ProfileStore(args.db)
    profile = store.load(args.user_id).profile

    if not profile.arxiv_categories and not args.category:
        print(
            "Inferring your arXiv categories with Codex...", file=sys.stderr, flush=True
        )
        profile = replace(
            profile,
            arxiv_categories=infer_arxiv_categories(profile, model=args.model),
        )
        store.save(profile)

    return profile


def _print_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)


def main() -> int:
    args = _parse_arguments()
    run_id = args.resume

    try:
        if args.trace:
            store = PaperStore(args.db)
            store.load_run(args.trace)
            print(json.dumps(store.events(args.trace), ensure_ascii=False, indent=2))
            return 0

        if args.resume:
            store = PaperStore(args.db)
            agent = DigestAgent(store, on_event=_print_event if args.verbose else None)
        else:
            profile = _load_profile(args)
            config = _build_config(args, profile)
            store = PaperStore(args.db)
            agent = DigestAgent(store, on_event=_print_event if args.verbose else None)
            run_id = agent.start(profile, config)

        print(
            f"Run ID: {run_id}. Progress and trace are saved in {args.db}\n"
            f"Readable log: {store.path.parent / 'traces' / run_id / 'activity.log'}",
            file=sys.stderr,
            flush=True,
        )
        result = agent.run(run_id)
        output = json.dumps(result, ensure_ascii=False, indent=2)
        print(format_digest(result) if args.format == "text" else output)

        if args.output is not None:
            args.output.write_text(output + "\n", encoding="utf-8")

        return 0 if result["status"] == "completed" else 2
    except (
        RetrievalError,
        CodexError,
        ValueError,
        KeyError,
        OSError,
        sqlite3.Error,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)

        if run_id:
            print(
                f"Resume with: python run_digest.py --db {str(args.db)!r} --resume {run_id}",
                file=sys.stderr,
            )

        return 1
    except KeyboardInterrupt:
        print(f"Interrupted. Resume run {run_id} with --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
