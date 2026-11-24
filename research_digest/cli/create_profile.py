"""Structure a user's research-profile text file with an LLM and save to SQLite.

    python create_profile.py profile.txt --user-id alice
    python create_profile.py --user-id alice --show

The UTF-8 file contains free-form interests, goals, and exclusions. Validated
profiles are saved automatically. Each changed profile creates a new revision.
URLs mentioned in the text are not fetched.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from research_digest.llm.codex_llm import CodexError
from research_digest.llm.openai_api import create_client
from research_digest.profiles.user_profile import ProfileStore, build_profile


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a research-interest profile with Codex."
    )
    parser.add_argument(
        "profile_file", nargs="?", type=Path, help="UTF-8 research-profile text file."
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--llm-backend", choices=("codex", "openai"), default="codex")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "digest.sqlite3",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the latest saved profile, no LLM call.",
    )
    args = parser.parse_args()

    if args.show and args.profile_file is not None:
        parser.error("Pass either a profile file or --show, not both")

    if not args.show and args.profile_file is None:
        parser.error("A research-profile text file is required")

    try:
        if args.show:
            saved = ProfileStore(args.db).load(args.user_id)
            print(
                json.dumps(
                    {
                        "revision": saved.revision,
                        "created_at": saved.created_at,
                        "fingerprint": saved.fingerprint,
                        "profile": saved.profile.to_dict(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        description_text = args.profile_file.read_text(encoding="utf-8")
        print("Building your profile with the LLM...", file=sys.stderr, flush=True)
        profile = build_profile(
            args.user_id,
            description_text,
            model=args.model,
            client=create_client(args.llm_backend, args.api_key_file),
        )
        saved = ProfileStore(args.db).save(profile)
        print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), flush=True)
        print(
            f"Saved {profile.user_id!r}, revision {saved.revision}, to {args.db}",
            file=sys.stderr,
        )
        return 0
    except (CodexError, OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelled. No new profile saved.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
