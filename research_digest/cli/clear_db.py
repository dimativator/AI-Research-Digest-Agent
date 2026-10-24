"""Clear paper and run data for fresh testing, preserving researcher profiles."""

import argparse
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


PAPER_TABLES = (
    "recommendations",
    "run_candidates",
    "agent_events",
    "digest_runs",
    "screenings",
    "inspections",
    "x_attention",
    "metadata_pages",
    "papers",
)


def clear_database(path: Path) -> tuple[Path, dict[str, int]]:
    """Back up and atomically delete paper data from an existing database."""
    path = path.resolve(strict=True)
    backup_directory = path.parent / "backups"
    backup_directory.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = backup_directory / f"{path.stem}-before-reset-{timestamp}.sqlite3"
    uri = path.as_uri() + "?mode=rw"

    with closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        try:
            # Block other writers until the backup and reset are complete.
            connection.execute("BEGIN IMMEDIATE")
            counts = {
                table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[
                    0
                ]
                for table in PAPER_TABLES
            }
            with (
                closing(sqlite3.connect(uri, uri=True)) as source,
                closing(sqlite3.connect(backup)) as destination,
            ):
                source.backup(destination)

            for table in PAPER_TABLES:
                connection.execute(f'DELETE FROM "{table}"')
            connection.execute("DELETE FROM sqlite_sequence WHERE name='agent_events'")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    return backup, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Back up the database and clear papers, runs, traces, recommendations, "
            "and caches. Preserve researcher profiles. Stop digest runs before using this."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "digest.sqlite3",
    )
    args = parser.parse_args()

    try:
        backup, counts = clear_database(args.db)
    except (OSError, sqlite3.Error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Cleared {counts['papers']} papers and associated run data and caches.")
    print("Researcher profiles preserved.")
    print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
