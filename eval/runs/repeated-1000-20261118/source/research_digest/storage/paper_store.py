"""Metadata cache, immutable run candidates, screening cache, and agent trace."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from research_digest.retrieval.arxiv_source import Page, Paper
from research_digest.retrieval.ranking import RankedPaper


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT NOT NULL, version INTEGER NOT NULL,
    fingerprint TEXT NOT NULL, metadata_json TEXT NOT NULL,
    PRIMARY KEY (paper_id, version)
);
CREATE TABLE IF NOT EXISTS metadata_pages (
    query TEXT NOT NULL, offset INTEGER NOT NULL, size INTEGER NOT NULL,
    fetched_at TEXT NOT NULL, page_json TEXT NOT NULL,
    PRIMARY KEY (query, offset, size)
);
CREATE TABLE IF NOT EXISTS digest_runs (
    run_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    status TEXT NOT NULL, state_json TEXT NOT NULL, result_json TEXT
);
CREATE TABLE IF NOT EXISTS run_candidates (
    run_id TEXT NOT NULL, rank INTEGER NOT NULL, paper_id TEXT NOT NULL,
    score REAL NOT NULL, paper_json TEXT NOT NULL,
    PRIMARY KEY (run_id, rank), UNIQUE (run_id, paper_id)
);
CREATE TABLE IF NOT EXISTS screenings (
    cache_key TEXT PRIMARY KEY, assessment_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inspections (
    fingerprint TEXT PRIMARY KEY, observation_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS x_attention (
    cache_key TEXT PRIMARY KEY, observation_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL, created_at TEXT NOT NULL,
    event_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recommendations (
    user_id TEXT NOT NULL, paper_id TEXT NOT NULL, run_id TEXT NOT NULL,
    PRIMARY KEY (user_id, paper_id)
);
"""


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def cached_page(self, query: str, offset: int, size: int) -> Page | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT page_json FROM metadata_pages WHERE query=? AND offset=? AND size=?",
                (query, offset, size),
            ).fetchone()

        if row is None:
            return None

        (page_json,) = row
        data = json.loads(page_json)
        return Page(
            data["total"], data["start"], [Paper(**item) for item in data["papers"]]
        )

    def save_page(self, query: str, offset: int, size: int, page: Page) -> None:
        with self.connect() as connection:
            for paper in page.papers:
                connection.execute(
                    """
                    INSERT INTO papers (paper_id, version, fingerprint, metadata_json) VALUES
                    (?, ?, ?, ?) ON CONFLICT(paper_id, version) DO UPDATE SET
                    fingerprint=excluded.fingerprint, metadata_json=excluded.metadata_json
                    """,
                    (
                        paper.paper_id,
                        paper.version,
                        paper.fingerprint,
                        json.dumps(asdict(paper)),
                    ),
                )

            connection.execute(
                """
                INSERT OR REPLACE INTO metadata_pages (query, offset, size, fetched_at,
                page_json) VALUES (?, ?, ?, ?, ?)
                """,
                (query, offset, size, timestamp(), json.dumps(asdict(page))),
            )

    def create_run(self, user_id: str, state: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex

        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO digest_runs (run_id, user_id, created_at, updated_at, status,
                state_json, result_json) VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    run_id,
                    user_id,
                    timestamp(),
                    timestamp(),
                    "running",
                    json.dumps(state),
                ),
            )

        return run_id

    def load_run(
        self, run_id: str
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status, state_json, result_json FROM digest_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()

        if row is None:
            raise ValueError(f"Unknown run ID: {run_id}")

        return (
            row[0],
            json.loads(row[1]),
            None if row[2] is None else json.loads(row[2]),
        )

    def update_run(
        self,
        run_id: str,
        state: dict[str, Any],
        *,
        status: str = "running",
        event: dict[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE digest_runs SET updated_at=?, status=?, state_json=? WHERE run_id=?",
                (timestamp(), status, json.dumps(state), run_id),
            )

            if event is not None:
                connection.execute(
                    "INSERT INTO agent_events(run_id, created_at, event_json) VALUES (?, ?, ?)",
                    (run_id, timestamp(), json.dumps(event)),
                )

    def freeze_candidates(self, run_id: str, ranked: list[RankedPaper]) -> None:
        with self.connect() as connection:
            # A failed transaction never leaves a half-built ranking snapshot.
            connection.execute("DELETE FROM run_candidates WHERE run_id=?", (run_id,))
            connection.executemany(
                """
                INSERT INTO run_candidates (run_id, rank, paper_id, score, paper_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        index,
                        item.paper.paper_id,
                        item.score,
                        json.dumps(asdict(item.paper)),
                    )
                    for index, item in enumerate(ranked)
                ],
            )

    def candidates(
        self, run_id: str, offset: int = 0, limit: int = 1_000_000
    ) -> list[RankedPaper]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT paper_json, score FROM run_candidates WHERE run_id=? ORDER BY rank
                LIMIT ? OFFSET ?
                """,
                (run_id, limit, offset),
            ).fetchall()

        return [
            RankedPaper(Paper(**json.loads(paper_json)), score)
            for paper_json, score in rows
        ]

    def recommended(self, user_id: str) -> set[str]:
        with self.connect() as connection:
            return {
                row[0]
                for row in connection.execute(
                    "SELECT paper_id FROM recommendations WHERE user_id=?", (user_id,)
                )
            }

    def cached_screening(self, key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT assessment_json FROM screenings WHERE cache_key=?", (key,)
            ).fetchone()

        return None if row is None else json.loads(row[0])

    def save_screening(self, key: str, assessment: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO screenings (cache_key, assessment_json) VALUES (?, ?)",
                (key, json.dumps(assessment)),
            )

    def cached_inspection(self, key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT observation_json FROM inspections WHERE fingerprint=?", (key,)
            ).fetchone()

        return None if row is None else json.loads(row[0])

    def save_inspection(self, key: str, observation: dict[str, Any]) -> None:
        if observation["status"] == "ok":
            with self.connect() as connection:
                connection.execute(
                    "INSERT OR REPLACE INTO inspections (fingerprint, observation_json) VALUES (?, ?)",
                    (key, json.dumps(observation)),
                )

    def finish(
        self, run_id: str, state: dict[str, Any], result: dict[str, Any]
    ) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            if result["status"] == "completed" and state.get("config", {}).get(
                "new_only", False
            ):
                user_id = state["profile"]["user_id"]
                previous = {
                    row[0]
                    for row in connection.execute(
                        "SELECT paper_id FROM recommendations WHERE user_id=? AND run_id!=?",
                        (user_id, run_id),
                    )
                }
                duplicates = previous.intersection(
                    item["paper_id"] for item in result["papers"]
                )

                if duplicates:
                    raise ValueError(
                        "Another run already recommended these papers: "
                        + ", ".join(sorted(duplicates))
                    )

            connection.execute(
                """
                UPDATE digest_runs SET updated_at=?, status=?, state_json=?, result_json=?
                WHERE run_id=?
                """,
                (
                    timestamp(),
                    result["status"],
                    json.dumps(state),
                    json.dumps(result),
                    run_id,
                ),
            )

            if result["status"] == "completed":
                connection.executemany(
                    "INSERT OR IGNORE INTO recommendations (user_id, paper_id, run_id) VALUES (?, ?, ?)",
                    [
                        (state["profile"]["user_id"], item["paper_id"], run_id)
                        for item in result["papers"]
                    ],
                )

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT event_json FROM agent_events WHERE run_id=? ORDER BY event_id",
                (run_id,),
            ).fetchall()

        return [json.loads(event_json) for (event_json,) in rows]

    def cached_x_attention(self, cache_key: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT observation_json FROM x_attention WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        observation = json.loads(row[0])
        age = datetime.now(timezone.utc) - datetime.fromisoformat(
            observation["collected_at"]
        )
        return observation if 0 <= age.total_seconds() < 21_600 else None

    def save_x_attention(self, cache_key: str, observation: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO x_attention (cache_key, observation_json) VALUES (?, ?)",
                (cache_key, json.dumps(observation)),
            )
