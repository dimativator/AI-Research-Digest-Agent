"""Export the failure window from an existing SQLite run as a JSON artifact."""

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/digest.sqlite3"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/failure_traces/01_live_evidence_rejection.json"),
    )
    args = parser.parse_args()

    with sqlite3.connect(args.db) as connection:
        run = connection.execute(
            "SELECT state_json, result_json FROM digest_runs WHERE run_id=?",
            (args.run_id,),
        ).fetchone()
        events = connection.execute(
            "SELECT event_id, event_json FROM agent_events "
            "WHERE run_id=? ORDER BY event_id",
            (args.run_id,),
        ).fetchall()
    if run is None:
        parser.error(f"Unknown run ID: {args.run_id}")

    decoded = [(event_id, json.loads(event)) for event_id, event in events]
    failures = [
        event_id
        for event_id, event in decoded
        if event.get("type") == "invalid_action"
        and "Final evidence" in event.get("error", "")
    ]
    if not failures:
        parser.error("Run contains no rejected final-evidence decisions")
    state = json.loads(run[0])
    artifact = {
        "source": "observed live Codex CLI run",
        "run_id": args.run_id,
        "failure_event_ids": failures,
        "profile_summary": state["profile"]["summary"],
        "config": state["config"],
        "result": json.loads(run[1]),
        "inspection_observations": {
            key: value
            for key, value in state.get("observations", {}).items()
            if key.startswith("inspect:")
        },
        "trace_events": [
            {"event_id": event_id, "event": event}
            for event_id, event in decoded
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(f"Exported {len(artifact['trace_events'])} events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())