"""Add explicit provenance to completed weekly JSONL exports, preserving content."""

import hashlib
import json
from pathlib import Path

ROOT = Path("eval/runs/weekly-20261118")
audit = []
for week in sorted(ROOT.glob("week-*")):
    for variant in ("react", "react-explore", "debate"):
        directory = week / variant
        assert (
            json.loads((directory / "outcome.json").read_text())["status"]
            == "completed"
        )
        for path in directory.rglob("*.jsonl"):
            original = path.read_bytes()
            rows = [json.loads(line) for line in original.splitlines()]
            updated = [{**row, "variant": variant, "task": week.name} for row in rows]
            for before, after in zip(rows, updated):
                assert all(after[k] == v for k, v in before.items())
            content = "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in updated
            ).encode()
            path.write_bytes(content)
            audit.append(
                dict(
                    path=str(path.relative_to(ROOT)),
                    records=len(rows),
                    original_sha256=hashlib.sha256(original).hexdigest(),
                    annotated_sha256=hashlib.sha256(content).hexdigest(),
                )
            )
(ROOT / "trace-annotation-audit.json").write_text(
    json.dumps(
        dict(
            operation="Add variant and task fields to JSONL exports only; original payloads unchanged. SQLite originals and source snapshots retained.",
            files=audit,
        ),
        indent=2,
    )
    + "\n"
)
print(f'Annotated {len(audit)} files, {sum(x["records"] for x in audit)} records.')
