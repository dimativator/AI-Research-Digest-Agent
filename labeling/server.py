"""Local-only annotation server with transactional labels and JSON export."""

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--page", type=Path, default=ROOT / "static/index.html")
    parser.add_argument("--sample", type=Path, default=ROOT / "sample.json")
    parser.add_argument("--db", type=Path, default=ROOT / "labels.sqlite3")
    args = parser.parse_args()
    raw = args.sample.read_bytes()
    sample = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    ids = {p["paper_id"] for p in sample["papers"]}
    if len(ids) != len(sample["papers"]):
        raise ValueError("Duplicate paper IDs")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS labels (sample TEXT, paper_id TEXT, label REAL, note TEXT, updated_at TEXT, PRIMARY KEY(sample,paper_id))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS history (sample TEXT, paper_id TEXT, label REAL, note TEXT, updated_at TEXT)"
        )

    def snapshot():
        with sqlite3.connect(args.db) as conn:
            labels = {
                r[0]: {"label": r[1], "note": r[2], "updated_at": r[3]}
                for r in conn.execute(
                    "SELECT paper_id,label,note,updated_at FROM labels WHERE sample=?",
                    (digest,),
                )
            }
        return {
            **sample,
            "sample_sha256": digest,
            "annotator": "dmitrii",
            "papers": [
                {
                    **p,
                    **labels.get(
                        p["paper_id"], {"label": None, "note": "", "updated_at": None}
                    ),
                }
                for p in sample["papers"]
            ],
        }

    class Handler(BaseHTTPRequestHandler):
        def send(self, data, status=200, kind="application/json", download=False):
            body = (
                data
                if isinstance(data, bytes)
                else json.dumps(data, ensure_ascii=False).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if download:
                self.send_header(
                    "Content-Disposition",
                    'attachment; filename="paper-interest-labels.json"',
                )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/api/sample", "/api/export"):
                self.send(snapshot(), download=self.path == "/api/export")
            elif self.path == "/":
                self.send(
                    args.page.read_bytes(),
                    kind="text/html; charset=utf-8",
                )
            else:
                self.send({"error": "Not found"}, 404)

        def do_POST(self):
            if self.path != "/api/label":
                return self.send({"error": "Not found"}, 404)
            if self.headers.get("Origin") not in (
                None,
                f"http://127.0.0.1:{args.port}",
                f"http://localhost:{args.port}",
            ):
                return self.send({"error": "Invalid origin"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                label = data["label"]
                if label is not None and (
                    type(label) not in (int, float) or label not in (0, 0.5, 1)
                ):
                    raise ValueError("Label must be 0, 0.5, 1, or null")
                if data["paper_id"] not in ids or data.get("sample_sha256") != digest:
                    raise ValueError("Unknown paper or changed sample; reload the page")
                note = data.get("note", "")
                if not isinstance(note, str) or len(note) > 4000:
                    raise ValueError("Note must be text, at most 4000 characters")
                values = (
                    digest,
                    data["paper_id"],
                    label,
                    note,
                    datetime.now(timezone.utc).isoformat(),
                )
                with sqlite3.connect(args.db) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO labels VALUES (?,?,?,?,?)", values
                    )
                    conn.execute("INSERT INTO history VALUES (?,?,?,?,?)", values)
                self.send({"saved": True, "updated_at": values[-1]})
            except (ValueError, KeyError, TypeError) as exc:
                self.send({"error": str(exc)}, 400)
            except sqlite3.Error:
                self.send({"error": "Database write failed; label not saved"}, 500)

    print(f"Label papers at http://127.0.0.1:{args.port}", flush=True)
    print(f"Labels saved to {args.db.resolve()}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
