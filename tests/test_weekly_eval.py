import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from eval.run_weekly import SYSTEMS, export_pool, save, weekly_windows


class WeeklyEvaluationTests(unittest.TestCase):
    def test_calendar_windows(self):
        windows = weekly_windows(date(2026, 11, 18))
        self.assertEqual(windows[0]["date_from"], "2026-09-28")
        self.assertEqual(windows[-1]["date_to"], "2026-11-15")
        days = []
        for w in windows:
            start = date.fromisoformat(w["date_from"])
            end = date.fromisoformat(w["date_to"])
            self.assertEqual((end - start).days, 6)
            self.assertEqual(start.weekday(), 0)
            days.extend(range(start.toordinal(), end.toordinal() + 1))
        self.assertEqual(len(set(days)), 49)

    def test_blind_pool_deduplicates_hides_provenance_and_keeps_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "week-1"
            directory.mkdir()
            paper = {
                "paper_id": "test-id",
                "title": "A paper",
                "abstract": "Text",
                "published": "2026-10-01",
            }
            save(directory / "corpus.json", [paper])
            for system in SYSTEMS:
                target = directory / f"{system}.json"
                if system not in ("bm25", "e5"):
                    (directory / system).mkdir()
                    target = directory / system / "outcome.json"
                save(
                    target,
                    {
                        "status": "failed" if system == "debate" else "completed",
                        "selected": [] if system == "debate" else ["test-id"],
                    },
                )
            manifest = {"windows": [{"task": "week-1"}], "shuffle_seed": 12}
            first = export_pool(root, manifest)
            self.assertEqual(first["papers"], [paper])
            self.assertNotIn("selections", first)
            self.assertNotIn("label", first["papers"][0])
            self.assertEqual(export_pool(root, manifest), first)
            private = json.loads((root / "private-selection-map.json").read_text())
            self.assertEqual(private["selections"][-1]["status"], "failed")
