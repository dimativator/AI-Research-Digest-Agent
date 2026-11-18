"""Estimate baseline latency using BM25 replay and original E5 timings."""
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parents[2]))
from research_digest.profiles.user_profile import UserProfile
from research_digest.retrieval.arxiv_source import Paper
from research_digest.retrieval.ranking import bm25

profile = UserProfile.from_dict(json.loads((ROOT / 'profile.json').read_text()))
query = profile.ranking_text()
rows = []
for directory in sorted(ROOT.glob('week-*')):
    papers = [Paper(**p) for p in json.loads((directory / 'corpus.json').read_text())]
    elapsed = []
    for _ in range(3):
        started = perf_counter()
        selected = [p.paper.paper_id for p in bm25(papers, query)[:7]]
        elapsed.append(perf_counter() - started)
    assert selected == json.loads((directory / 'bm25.json').read_text())['selected']
    e5 = json.loads((directory / 'e5-details.json').read_text())['latency_seconds']
    rows.append(dict(week=directory.name, papers=len(papers), bm25_replay_seconds=elapsed,
                     bm25_mean_seconds=statistics.mean(elapsed), original_e5_seconds=e5,
                     estimated_e5_pipeline_seconds=statistics.mean(elapsed)+e5))
result = dict(method='Three BM25 ranking replays per frozen week, including tokenization and top-7 selection. E5 pipeline estimate adds original rerank latency, including loading, to mean replayed BM25 time. Excludes corpus loading, fetching, persistence and process startup.',
              python=sys.version, platform=platform.platform(), processor=platform.machine(),
              weeks=rows,
              bm25_mean_seconds=statistics.mean(r['bm25_mean_seconds'] for r in rows),
              original_e5_mean_seconds=statistics.mean(r['original_e5_seconds'] for r in rows),
              estimated_e5_pipeline_mean_seconds=statistics.mean(r['estimated_e5_pipeline_seconds'] for r in rows))
(HERE / 'baseline-latency.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
