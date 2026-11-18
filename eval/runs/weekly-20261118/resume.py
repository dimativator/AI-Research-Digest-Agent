"""Resume weekly evaluation with fewer, slower arXiv metadata requests."""
import os
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from research_digest.retrieval import arxiv_source
from eval import run_weekly

arxiv_source.REQUEST_INTERVAL_SECONDS = 10.0
run_weekly.collect_metadata = partial(arxiv_source.collect_metadata, page_size=1000)
print("Transport: page_size=1000, delay=10 seconds, HTTP429 retry minimum=75 seconds", flush=True)
sys.argv = ["eval.run_weekly", "--output", str(Path(__file__).resolve().parent), "--resume"]
run_weekly.main()
