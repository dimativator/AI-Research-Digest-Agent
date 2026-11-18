"""Resume the frozen weekly evaluation after arXiv rate limits clear."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from research_digest.retrieval import arxiv_source
arxiv_source.REQUEST_INTERVAL_SECONDS = 10.0
from eval.run_weekly import main
sys.argv = ["eval.run_weekly", "--output", str(Path(__file__).resolve().parent), "--resume"]
main()
