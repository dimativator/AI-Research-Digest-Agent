"""Freeze a stratified annotation sample from cached arXiv metadata and liked IDs.

python -m labeling.build_sample --liked labeling/liked_ids.json
The liked file contains a JSON list of arXiv IDs observed on AlphaXiv.
Existing samples are never overwritten.
"""

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
from urllib.parse import urlencode
from research_digest.retrieval.arxiv_source import ArxivClient, parse_feed

ROOT = Path(__file__).resolve().parent
TOPICS = {
    "optimization": r"\b(optimizer|optimization|gradient|muon|hessian)\b",
    "low_rank_peft": r"\b(lora|low.rank|fine.tuning|parameter.efficient)\b",
    "training_efficiency": r"\b(pre.training|pretraining|training efficiency|memory.efficient|distributed training)\b",
    "compression_inference": r"\b(quantization|compression|pruning|kv.cache|decoding)\b",
    "theory_generalization": r"\b(generalization|memorization|convergence|spectral|label noise)\b",
    "agents_retrieval": r"\b(agent|agents|retrieval|rag|tool.use)\b",
    "alignment_reasoning": r"\b(reasoning|reinforcement|reward|alignment)\b",
    "vision_multimodal": r"\b(vision|multimodal|diffusion|image generation)\b",
    "continual_federated": r"\b(continual|federated|domain adaptation|transfer learning)\b",
    "data_evaluation": r"\b(benchmark|evaluation|data selection|synthetic data|dataset)\b",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT.parent / "data/digest.sqlite3")
    parser.add_argument("--liked", type=Path, required=True)
    parser.add_argument("--liked-metadata", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "sample.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Sample already exists; choose another output to preserve labels")
    liked = list(dict.fromkeys(json.loads(args.liked.read_text())))[:20]
    if not liked or any(not re.fullmatch(r"\d{4}\.\d{4,5}", pid) for pid in liked):
        parser.error("Supply verified liked arXiv IDs")
    with sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        papers = {
            p["paper_id"]: p
            for (raw,) in conn.execute(
                "SELECT metadata_json FROM papers ORDER BY version"
            )
            if (p := json.loads(raw))
        }
    if args.liked_metadata:
        papers.update(
            {
                p["paper_id"]: p
                for p in json.loads(args.liked_metadata.read_text())
                if p["paper_id"] in liked
            }
        )
    missing = [pid for pid in liked if pid not in papers]
    if missing:
        client = ArxivClient()
        page = parse_feed(
            client._get(
                "https://export.arxiv.org/api/query?"
                + urlencode({"id_list": ",".join(missing), "max_results": len(missing)})
            )
        )
        papers.update({p.paper_id: asdict(p) for p in page.papers})
    if set(liked) - papers.keys():
        raise ValueError("Could not retrieve every liked paper")
    selected = [
        {**papers[pid], "selection_source": "alphaxiv_liked", "selection_topic": None}
        for pid in liked
    ]
    used = set(liked)
    rng = random.Random(20261117)
    candidates = sorted(papers.values(), key=lambda p: p["paper_id"])
    rng.shuffle(candidates)
    counts = {t: 0 for t in TOPICS}
    while len(selected) < 100:
        added = False
        for topic, pattern in TOPICS.items():
            choices = [
                p
                for p in candidates
                if p["paper_id"] not in used
                and re.search(pattern, p["title"] + " " + p["abstract"], re.I)
            ]
            if choices:
                p = choices[0]
                selected.append(
                    {
                        **p,
                        "selection_source": "cached_arxiv_topic_sample",
                        "selection_topic": topic,
                    }
                )
                used.add(p["paper_id"])
                counts[topic] += 1
                added = True
                if len(selected) == 100:
                    break
        if not added:
            raise ValueError("Not enough topic candidates")
    rng.shuffle(selected)
    result = {
        "schema_version": 1,
        "sample_id": "dmitrii-interest-100-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_scale": {
            "0": "not interesting",
            "0.5": "potentially interesting",
            "1": "interesting",
        },
        "selection": {
            "seed": 20261117,
            "liked_count": len(liked),
            "topic_counts": counts,
            "source": "Existing arXiv metadata cache plus verified AlphaXiv likes; stratified convenience sample, not representative random ML sample",
            "liked_page": "https://www.alphaxiv.org/bookmarks/liked",
            "topic_patterns": TOPICS,
        },
        "papers": selected,
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    main()
