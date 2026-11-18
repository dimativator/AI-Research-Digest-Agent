# Reproducing M3

Run commands from the repository root. The recorded runs used Python 3.12, Codex CLI 0.169.0, and `gpt-6-sol`. You need a Codex account with access to that model for fresh runs. Hosted model outputs are stochastic, so identical inputs do not guarantee identical selections.

## Environment

```bash
conda env create -f environment.yml
conda activate ai-research-digest
codex --version
codex login
```

The CLI must already be installed. No OpenAI API key is required: the harness uses Codex login. `.env.example` documents optional variables. The program does not automatically load `.env` files. X is disabled in these evaluations, so browser setup is unnecessary. E5 runs locally on CPU and downloads its pinned weights on first use.

## Recompute scores after the CORE label revision

```bash
python -m eval.recalculate_core_label
```

This applies the user-requested CORE rating change from 0 to 1 to both evaluations. Revised weekly results are saved under `analysis/core-label-revision/`, and repeated-run results in `analysis-core-label-revision.json`. Original frozen labels and results remain unchanged. Current report tables use this revision.

## Recompute the saved results without model calls

```bash
python eval/runs/weekly-20261118/analysis/analyze.py
```

Expected: the five-system table, weekly bootstrap intervals, and pairwise statistics. Mean relevance@7 is 0.490 / 0.469 / 0.531 / 0.551 / 0.582 for BM25 / E5 / Agents 1 / 2 / 3. The script uses the exported human labels, not the local labeling database. It writes `results.json`, `weekly-scores.csv`, and analysis figures beside the script.

To regenerate the pyplot figure, use a Python environment with Matplotlib installed:

```bash
python eval/runs/weekly-20261118/analysis/plot_weekly_comparisons.py
```

Expected: PNG, SVG, and PDF files under `reports/figures/`. Matplotlib is optional and is not installed by the core environment file.

## Fresh evaluation on the frozen weekly corpora

These commands use the saved 21,006-paper metadata, profile, rankings, and historical source snapshot. They bypass arXiv metadata collection, preserve the original 7-paper digest size, and make fresh Codex calls. Choose a new output directory for every launch.

```bash
# Agent 1
python -m eval.replay_weekly --variant react --output eval/runs/replay-agent1

# Agent 2
python -m eval.replay_weekly --variant react-explore --output eval/runs/replay-agent2

# Agent 3
python -m eval.replay_weekly --variant debate --output eval/runs/replay-agent3

# Or all three, with three concurrent workers
python -m eval.replay_weekly --output eval/runs/replay-all
```

Expected: seven runs per selected variant, with `result.json`, `outcome.json`, `state.json`, `calls.jsonl`, `trace.jsonl`, and SQLite state under each weekly directory. Each completed digest contains seven papers. Failures are saved and must not be counted as successful negative predictions. Tool responses replay existing fixtures, capturing any previously unseen requests. Newly selected papers outside the original 122-paper pool require new labels before scoring.

## Repeated-run experiment on 1,000 papers

```bash
# Build the same seeded sample, then run 10 repetitions of each agent
# with up to 10 workers. Use a NEW output directory for a fresh experiment.
python -m eval.run_repeated --output eval/runs/repeated-1000-replay

# Resume unfinished jobs in that directory; existing outcomes are preserved
python -m eval.run_repeated --output eval/runs/repeated-1000-replay

# Analyze the original experiment without starting new model calls
python -m eval.run_repeated --analyze-only
```

Expected: 30 outcomes, each successful run selecting ten papers. `sample.json` contains 122 human-labeled papers and 878 seeded random papers assigned assumed-negative labels. `shortlist-coverage.json` records pre-screening coverage: BM25 11/31 interesting and 20/76 interesting-or-potential papers, E5 9/31 and 22/76. `analysis.json` reports completion, mean relevance, standard deviation, score range, and selection overlap. The report subsection is generated when all 30 outcomes exist. The current report updater links to the original experiment directory, so use `analysis.json` as the authoritative result for a custom output directory.

The original experiment began with 20 repetitions and three workers. The user changed it to ten repetitions and ten workers during execution. The first ten jobs per variant in the original shuffled order were retained, including every started run. `schedule-change.json` documents interrupted/resumed calls. A fresh replay starts with ten workers and cannot reproduce the original timing conditions exactly.

## Run a live digest through the common entry point

Import the frozen structured profile into a separate local database:

```bash
python - <<'PY'
import json
from pathlib import Path
from research_digest.profiles.user_profile import ProfileStore, UserProfile
profile = UserProfile.from_dict(json.loads(Path('eval/runs/weekly-20261118/profile.json').read_text()))
ProfileStore(Path('data/reproduce.sqlite3')).save(profile)
PY

python run_digest.py --db data/reproduce.sqlite3 --user-id dmitrii --variant react --model gpt-6-sol --date-from 2026-11-09 --date-to 2026-11-15 --digest-size 7 --no-X
python run_digest.py --db data/reproduce.sqlite3 --user-id dmitrii --variant react-explore --model gpt-6-sol --date-from 2026-11-09 --date-to 2026-11-15 --digest-size 7 --no-X
python run_digest.py --db data/reproduce.sqlite3 --user-id dmitrii --variant debate --model gpt-6-sol --date-from 2026-11-09 --date-to 2026-11-15 --digest-size 7 --no-X
```

These are live demonstrations using CLI defaults, not the frozen evaluation protocol. They access arXiv and print a readable digest or an explicit error/resume command. The evaluation runners use unlimited budgets, fresh databases, and frozen inputs.

## Trace provenance

The completed weekly evaluation is under `eval/runs/weekly-20261118/`. Every JSONL record in its per-agent exports includes `variant` and `task`. `trace-annotation-audit.json` records hashes before and after adding these identifiers. Original event payloads were preserved. SQLite databases and historical source snapshots remain unchanged. The active repeated-run experiment writes variant-tagged events from the outset and is committed only after completion.
