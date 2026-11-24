# AI Research Digest Agent

A Python research digest agent with a handwritten ReAct harness. It retrieves arXiv papers for a date range, ranks them against a researcher profile, screens abstracts, and lets an LLM gather more evidence before choosing the digest.

The thinking step defaults to Codex CLI through a small OpenAI-style Python adapter. An optional OpenAI API backend reads a key from a file. Python owns retrieval, tools, validation, budgets, and SQLite state. Neither backend requires the OpenAI Python SDK.

Milestone 2 evaluation, results, and annotated failure traces: [reports/M2.md](reports/M2.md).

## OpenAI API backend

Put your API key alone in `.secrets/openai-key.txt` (create `.secrets/` first).
This directory is ignored by Git. Restrict file access with
`chmod 600 .secrets/openai-key.txt`.

```bash
python run_digest.py --user-id dmitrii \
  --date-from 2026-11-09 --date-to 2026-11-15 \
  --variant react-explore --no-X \
  --llm-backend openai --api-key-file .secrets/openai-key.txt \
  --model YOUR_API_MODEL
```

Use an API model available to your account that supports Chat Completions and
JSON Schema structured output. Codex model names are not necessarily API model
names. The backend uses the [Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)
with no native tools. The harness executes application actions returned as JSON.
API charges are separate from your Codex subscription.

The same three flags work with `create_profile.py`. Omitting them keeps the
Codex backend. API runs require an explicit model and key-file path.

The backend and model are saved in run configuration. The key and its file path
are not saved in traces or SQLite. Supply the key file again when resuming:

```bash
python run_digest.py --resume RUN_ID --api-key-file .secrets/openai-key.txt
```

Usage and latency flow through the existing metrics. Screening caches are
separate for each backend. HTTP 429 and server errors receive up to two retries,
with numeric Retry-After delays capped at 75 seconds. Other errors fail visibly.

## Quick start

Run commands from the repository root.

```bash
conda env create -f environment.yml
conda activate ai-research-digest
codex login
python try_codex.py "Explain BM25 in one sentence."
```

The environment uses Python 3.12. The adapter targets Codex CLI flags compatible with version 0.146.0. Incompatible CLI flags fail visibly.

Create a structured profile from a UTF-8 text file:

```bash
python create_profile.py researcher_profiles/dmitrii.txt --user-id dmitrii
```

Run a digest without X analysis:

```bash
python run_digest.py --user-id dmitrii --date-from 2026-10-20 --date-to 2026-10-24 --no-X --verbose
```

The terminal shows live JSON trace events on stderr and a readable digest on stdout. The default database is `data/digest.sqlite3`, resolved relative to the repository rather than the working directory.

To use X analysis, set up the Arc session first:

```bash
python setup_x.py
python run_digest.py --user-id dmitrii --date-from 2026-10-20 --date-to 2026-10-24 --verbose
```

Arc integration requires macOS, Arc, an authenticated X session in its active profile, and permission to automate Arc and execute JavaScript through Apple Events. Production collection uses Arc through AppleScript. Playwright is included for browser fixture tests.

## Pipeline

```text
Researcher profile text
    -> LLM structures interests and infers arXiv categories
    -> SQLite stores a versioned profile

Categories + inclusive UTC submission dates
    -> Fetch matching arXiv metadata
    -> Deduplicate and cache in SQLite
    -> BM25 rank titles and abstracts
    -> Screen the initial shortlist with the LLM in batches
    -> Collect X attention for relevant/uncertain papers, unless --no-X
    -> ReAct: choose action -> validate -> execute Python tool -> observe
    -> Validate final selections and supporting quotes
    -> Save digest and recommendation history
    -> Print readable output
```

### 1. Researcher profile

The input file describes interests, goals, and exclusions in plain text. The LLM turns it into a structured profile with topics, keywords, supporting quotes, and relevant arXiv categories. Source URLs in the file are not fetched.

Profiles are stored by user ID with revision history. An unchanged latest profile does not create another revision. Editing the text file does not update the database until `create_profile.py` is run again. Empty profile files cannot produce a valid profile.

```bash
python create_profile.py researcher_profiles/dmitrii.txt --user-id dmitrii
python create_profile.py --user-id dmitrii --show
```

Categories normally come from the saved profile. If an older profile has none, the digest command infers categories and saves an updated profile. Repeated `--category` flags override categories for one run.

### 2. arXiv retrieval and storage

Dates refer to first submission, in UTC, with both endpoints included. Matching includes category cross-lists. The client paginates results, throttles requests, retries retrieval failures, and checks completeness. Large result windows are split by date to stay within the retrieval window limit.

Metadata pages are cached. `--refresh-metadata` requests fresh metadata, which can capture delayed indexing and updated records. Each run stores its own candidate snapshot.

Previously recommended papers remain eligible by default. `--new` excludes papers already recommended to that user. Repeating a date range permits the same recommendations, but LLM decisions can vary between runs.

### 3. BM25 shortlist

BM25 scores each title plus abstract against the profile's topic names and keywords. Tokenization uses lowercase ASCII letters and digits. The implementation uses $k_1=1.5$ and $b=0.75$, with paper ID as the tie-breaker. There are no embeddings or cosine similarity in the current implementation.

The initial shortlist defaults to 30 papers. The full eligible corpus remains available for later expansion and focused searches.

### 4. LLM batch screening

Each screening request contains the profile and a batch of abstracts. The model returns one label per paper: `relevant`, `uncertain`, or `irrelevant`, with a reason and an exact supporting quote from that paper's visible title or abstract.

The default batch size is 10. Separate internal bounds limit batch payloads to about 32,000 characters and each screening abstract to 8,000 characters. Invalid screening results receive validation feedback and up to three total attempts. Attempts consume the LLM-call budget.

Labels guide the agent and automatic X collection. They do not permanently exclude a screened paper from final selection.

### 5. ReAct decisions and tools

The model receives the profile, screened candidates, available tools, observations, and remaining budgets. It emits a structured decision with a brief action reason. The harness validates it, executes the selected Python function, saves the observation, and requests another decision. The trace records action reasons, not hidden chain-of-thought.

| Action | Behavior |
| --- | --- |
| `inspect_paper` | Takes a screened paper ID and a specific question. Retrieves matching arXiv HTML passages with section names, chunk IDs, and neighboring context. |
| `search_candidates` | Runs a focused BM25 query over the entire fetched eligible corpus, admits new hits within the candidate cap, and screens them. Accepts 1–20 results. |
| `expand_shortlist` | Screens the next papers in the original profile-based BM25 ranking. Uses `--expansion-size`. |
| `find_code` | Finds GitHub candidates from paper links and GitHub README searches by arXiv ID/title. Reports association evidence and available repository metadata. |
| `analyze_twitter` | Returns cached X attention or collects it for a screened paper. Unavailable when `--no-X` is set. |
| `finish` | Submits final selections with summaries, relevance explanations, supporting quotes, and limitations. |

Inspection uses lexical passage retrieval over HTML, with no PDF fallback or extra LLM summarization step. A missing passage does not prove the requested experiment is absent. Retrieved documents are cached, and repeated identical inspection questions reuse observations. New questions consume the inspection budget.

`search_candidates` and `expand_shortlist` share the expansion budget. Neither changes the run's dates or categories. Only admitted, screened paper IDs can be selected.

`find_code` distinguishes paper-linked repositories, README mentions, and unverified title matches. It never clones or executes code. `GITHUB_TOKEN` is optional for GitHub requests. Repository availability does not establish reproducibility.

### 6. X attention

Automatic X collection runs before ReAct decisions for screened papers labeled relevant or uncertain, including newly admitted papers. Queries use arXiv URL/ID variants and the exact paper title, with the digest's date range and retweets excluded. Searches use Latest results, capped at 50 collected posts per query and five scroll iterations.

Attention levels use observed distinct posting accounts:

| Accounts | Level |
| --- | --- |
| 0 verified matches | None |
| 1–2 | Low |
| 3–9 | Moderate |
| 10+ | High |

Unavailable searches mean unknown attention. Engagement counts are snapshots, and these searches are samples rather than complete measurements of X activity.

One paper with the most observed attention among checked relevant/uncertain papers becomes the X feature. Ranking uses distinct accounts, then reposts, then likes. The featured paper must appear exactly once in the digest and counts toward its size. Its feature includes up to five post links ranked by observed engagement. If no verified posts are found, there is no X feature. Other papers' attention summaries are available to the model as supplemental selection signals.

`--no-X` skips collection, hides the X tool, and disables the featured paper. It is saved with the run configuration.

### 7. Final validation and output

Final selections must reference screened candidates. Supporting quotes must match abstract text shown to the decision model or an inspected passage actually retrieved for that paper. Inspection quotes cannot span unrelated passages. The decision context uses abstract excerpts of up to 1,200 characters, alongside screening evidence.

The harness enforces digest size and the X feature requirement, adds evidence limitations, and saves the result and recommendation history in a transaction. A budget-exhausted run returns no final recommendations. Readable text is the default terminal format. JSON remains available for export and traces.

## Digest command flags

These are the defaults of `python run_digest.py`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--user-id ID` | Required for new runs | Load this user's latest saved profile. |
| `--date-from YYYY-MM-DD` | Required for new runs | Inclusive first-submission start date, UTC. |
| `--date-to YYYY-MM-DD` | Required for new runs | Inclusive first-submission end date, UTC. |
| `--category CATEGORY` | Profile categories | Repeat to override categories for this run. |
| `--model MODEL` | CLI built-in default | Pass an explicit model name to the Codex subprocess. |
| `--db PATH` | `data/digest.sqlite3` | SQLite database path. |
| `--new` | Off | Exclude this user's previous recommendations. |
| `--no-X` | Off | Disable all X analysis for the run. Case-sensitive. |
| `--refresh-metadata` | Off | Fetch metadata instead of reusing cached pages. |
| `--verbose` | Off | Print saved trace events to stderr as execution proceeds. |
| `--format text\|json` | `text` | Final terminal output format. |
| `--output PATH` | None | Also write the final result as JSON, regardless of `--format`. |
| `--resume RUN_ID` | None | Continue with the saved configuration or return the stored result. |
| `--trace RUN_ID` | None | Print saved JSON events without LLM calls. |
| `--shortlist-size N` | `30` | Initial number of papers admitted for screening. |
| `--batch-size N` | `10` | Maximum papers per screening request. |
| `--expansion-size N` | `10` | Papers added by an ordinary shortlist expansion. |
| `--max-candidates N` | `-1` | Maximum total admitted candidates, across all searches and expansions. |
| `--digest-size N` | `5` | Maximum final selections, including the X feature. Fewer are allowed. |
| `--max-steps N` | `-1` | Maximum ReAct decision requests. |
| `--max-llm-calls N` | `-1` | Maximum screening and decision attempts during a run. |
| `--max-inspections N` | `-1` | Maximum new paper-inspection questions. |
| `--max-expansions N` | `-1` | Shared budget for shortlist expansion and focused search. |
| `--page-size N` | `100` | arXiv request page size, from 1 to 1,000. |
| `--max-context-chars N` | `-1` | Local bound on serialized payload plus instructions, in characters. |

`-1` means unlimited for maximum limits and `--digest-size`. It is stored internally as `None`. It does not remove model context limits, retrieval bounds, or the finite corpus. Shortlist, batch, expansion, and page sizes must be positive. A finite candidate cap must be at least the initial shortlist size. Zero is allowed for step, LLM-call, inspection, and expansion budgets to disable those operations.

Each ReAct decision request consumes a step, including `finish` and invalid decisions. Batch screening consumes LLM calls but no ReAct steps. Profile creation and category inference are outside the run's LLM-call counter.

The Python `DigestConfig` class has different defaults for direct library use: 60 candidates, digest size 10, 8 steps, 20 LLM calls, 4 inspections, 3 expansions, and 160,000 context characters. The CLI explicitly overrides these with the defaults in the table.

Resume and trace modes accept only `--db`, `--output`, `--verbose`, and `--format` alongside their run ID. You cannot change a saved run's budgets, model, dates, or X setting through resume flags. Stored results, including budget-exhausted results, are returned as saved. Start a new run to use different settings. Incompatible pipeline versions cannot be resumed.

## Examples

Use explicit categories and only new recommendations:

```bash
python run_digest.py --user-id dmitrii --date-from 2026-10-20 --date-to 2026-10-24 --category cs.LG --category cs.CL --new --no-X
```

Set finite budgets and save JSON while keeping readable terminal output:

```bash
python run_digest.py --user-id dmitrii --date-from 2026-10-20 --date-to 2026-10-24 --no-X --max-steps 20 --max-llm-calls 40 --max-inspections 8 --max-expansions 4 --output digest.json
```

Select a model explicitly, if it is available to your Codex account:

```bash
python try_codex.py "Explain BM25 briefly." --model gpt-5.6-sol
python create_profile.py researcher_profiles/dmitrii.txt --user-id dmitrii --model gpt-5.6-sol
python run_digest.py --user-id dmitrii --date-from 2026-10-20 --date-to 2026-10-24 --model gpt-5.6-sol --no-X
```

View a trace or resume an interrupted run, replacing `RUN_ID` with the printed ID:

```bash
python run_digest.py --trace RUN_ID
python run_digest.py --resume RUN_ID --verbose
```

Separate machine-readable output from the live trace:

```bash
python run_digest.py --user-id dmitrii --date-from 2026-10-20 --date-to 2026-10-24 --no-X --format json --verbose > digest.json 2> trace.log
```

## Reset for a fresh test

Stop any active digest run first, then run:

```bash
python clear_db.py
# Or use a separate database:
python clear_db.py --db data/test.sqlite3
```

The script backs up the existing database under its sibling `backups/` directory before atomically deleting papers, metadata pages, runs, candidate snapshots, traces, recommendations, screenings, inspections, and X caches. Researcher profiles remain intact. Arc login is stored outside this database and is unaffected. A missing database produces an error.

## Implementation

### Code map

| Location | Responsibility |
| --- | --- |
| `run_digest.py`, `create_profile.py`, `try_codex.py`, `clear_db.py` | Root CLI launchers. |
| `setup_x.py` | Check Arc scripting and X login. |
| `research_digest/cli/` | Arguments, command entry points, readable digest formatting, reset. |
| `research_digest/agent/config.py` | Run configuration, budget helpers, pipeline version. |
| `research_digest/agent/protocol.py` | Tool descriptions, prompts, JSON schemas, decision validation. |
| `research_digest/agent/digest_agent.py` | Pipeline state machine, tool dispatch, evidence validation, resume. |
| `research_digest/llm/codex_llm.py` | Synchronous OpenAI-style interface over Codex CLI. |
| `research_digest/profiles/user_profile.py` | Profile structuring, validation, category inference, revisions. |
| `research_digest/retrieval/arxiv_source.py` | arXiv queries, pagination, parsing, HTML retrieval. |
| `research_digest/retrieval/ranking.py` | BM25 over papers and inspection passages. |
| `research_digest/retrieval/paper_content.py` | HTML sections, chunks, links, question-based passage retrieval. |
| `research_digest/retrieval/code_source.py` | GitHub search and repository association evidence. |
| `research_digest/retrieval/arc_browser.py` | Native Arc tab automation. |
| `research_digest/retrieval/x_source.py` | X queries and rendered post collection. |
| `research_digest/retrieval/x_attention.py` | Post matching, attention levels, feature ranking. |
| `research_digest/storage/paper_store.py` | SQLite paper/run/cache persistence and event history. |
| `researcher_profiles/` | Plain-text researcher profile inputs. |
| `tests/` | Adapter, pipeline, retrieval, tool, and browser fixture checks. |

### Codex adapter

```python
from research_digest.llm.codex_llm import CodexOpenAI

client = CodexOpenAI(timeout=120)
response = client.responses.create(
    instructions="Assess papers using only the supplied evidence.",
    input="A paper reports an improvement but gives no baseline. What is missing?",
)
print(response.output_text)
```

`client.chat.completions.create(...)` is also supported. See `examples/codex_thinking.py` for JSON Schema output. This is a small synchronous subset, with no streaming, multimodal input, sampling controls, or complete OpenAI SDK compatibility.

Every request starts a fresh ephemeral `codex exec` session in a temporary directory. The adapter ignores user configuration and project documents, supplies its own instructions, disables tool providers, requests a read-only sandbox, removes API-key environment variables from the child process, and reuses CLI authentication. An explicit model is passed with `--model`. Without one, the subprocess uses its built-in default rather than a model override in the user's ignored configuration.

Messages are serialized into one prompt, so API-style role precedence is approximate. JSON Schema output is requested through the CLI. Unexpected tool events cause the adapter to reject the output. Event rejection cannot undo an action that already happened.

### Context and state

The harness constructs context explicitly for every LLM call. ReAct context includes the profile, candidate excerpts, screening assessments, observations, last error/result, and budgets. There is no harness context compaction or automatic summarization. Full observations can accumulate across steps.

`--max-context-chars` checks payload plus instructions before calling Codex. It counts characters rather than tokens and excludes CLI wrappers and schema overhead. Disabling it does not make the model's context unlimited.

SQLite stores profile revisions, paper versions, metadata pages, immutable run candidates, screening caches, inspected documents/code results, X caches, run state, events, and recommendation history. Screening cache keys include pipeline version, profile, model, and paper fingerprints. With no explicit model, screening cache reuse is scoped to the run because the CLI default may change. Persisted admission and pending-action state allow interrupted work to continue without restarting the whole pipeline. External calls can be repeated after an interruption that occurred before their result was saved.

## Troubleshooting and checks

| Message or symptom | Meaning / next action |
| --- | --- |
| `Agent decision-step budget exhausted` | The model reached `--max-steps` before a valid finish. Start a new run with a larger value or `-1`. |
| `LLM call budget exhausted` | Screening, retries, or decisions used the call budget. Increase it for a new run. |
| `Prompt exceeds max_context_chars` | The harness's local character check failed. Use a larger value, `-1`, or smaller inputs on a new run. |
| `Screening evidence is absent from the supplied text` | A returned quote did not exactly match visible source text. Screening retries validation failures, then fails if all attempts remain invalid. Inspect the trace. |
| X is unavailable | Read the recorded error. Check Arc/X login with `python setup_x.py`, or run with `--no-X`. Unavailable attention is unknown, not zero. |
| Codex rejects a flag or fails to start | Check CLI installation, authentication, and compatibility with the adapter's flags. |

Run the test suite in the project environment:

```bash
conda activate ai-research-digest
python -m playwright install chromium
python -m unittest discover -s tests -q
```

Chromium is used by browser fixture tests. These tests do not require a live X login or live LLM calls.
