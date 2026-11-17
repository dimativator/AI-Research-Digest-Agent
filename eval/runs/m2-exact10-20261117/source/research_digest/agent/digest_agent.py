"""A bounded ReAct harness: decide -> validate -> act -> observe -> repeat.

Codex emits structured decisions only. Python owns every tool and all state.
The recorded reason is a short action justification, not hidden model reasoning.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from research_digest.retrieval.arxiv_source import (
    ArxivClient,
    Paper,
    collect_metadata,
)
from research_digest.llm.usage import measure_call, summarize_usage
from research_digest.llm.codex_llm import CodexOpenAI
from research_digest.storage.paper_store import PaperStore
from research_digest.retrieval.ranking import RankedPaper, bm25
from research_digest.retrieval.paper_content import EXTRACTOR_VERSION, retrieve_passages
from research_digest.retrieval.code_source import CODE_FINDER_VERSION, CodeFinder
from research_digest.retrieval.x_source import COLLECTOR_VERSION, XBrowserClient
from research_digest.retrieval.x_attention import attention_summary, featured_paper
from research_digest.profiles.user_profile import UserProfile

from .config import DigestConfig, PIPELINE_VERSION, limit_reached, remaining_budget
from .protocol import (
    DECISION_INSTRUCTIONS,
    DECISION_SCHEMA,
    SCREENING_INSTRUCTIONS,
    SCREEN_SCHEMA,
    TOOLS,
    require_text,
    validate_decision,
    validate_screening_response,
)


SCREENING_ABSTRACT_CHARS = 8_000
SCREENING_BATCH_CHARS = 32_000
SCREENING_MAX_ATTEMPTS = 3
DECISION_ABSTRACT_CHARS = 1_200


class BudgetExhausted(RuntimeError):
    pass


class DigestAgent:
    """Coordinate retrieval, screening, and resumable tool-driven digest runs."""

    def __init__(
        self,
        store: PaperStore,
        *,
        client: CodexOpenAI | None = None,
        arxiv: ArxivClient | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        x_client: XBrowserClient | None = None,
        code_finder: CodeFinder | None = None,
    ) -> None:
        self.store = store
        self.client = client if client is not None else CodexOpenAI()
        self.arxiv = arxiv if arxiv is not None else ArxivClient()
        self.on_event = on_event
        self.x_client = x_client if x_client is not None else XBrowserClient()
        self.code_finder = code_finder if code_finder is not None else CodeFinder()
        self.run_id = ""
        self.state: dict[str, Any] = {}

    def start(self, profile: UserProfile, config: DigestConfig) -> str:
        config.validate()
        self.state = {
            "pipeline_version": PIPELINE_VERSION,
            "profile": profile.to_dict(),
            "config": asdict(config),
            "phase": "harvest",
            "screened": {},
            "admitted_ids": [],
            "observations": {},
            "x_collector": COLLECTOR_VERSION,
            "pending": None,
            "screen_target": config.shortlist_size,
            "llm_calls": 0,
            "steps": 0,
            "inspections": 0,
            "expansions": 0,
        }
        self.run_id = self.store.create_run(profile.user_id, self.state)
        return self.run_id

    def run(self, run_id: str) -> dict[str, Any]:
        """Resume persisted work, execute the pipeline, and commit its outcome."""
        completed_result = self._restore_run(run_id)

        if completed_result is not None:
            return completed_result

        try:
            self._prepare_candidates()

            if (
                self.config.digest_size is not None
                and self._candidate_limit() < self.config.digest_size
            ):
                raise ValueError(
                    f"Cannot select exactly {self.config.digest_size} papers: only "
                    f"{self._candidate_limit()} candidates are available within the run limits."
                )

            if not self.state["candidate_count"]:
                return self._complete_run(
                    "completed",
                    [],
                    ["No eligible matching papers were available in the API index."],
                )

            if self.config.variant == "selection-critic":
                from .selection_critic import SelectionCritic

                return SelectionCritic(self).run()
            return self._run_react_loop()
        except BudgetExhausted as error:
            return self._complete_run(
                "budget_exhausted", [], [str(error), "No final digest was committed."]
            )
        except Exception as error:
            self._persist_run(
                {"type": "run_failed", "error": str(error)}, status="failed"
            )
            raise
        finally:
            self.x_client.close()

    def _persist_run(
        self, event: dict[str, Any] | None = None, *, status: str = "running"
    ) -> None:
        self.store.update_run(self.run_id, self.state, status=status, event=event)

        if event is not None and self.on_event is not None:
            self.on_event(event)

    def _request_model(
        self,
        instructions: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
        name: str,
    ) -> dict[str, Any]:
        if limit_reached(self.state["llm_calls"], self.config.max_llm_calls):
            raise BudgetExhausted("LLM call budget exhausted")

        content = json.dumps(payload, ensure_ascii=False)

        if (
            self.config.max_context_chars is not None
            and len(content) + len(instructions) > self.config.max_context_chars
        ):
            raise ValueError(
                "Prompt exceeds max_context_chars; reduce shortlist or batch size"
            )

        self.state["llm_calls"] += 1
        # Charge attempts before inference so restart cannot bypass the budget.
        self._persist_run(
            {"type": "llm_request", "purpose": name, "call": self.state["llm_calls"]}
        )
        records = self.state.setdefault("llm_usage", [])
        try:
            return self._measured_model_request(
                records, instructions, content, schema, name
            )
        finally:
            self._persist_run({"type": "llm_usage", "purpose": name, **records[-1]})

    def _measured_model_request(self, records, instructions, content, schema, name):
        response = measure_call(
            self.client.responses.create,
            records,
            model=self.config.model,
            instructions=instructions,
            input=content,
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        return json.loads(response.output_text)

    def _build_profile_context(self) -> dict[str, Any]:
        return {
            key: self.state["profile"][key]
            for key in ("summary", "topics", "goals", "exclusions")
        }

    def _screening_cache_key(self, paper: Paper) -> str:
        # An unspecified CLI default can change between runs. Reuse that cache
        # only within this run; an explicit model enables cross-run reuse.
        model_key = self.config.model or ("cli-default:" + self.run_id)
        payload = [
            PIPELINE_VERSION,
            self.profile.fingerprint,
            model_key,
            paper.fingerprint,
        ]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    def _screen_candidates(self, target: int) -> None:
        target = min(target, self._candidate_limit())
        admitted = self.state["admitted_ids"]
        for candidate in self._corpus_candidates():
            if len(admitted) >= target:
                break
            if candidate.paper.paper_id not in admitted:
                admitted.append(candidate.paper.paper_id)
        # Save explicit admission before inference so an interrupted focused search
        # resumes the same paper IDs rather than a prefix of the original ranking.
        self._persist_run()

        while any(pid not in self.state["screened"] for pid in admitted[:target]):
            batch = self._prepare_screening_batch(target)

            if batch:
                self._screen_batch(batch)

            self._persist_run(
                {
                    "type": "screening_progress",
                    "screened": len(self.state["screened"]),
                    "target": target,
                }
            )

    def _prepare_screening_batch(
        self, target: int
    ) -> list[tuple[Paper, dict[str, Any]]]:
        requested = self.state["admitted_ids"][:target]
        by_id = {
            candidate.paper.paper_id: candidate
            for candidate in self._corpus_candidates()
        }
        candidates = [
            by_id[pid] for pid in requested if pid not in self.state["screened"]
        ]
        batch = []
        batch_chars = 0

        for candidate in candidates:
            if len(batch) >= self.config.batch_size:
                break
            paper = candidate.paper
            cached = self.store.cached_screening(self._screening_cache_key(paper))

            if cached is not None:
                self.state["screened"][paper.paper_id] = cached
                continue

            visible_paper = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": paper.abstract[:SCREENING_ABSTRACT_CHARS],
                "abstract_truncated": len(paper.abstract) > SCREENING_ABSTRACT_CHARS,
            }
            paper_chars = len(json.dumps(visible_paper))

            if batch and batch_chars + paper_chars > SCREENING_BATCH_CHARS:
                break

            batch.append((paper, visible_paper))
            batch_chars += paper_chars

        return batch

    def _screen_batch(self, batch: list[tuple[Paper, dict[str, Any]]]) -> None:
        payload = {
            "profile": self._build_profile_context(),
            "papers": [visible for _, visible in batch],
        }
        visible_papers = {paper.paper_id: visible for paper, visible in batch}

        for attempt in range(1, SCREENING_MAX_ATTEMPTS + 1):
            response = self._request_model(
                SCREENING_INSTRUCTIONS, payload, SCREEN_SCHEMA, "screen_papers"
            )

            try:
                assessments = validate_screening_response(response, visible_papers)
                break
            except ValueError as error:
                self._persist_run(
                    {
                        "type": "invalid_screening",
                        "attempt": attempt,
                        "max_attempts": SCREENING_MAX_ATTEMPTS,
                        "error": str(error),
                    }
                )

                if attempt == SCREENING_MAX_ATTEMPTS:
                    raise

                payload["validation_error"] = str(error)

        for paper, _ in batch:
            assessment = assessments[paper.paper_id]
            self.store.save_screening(self._screening_cache_key(paper), assessment)
            self.state["screened"][paper.paper_id] = assessment

        self._persist_run({"type": "screening_result", "assessments": assessments})

    def _candidate_limit(self) -> int:
        """Unlimited candidates still stop at the size of the fetched corpus."""
        candidate_count = self.state["candidate_count"]

        if self.config.max_candidates is None:
            return candidate_count

        return min(candidate_count, self.config.max_candidates)

    def _corpus_candidates(self) -> list[RankedPaper]:
        return self.store.candidates(self.run_id, limit=self.state["candidate_count"])

    def _admitted_candidates(self) -> list[RankedPaper]:
        by_id = {
            candidate.paper.paper_id: candidate
            for candidate in self._corpus_candidates()
        }
        return [by_id[pid] for pid in self.state["admitted_ids"]]

    def _paper(self, paper_id: str) -> Paper:
        return next(
            candidate.paper
            for candidate in self._admitted_candidates()
            if candidate.paper.paper_id == paper_id
        )

    def _build_decision_context(self) -> dict[str, Any]:
        visible = []

        for candidate in self._admitted_candidates():
            paper = candidate.paper

            if paper.paper_id in self.state["screened"]:
                visible.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "abstract_excerpt": paper.abstract[:DECISION_ABSTRACT_CHARS],
                        "abstract_truncated": len(paper.abstract)
                        > DECISION_ABSTRACT_CHARS,
                        "bm25_score": candidate.score,
                        "assessment": self.state["screened"][paper.paper_id],
                        "code": self.state["observations"].get(
                            f"code:{paper.paper_id}", {"status": "not_checked"}
                        ),
                        "x_attention": attention_summary(
                            self.state.get("x_attention", {}).get(
                                paper.paper_id,
                                {
                                    "status": (
                                        "not_checked"
                                        if self.config.x_enabled
                                        else "disabled"
                                    ),
                                    "level": "unknown",
                                },
                            )
                        ),
                    }
                )

        return {
            "profile": self._build_profile_context(),
            "tools": [
                tool
                for tool in TOOLS
                if self.config.x_enabled or tool["name"] != "analyze_twitter"
            ],
            "candidates": visible,
            "featured_x_paper_id": self._featured_x_paper_id(),
            "observations": self.state["observations"],
            "last_error": self.state.get("last_error"),
            "last_observation": self.state.get("last_observation"),
            "bounds": {
                "categories": self.config.categories,
                "date_from": self.config.date_from,
                "date_to": self.config.date_to,
                "digest_size": self.config.digest_size,
                "max_candidates": self.config.max_candidates,
                "new_only": self.config.new_only,
                "x_enabled": self.config.x_enabled,
            },
            "remaining": {
                "llm_calls": remaining_budget(
                    self.state["llm_calls"], self.config.max_llm_calls
                ),
                "steps": remaining_budget(self.state["steps"], self.config.max_steps),
                "inspections": remaining_budget(
                    self.state["inspections"], self.config.max_inspections
                ),
                "expansions": remaining_budget(
                    self.state["expansions"], self.config.max_expansions
                ),
                "candidates": self._candidate_limit() - len(visible),
            },
        }

    def _build_selections(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        if (
            self.config.digest_size is not None
            and len(decision["selections"]) != self.config.digest_size
        ):
            raise ValueError(
                f"Digest must contain exactly {self.config.digest_size} unique papers; "
                f"received {len(decision['selections'])}. Revise the selection to "
                "the required count, using screened candidates."
            )

        candidates = self._admitted_candidates()
        papers_by_id = {
            candidate.paper.paper_id: candidate.paper for candidate in candidates
        }
        selected_papers = []
        selected_ids = set()

        for selection in decision["selections"]:
            required_fields = {"paper_id", "summary", "relevance", "evidence"}

            if not isinstance(selection, dict) or set(selection) != required_fields:
                raise ValueError("Invalid selection fields")

            paper_id = require_text(selection["paper_id"], "selected paper_id", 100)

            if paper_id not in self.state["screened"] or paper_id in selected_ids:
                raise ValueError("Selections must be unique screened papers")

            paper = papers_by_id[paper_id]
            require_text(selection["summary"], "summary")
            require_text(selection["relevance"], "relevance")
            self._validate_selection_evidence(paper, selection["evidence"])

            selected_ids.add(paper_id)
            selected_papers.append(
                {
                    **selection,
                    "title": paper.title,
                    "url": paper.url,
                    "version": paper.version,
                    "code": self.state["observations"].get(
                        f"code:{paper_id}", {"status": "not_checked"}
                    ),
                    "attention": attention_summary(
                        self.state.get("x_attention", {}).get(
                            paper_id, {"status": "not_checked", "level": "unknown"}
                        )
                    ),
                    "featured_on_x": paper_id == self._featured_x_paper_id(),
                }
            )

        featured_id = self._featured_x_paper_id()
        if featured_id is not None and featured_id not in selected_ids:
            raise ValueError(
                f"Include featured X paper {featured_id} once in selections, with grounded "
                "summary and relevance. It counts toward digest_size."
            )
        return selected_papers

    def _validate_selection_evidence(self, paper: Paper, evidence_items: Any) -> None:
        if not isinstance(evidence_items, list) or not evidence_items:
            raise ValueError("Each selection requires supporting evidence")

        inspection = self.state["observations"].get(f"inspect:{paper.paper_id}", {})

        for evidence in evidence_items:
            if not isinstance(evidence, dict) or set(evidence) != {"source", "quote"}:
                raise ValueError("Invalid evidence fields")

            require_text(evidence["quote"], "evidence quote", 2000)
            if evidence["source"] == "abstract":
                continue
            if evidence["source"] != "inspection" or inspection.get("status") != "ok":
                raise ValueError("Evidence source was not retrieved")

    def _execute_tool(self, decision: dict[str, Any]) -> dict[str, Any]:
        action = decision["action"]

        if action == "expand_shortlist":
            return self._expand_shortlist()

        if action == "inspect_paper":
            return self._inspect_paper(decision["paper_id"], decision["question"])

        if action == "search_candidates":
            return self._search_candidates(decision["query"], decision["limit"])

        if action == "find_code":
            return self._find_code(decision["paper_id"])

        if action == "analyze_twitter":
            return self._analyze_twitter(decision["paper_id"])

        raise ValueError(f"Unknown tool: {action}")

    def _expand_shortlist(self) -> dict[str, Any]:
        progress = self.state.get("expansion_in_progress")

        if progress is None:
            if limit_reached(self.state["expansions"], self.config.max_expansions):
                return {
                    "status": "budget_exhausted",
                    "reason": "No expansion calls remain",
                }

            screened_ids = list(self.state["screened"])
            candidate_limit = self._candidate_limit()

            if len(screened_ids) >= candidate_limit:
                return {
                    "status": "exhausted",
                    "reason": "No more eligible candidates within the configured bound",
                }

            self.state["expansions"] += 1
            progress = {
                "before": screened_ids,
                "target": min(
                    len(screened_ids) + self.config.expansion_size,
                    candidate_limit,
                ),
            }
            self.state["expansion_in_progress"] = progress
            self._persist_run()

        self._screen_candidates(progress["target"])
        new_ids = set(self.state["screened"]) - set(progress["before"])
        observation = {
            "status": "ok",
            "new_ids": sorted(new_ids),
            "screened_total": len(self.state["screened"]),
        }
        observation_key = f"expand:{self.state['expansions']}"
        self.state["observations"][observation_key] = observation
        self.state.pop("expansion_in_progress", None)

        return observation

    def _document(self, paper: Paper) -> dict[str, Any]:
        key = EXTRACTOR_VERSION + ":" + paper.fingerprint
        document = self.store.cached_inspection(key)
        if document is None:
            document = self.arxiv.document(paper)
            self.store.save_inspection(key, document)
        return document

    def _inspect_paper(self, paper_id: str, question: str) -> dict[str, Any]:
        key = paper_id + ":" + hashlib.sha256(question.encode()).hexdigest()
        queries = self.state.setdefault("inspection_queries", {})
        if key in queries:
            return queries[key]
        if limit_reached(self.state["inspections"], self.config.max_inspections):
            return {
                "status": "budget_exhausted",
                "reason": "No inspection calls remain",
            }

        self.state["inspections"] += 1
        self._persist_run()
        observation = retrieve_passages(self._document(self._paper(paper_id)), question)
        queries[key] = observation
        observation_key = f"inspect:{paper_id}"
        previous = self.state["observations"].get(observation_key, {})
        by_chunk = {item["chunk_id"]: item for item in previous.get("passages", [])}
        by_chunk.update(
            {item["chunk_id"]: item for item in observation.get("passages", [])}
        )
        passages = list(by_chunk.values())
        self.state["observations"][observation_key] = {
            **observation,
            "status": "ok" if passages else observation["status"],
            "passages": passages,
            "text": "\n\n".join(item["text"] for item in passages),
            "questions": [*previous.get("questions", []), question],
        }
        return observation

    def _find_code(self, paper_id: str) -> dict[str, Any]:
        observation_key = f"code:{paper_id}"
        if observation_key in self.state["observations"]:
            return self.state["observations"][observation_key]
        paper = self._paper(paper_id)
        key = CODE_FINDER_VERSION + ":" + paper.fingerprint
        cached = self.store.cached_inspection(key)
        if cached is not None:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(
                cached["result"]["collected_at"]
            )
            if not 0 <= age.total_seconds() < 21_600:
                cached = None
        if cached is None:
            self._persist_run({"type": "code_search_request", "paper_id": paper_id})
            result = self.code_finder.find(paper, self._document(paper))
            if result["status"] != "unavailable":
                self.store.save_inspection(key, {"status": "ok", "result": result})
        else:
            result = cached["result"]
        self.state["observations"][observation_key] = result
        return result

    def _search_candidates(self, query: str, limit: int) -> dict[str, Any]:
        query_key = hashlib.sha256(json.dumps([query, limit]).encode()).hexdigest()
        observation_key = "search:" + query_key
        if observation_key in self.state["observations"]:
            return self.state["observations"][observation_key]
        progress = self.state.get("search_in_progress")
        if progress is None:
            if limit_reached(self.state["expansions"], self.config.max_expansions):
                return {
                    "status": "budget_exhausted",
                    "reason": "No search/expansion calls remain",
                }
            ranked = bm25(
                [candidate.paper for candidate in self._corpus_candidates()], query
            )
            hits = [candidate for candidate in ranked if candidate.score > 0][:limit]
            remaining = self._candidate_limit() - len(self.state["admitted_ids"])
            new_ids = [
                candidate.paper.paper_id
                for candidate in hits
                if candidate.paper.paper_id not in self.state["admitted_ids"]
            ][:remaining]
            self.state["admitted_ids"].extend(new_ids)
            self.state["expansions"] += 1
            progress = {
                "query_key": query_key,
                "query": query,
                "new_ids": new_ids,
                "matches": [
                    {"paper_id": item.paper.paper_id, "score": item.score}
                    for item in hits
                ],
                "target": len(self.state["admitted_ids"]),
            }
            self.state["search_in_progress"] = progress
            self._persist_run()
        elif progress["query_key"] != query_key:
            raise ValueError(
                "Resume the pending focused search before starting another"
            )
        self._screen_candidates(progress["target"])
        by_id = {
            candidate.paper.paper_id: candidate.paper
            for candidate in self._corpus_candidates()
        }
        matches = [
            {
                **hit,
                "title": by_id[hit["paper_id"]].title,
                "abstract_excerpt": by_id[hit["paper_id"]].abstract[:800],
                "screened": hit["paper_id"] in self.state["screened"],
                "assessment": self.state["screened"].get(hit["paper_id"]),
            }
            for hit in progress["matches"]
        ]
        observation = {
            "status": "ok" if matches else "no_matches",
            "query": query,
            "new_ids": progress["new_ids"],
            "matches": matches,
            "remaining_admission_slots": self._candidate_limit()
            - len(self.state["admitted_ids"]),
            "note": "Query-specific scores over the fixed run corpus. Only screened IDs are selectable.",
        }
        self.state["observations"][observation_key] = observation
        self.state.pop("search_in_progress", None)
        return observation

    def _analyze_twitter(self, paper_id: str) -> dict[str, Any]:
        if not self.config.x_enabled:
            raise ValueError("X analysis is disabled for this run (--no-X)")

        self._collect_x_attention(paper_id)
        observation = attention_summary(self.state["x_attention"][paper_id])
        self.state["observations"][f"twitter:{paper_id}"] = observation
        return observation

    def _collect_x_attention(self, requested_id: str | None = None) -> None:
        if not self.config.x_enabled:
            return

        observations = self.state.setdefault("x_attention", {})
        for candidate in self._admitted_candidates():
            paper = candidate.paper
            assessment = self.state["screened"].get(paper.paper_id)
            if assessment is None or paper.paper_id in observations:
                continue
            if requested_id is not None:
                if paper.paper_id != requested_id:
                    continue
            elif assessment["label"] == "irrelevant":
                continue

            cache_key = hashlib.sha256(
                json.dumps(
                    [
                        COLLECTOR_VERSION,
                        paper.fingerprint,
                        self.config.date_from,
                        self.config.date_to,
                    ]
                ).encode()
            ).hexdigest()
            observation = self.store.cached_x_attention(cache_key)
            self._persist_run(
                {
                    "type": "x_attention_request",
                    "paper_id": paper.paper_id,
                    "cached": observation is not None,
                }
            )
            if observation is None:
                observation = self.x_client.collect(
                    paper, self.config.date_from, self.config.date_to
                )
                if observation["status"] in {"ok", "no_matches"}:
                    self.store.save_x_attention(cache_key, observation)
            observations[paper.paper_id] = observation
            self._persist_run(
                {"type": "x_attention_result", **attention_summary(observation)}
            )

    def _featured_x_paper_id(self) -> str | None:
        if not self.config.x_enabled:
            return None

        eligible = {
            paper_id: observation
            for paper_id, observation in self.state.get("x_attention", {}).items()
            if self.state["screened"][paper_id]["label"] in {"relevant", "uncertain"}
        }
        return featured_paper(eligible)

    def _build_result(
        self, status: str, papers: list[dict[str, Any]], limitations: list[str]
    ) -> dict[str, Any]:
        # Always disclose unavailable/truncated evidence even if the LLM omitted it.
        notes = list(limitations)
        attention = self.state.get("x_attention", {})
        if attention:
            unavailable = sum(
                item["status"] == "unavailable" for item in attention.values()
            )
            notes.append(
                "X attention is based on sampled Latest searches among checked relevant/uncertain papers, not all papers in your field. Engagement counts are snapshots at collection time."
            )
            if unavailable:
                reasons = list(
                    dict.fromkeys(
                        error
                        for observation in attention.values()
                        if observation["status"] == "unavailable"
                        for error in observation.get("errors", [])
                    )
                )
                notes.append(
                    f"X activity was unavailable for {unavailable} checked papers. "
                    + "Reported errors: "
                    + "; ".join(reasons[:3])
                )
            if not self._featured_x_paper_id():
                notes.append(
                    "No X feature was selected because no verified matching posts were collected."
                )

        for key, observation in self.state["observations"].items():
            if (
                key.startswith("inspect:")
                and observation.get("status") == "no_relevant_passages"
            ):
                notes.append(
                    f"{key}: no passages matching the inspection question were retrieved; this does not establish absence of the requested experiment."
                )
            elif key.startswith("inspect:") and observation.get("status") != "ok":
                notes.append(
                    f"{key}: full-text inspection unavailable; only abstract evidence was available."
                )
            elif key.startswith("inspect:") and observation.get("truncated"):
                notes.append(
                    f"{key}: only question-relevant HTML passages were inspected, not the whole paper."
                )

        return {
            "run_id": self.run_id,
            "status": status,
            "user_id": self.profile.user_id,
            "profile_fingerprint": self.profile.fingerprint,
            "pipeline_version": PIPELINE_VERSION,
            "categories": self.config.categories,
            "date_from": self.config.date_from,
            "date_to": self.config.date_to,
            "papers": papers,
            "x_feature": (
                self._build_x_feature() if status == "completed" and papers else None
            ),
            "limitations": list(dict.fromkeys(notes)),
            "stats": {
                key: self.state.get(key, 0)
                for key in (
                    "fetched_count",
                    "candidate_count",
                    "llm_calls",
                    "steps",
                    "inspections",
                    "expansions",
                )
            }
            | {
                "screened": len(self.state["screened"]),
                "usage": summarize_usage(self.state.get("llm_usage", [])),
                "tool_calls": self.state.get("tool_calls", 0),
                "tool_latency_seconds": self.state.get("tool_latency_seconds", 0.0),
            },
        }

    def _build_x_feature(self) -> dict[str, Any] | None:
        paper_id = self._featured_x_paper_id()
        if paper_id is None:
            return None
        observation = self.state["x_attention"][paper_id]
        return {
            "paper_id": paper_id,
            "attention": attention_summary(observation),
            "top_posts": [
                {key: post[key] for key in ("url", "author", "text", "metrics")}
                for post in observation["posts"][:5]
            ],
            "checked_papers": sum(
                assessment["label"] in {"relevant", "uncertain"}
                for assessment in self.state["screened"].values()
            ),
            "ranking": "Distinct posting accounts, then reposts, then likes; sampled results only.",
        }

    def _restore_run(self, run_id: str) -> dict[str, Any] | None:
        self.run_id = run_id
        previous_status, self.state, completed_result = self.store.load_run(run_id)

        if completed_result is not None:
            return completed_result

        if self.state["pipeline_version"] != PIPELINE_VERSION:
            raise ValueError("Cannot resume a run created by another pipeline version")

        # Runs created before --new existed always excluded recommendation history.
        self.state["config"].setdefault("new_only", True)
        self.config = DigestConfig(**self.state["config"])
        self.config.validate()
        self.profile = UserProfile.from_dict(self.state["profile"])
        if self.state.get("x_collector") != COLLECTOR_VERSION:
            self.state["x_attention"] = {}
            self.state["x_collector"] = COLLECTOR_VERSION
            self.state["observations"] = {
                key: value
                for key, value in self.state["observations"].items()
                if not key.startswith("twitter:")
            }
        self._persist_run({"type": "run_started", "previous_status": previous_status})

        return None

    def _prepare_candidates(self) -> None:
        if self.state["phase"] == "harvest":
            self._harvest_and_rank_candidates()

        if self.state["phase"] == "screen":
            self._screen_candidates(self.state["screen_target"])
            self.state["phase"] = "react"
            self._persist_run()

    def _harvest_and_rank_candidates(self) -> None:
        papers = collect_metadata(
            self.arxiv,
            self.store,
            self.config.categories,
            self.config.date_from,
            self.config.date_to,
            page_size=self.config.page_size,
            refresh=self.config.refresh_metadata,
        )
        ranked_papers = bm25(papers, self.profile.ranking_text())
        recommended_ids = (
            self.store.recommended(self.profile.user_id)
            if self.config.new_only
            else set()
        )
        eligible_papers = [
            candidate
            for candidate in ranked_papers
            if candidate.paper.paper_id not in recommended_ids
        ]

        self.store.freeze_candidates(self.run_id, eligible_papers)
        self.state.update(
            phase="screen",
            fetched_count=len(papers),
            candidate_count=len(eligible_papers),
        )
        self._persist_run(
            {
                "type": "metadata_complete",
                "fetched": len(papers),
                "eligible": len(eligible_papers),
            }
        )

    def _run_react_loop(self) -> dict[str, Any]:
        while True:
            self._collect_x_attention()
            decision = self.state["pending"]

            if decision is None:
                decision = self._request_next_decision()

                if decision is None:
                    continue

            result = self._process_decision(decision)

            if result is not None:
                return result

    def _request_next_decision(self) -> dict[str, Any] | None:
        if limit_reached(self.state["steps"], self.config.max_steps):
            raise BudgetExhausted("Agent decision-step budget exhausted")

        self.state["steps"] += 1
        self._persist_run()
        decision = self._request_model(
            DECISION_INSTRUCTIONS,
            self._build_decision_context(),
            DECISION_SCHEMA,
            "agent_decision",
        )

        try:
            validate_decision(decision, self.state["screened"])
        except ValueError as error:
            self._record_invalid_action(error, decision)
            return None

        self.state["pending"] = decision
        self._persist_run({"type": "action", "decision": decision})
        return decision

    def _process_decision(self, decision: dict[str, Any]) -> dict[str, Any] | None:
        try:
            validate_decision(decision, self.state["screened"])

            if decision["action"] == "finish":
                selected_papers = self._build_selections(decision)
                return self._complete_run(
                    "completed", selected_papers, decision["limitations"]
                )

            started = time.perf_counter()
            self.state["tool_calls"] = self.state.get("tool_calls", 0) + 1
            try:
                observation = self._execute_tool(decision)
            finally:
                self.state["tool_latency_seconds"] = (
                    self.state.get("tool_latency_seconds", 0.0)
                    + time.perf_counter()
                    - started
                )
                self._persist_run()
            self.state["last_observation"] = {
                "action": decision["action"],
                "result": observation,
            }
            self.state["last_error"] = None
            self.state["pending"] = None
            self._persist_run(
                {
                    "type": "observation",
                    "action": decision["action"],
                    "result": observation,
                }
            )
        except ValueError as error:
            self._record_invalid_action(error, decision)

        return None

    def _record_invalid_action(
        self, error: ValueError, decision: dict[str, Any] | None = None
    ) -> None:
        self.state["last_error"] = str(error)
        self.state["pending"] = None
        event = {"type": "invalid_action", "error": str(error)}
        if decision is not None:
            event["decision"] = decision
        self._persist_run(event)

    def _complete_run(
        self, status: str, papers: list[dict[str, Any]], limitations: list[str]
    ) -> dict[str, Any]:
        result = self._build_result(status, papers, limitations)
        self.store.finish(self.run_id, self.state, result)
        return result
