"""Additional retrieval tools and instructions for Agent 2."""

from collections import Counter
from copy import deepcopy
from typing import Any

from .protocol import DECISION_SCHEMA, DECISION_INSTRUCTIONS, TOOLS as BASE_TOOLS
from research_digest.retrieval.ranking import bm25, tokens

TOOLS = [
    {
        "name": "get_candidate_overview",
        "arguments": {},
        "description": "Return corpus, screened and unseen counts, screening-label counts and IDs, and lexical coverage of profile topics. Topic matches are keyword overlap, not relevance judgments. No model call or expansion budget is consumed.",
    },
    {
        "name": "search_similar",
        "arguments": {"paper_id": "screened seed paper ID"},
        "description": "Use the seed title and abstract as a BM25 query over unseen papers in the fixed eligible corpus. Admit and screen up to expansion_size matches, subject to max_candidates. Shares the search/expansion budget. Repeated calls for the same seed return the cached result. No embeddings or web search; lexical similarity does not imply relevance.",
    },
]

SCHEMA = deepcopy(DECISION_SCHEMA)
SCHEMA["properties"]["action"]["enum"].extend(tool["name"] for tool in TOOLS)
INSTRUCTIONS = (
    " Additional tools: get_candidate_overview can reveal unselected relevant IDs "
    "and unseen topic coverage; use it when deciding whether further retrieval is useful. "
    "Use search_similar with a screened paper_id when a promising paper suggests "
    "related unseen candidates. It uses the seed title and abstract, so query and "
    "question must be empty. For get_candidate_overview, paper_id, query and question "
    "must all be empty. Set limit to 10 for both tools. "
    "Lexical coverage and similarity are not evidence of personal relevance. "
    "Tool examples illustrate situations and arguments, not complete decision JSON. "
    "Replace placeholder IDs with actual screened IDs and follow the decision schema."
)


# Keep Agent 1 unchanged while removing its tool-restraint guidance for Agent 2.
TOOL_RESTRAINT_GUIDANCE = (
    "Use a tool when it can resolve that uncertainty. ",
    "Finish when further available actions are unlikely to change the supported "
    "selection, or disclose exhausted budgets. ",
    "Do not call tools merely to increase the call count. ",
    "If repeated inspections or searches return the same evidence, do not keep "
    "paraphrasing the same request; disclose the uncertainty, consider another "
    "candidate or finish. ",
)

DECISION_PROMPT = DECISION_INSTRUCTIONS
for guidance in TOOL_RESTRAINT_GUIDANCE:
    DECISION_PROMPT = DECISION_PROMPT.replace(guidance, "")
DECISION_PROMPT = DECISION_PROMPT.replace(
    "original profile-based ranking",
    "saved retrieval order (E5-ranked BM25 top 200, then the BM25 tail)",
)
DECISION_PROMPT += INSTRUCTIONS + (
    " Your first valid finish receives one selection check. If selection_feedback "
    "is supplied, consider its draft and review, revise or investigate as justified, "
    "and then finish. The check is not repeated; its feedback is advisory and may "
    "omit inspection evidence you already obtained."
)


TOOL_EXAMPLES = {
    "inspect_paper": {
        "situation": "The abstract says efficient training, but the profile prioritizes GPU memory. Whether the method reduces memory could change inclusion.",
        "arguments": {
            "paper_id": "<screened_paper_id>",
            "question": "Does the method reduce peak GPU training memory, or only the number of training examples?",
        },
    },
    "search_candidates": {
        "situation": "The shortlist covers low-rank adapters but contains few papers on optimizer-state compression, another profile interest.",
        "arguments": {
            "query": "optimizer state compression memory efficient training",
            "limit": 10,
        },
    },
    "expand_shortlist": {
        "situation": "Several current candidates are weak matches. More papers from the original profile ranking could replace them, and there is no specific new search direction.",
        "arguments": {},
    },
    "find_code": {
        "situation": "The researcher explicitly wants methods they can try. Finding a linked implementation could affect which of two otherwise suitable papers to recommend.",
        "arguments": {"paper_id": "<screened_paper_id>"},
    },
    "analyze_twitter": {
        "situation": "X analysis is enabled and the digest needs an attention indicator for a candidate. Check observed activity without treating popularity as scientific quality.",
        "arguments": {"paper_id": "<screened_paper_id>"},
    },
    "get_candidate_overview": {
        "situation": "Before finishing, it is unclear whether stronger screened candidates were overlooked or profile topics still have many unseen matches.",
        "arguments": {},
    },
    "search_similar": {
        "situation": "A screened paper on low-rank gradient optimization strongly fits the profile. Related unseen papers could improve the digest.",
        "arguments": {"paper_id": "<screened_paper_id>"},
    },
}


def tool_descriptions() -> list[dict[str, Any]]:
    """Add examples without mutating Agent 1's tool definitions."""
    tools = [
        {**deepcopy(tool), "usage_example": deepcopy(TOOL_EXAMPLES[tool["name"]])}
        for tool in BASE_TOOLS + TOOLS
    ]
    for tool in tools:
        if tool["name"] == "expand_shortlist":
            tool["description"] = (
                "Screen the next candidates in the saved retrieval order: E5-ranked "
                "BM25 top 200 first, then the remaining BM25 ranking. "
                "Does not alter dates or categories; uses expansion_size and the shared expansion budget."
            )
    return tools


def candidate_overview(agent: Any) -> dict[str, Any]:
    papers = [item.paper for item in agent._corpus_candidates()]
    screened = agent.state["screened"]
    admitted = set(agent.state["admitted_ids"])
    labels = {
        label: sorted(pid for pid, item in screened.items() if item["label"] == label)
        for label in ("relevant", "uncertain", "irrelevant")
    }
    coverage = []
    for topic in agent.state["profile"]["topics"]:
        query = " ".join([topic["name"], *topic["keywords"]])
        matches = (
            [item.paper.paper_id for item in bm25(papers, query) if item.score > 0]
            if tokens(query)
            else []
        )
        coverage.append(
            {
                "topic": topic["name"],
                "matching_papers": len(matches),
                "screened_matches": sum(pid in screened for pid in matches),
                "unseen_matches": sum(pid not in admitted for pid in matches),
            }
        )
    return {
        "status": "ok",
        "corpus_count": len(papers),
        "screened_count": len(screened),
        "unseen_count": sum(p.paper_id not in admitted for p in papers),
        "admitted_unscreened_count": len(admitted - screened.keys()),
        "remaining_admission_slots": agent._candidate_limit() - len(admitted),
        "screening_counts": dict(Counter(item["label"] for item in screened.values())),
        "screened_ids_by_label": labels,
        "topic_coverage": coverage,
        "note": "Coverage is lexical overlap with topic names/keywords, not ground-truth relevance. Topics may overlap. No human evaluation labels are used.",
    }
