"""Configuration and version for a digest pipeline run."""

from dataclasses import dataclass
from datetime import date

from research_digest.retrieval.arxiv_source import categories


PIPELINE_VERSION = "react-v2"

UNLIMITED_LIMIT_FIELDS = (
    "max_candidates",
    "digest_size",
    "max_steps",
    "max_llm_calls",
    "max_inspections",
    "max_expansions",
    "max_context_chars",
)


def limit_reached(used: int, limit: int | None) -> bool:
    return limit is not None and used >= limit


def remaining_budget(used: int, limit: int | None) -> int | None:
    return None if limit is None else limit - used


@dataclass(frozen=True)
class DigestConfig:
    """Run settings. Optional limits accept -1 or None to disable the cap."""

    categories: tuple[str, ...]
    date_from: str
    date_to: str
    model: str | None = None
    shortlist_size: int = 30
    batch_size: int = 10
    expansion_size: int = 10
    max_candidates: int | None = 60
    digest_size: int | None = 10
    max_steps: int | None = 8
    max_llm_calls: int | None = 20
    max_inspections: int | None = 4
    max_expansions: int | None = 3
    page_size: int = 100
    refresh_metadata: bool = False
    max_context_chars: int | None = 160_000
    new_only: bool = False
    x_enabled: bool = True
    variant: str = "react"

    def __post_init__(self) -> None:
        if self.variant == "selection-critic":
            object.__setattr__(self, "x_enabled", False)

        for name in UNLIMITED_LIMIT_FIELDS:
            value = getattr(self, name)

            if type(value) is int and value == -1:
                object.__setattr__(self, name, None)

    def validate(self) -> None:
        categories(self.categories)
        if self.variant not in {"react", "react-explore", "selection-critic"}:
            raise ValueError("Unknown agent variant")

        if type(self.new_only) is not bool:
            raise ValueError("new_only must be a boolean")

        if type(self.x_enabled) is not bool:
            raise ValueError("x_enabled must be a boolean")

        if date.fromisoformat(self.date_from) > date.fromisoformat(self.date_to):
            raise ValueError("date_from must not be after date_to")

        for name in (
            "shortlist_size",
            "batch_size",
            "expansion_size",
            "page_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

        for name in ("max_steps", "max_llm_calls", "max_inspections", "max_expansions"):
            value = getattr(self, name)

            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be nonnegative, or -1 for unlimited")

        for name in ("max_candidates", "digest_size", "max_context_chars"):
            value = getattr(self, name)

            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be positive, or -1 for unlimited")

        if (
            self.max_candidates is not None
            and self.shortlist_size > self.max_candidates
        ):
            raise ValueError("shortlist_size must be <= max_candidates")

        if self.page_size > 1000:
            raise ValueError("page_size must be <= 1000")
