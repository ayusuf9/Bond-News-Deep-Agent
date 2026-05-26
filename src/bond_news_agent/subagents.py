"""Specialist subagent specifications.

Each subagent is a `deepagents.SubAgent` TypedDict that the main agent can
delegate to via the built-in ``task`` tool. The orchestrator's prompt names
these subagents explicitly, so the ``name`` strings here must stay in sync
with the prompt and the CLI ``--category`` flag.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from deepagents import SubAgent

from .prompts import CATEGORY_DESCRIPTIONS, SPECIALIST_PROMPTS

CATEGORY_NAMES: tuple[str, ...] = (
    "treasuries-rates",
    "ig-corporates",
    "high-yield",
    "sovereign-em",
    "munis",
)


def build_subagents(
    search_tool: Callable[..., Any],
    *,
    only: Sequence[str] | None = None,
) -> list[SubAgent]:
    """Construct ``SubAgent`` specs for each fixed-income specialist.

    Args:
        search_tool: The shared ``search_bond_news`` tool from
            :func:`bond_news_agent.tools.make_news_search_tool`.
        only: Optional iterable of category names to include. When provided,
            unknown names raise ``ValueError``. When omitted, all specialists
            are returned.

    Returns:
        A list of ``SubAgent`` TypedDicts ready to pass to
        ``create_deep_agent(subagents=...)``.
    """
    if only is not None:
        wanted = tuple(only)
        unknown = [n for n in wanted if n not in CATEGORY_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown subagent category/categories: {unknown}. "
                f"Valid options: {list(CATEGORY_NAMES)}"
            )
        names = wanted
    else:
        names = CATEGORY_NAMES

    subagents: list[SubAgent] = []
    for name in names:
        subagents.append(
            SubAgent(
                name=name,
                description=CATEGORY_DESCRIPTIONS[name],
                system_prompt=SPECIALIST_PROMPTS[name],
                tools=[search_tool],
            )
        )
    return subagents
