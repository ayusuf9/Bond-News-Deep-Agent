"""Subagent specs assembly."""

from __future__ import annotations

import pytest

from bond_news_agent.subagents import CATEGORY_NAMES, build_subagents


def _stub_tool() -> object:
    def search_bond_news(query: str) -> dict:  # pragma: no cover - placeholder
        return {"query": query, "results": []}

    return search_bond_news


def test_build_all_subagents() -> None:
    subs = build_subagents(_stub_tool())
    assert {s["name"] for s in subs} == set(CATEGORY_NAMES)
    for s in subs:
        assert s["description"]
        assert s["system_prompt"]
        assert s["tools"]


def test_build_subset() -> None:
    subs = build_subagents(_stub_tool(), only=["ig-corporates", "high-yield"])
    assert [s["name"] for s in subs] == ["ig-corporates", "high-yield"]


def test_unknown_category_raises() -> None:
    with pytest.raises(ValueError, match="Unknown subagent"):
        build_subagents(_stub_tool(), only=["not-a-real-category"])
