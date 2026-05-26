"""Public API for the bond-news Deep Agent."""

from __future__ import annotations

from .agent import BondNewsResult, build_agent, run_agent
from .config import Settings
from .prompts import CATEGORY_DESCRIPTIONS, REPORT_FILENAME
from .subagents import CATEGORY_NAMES, build_subagents
from .tools import make_news_search_tool

__all__ = [
    "CATEGORY_DESCRIPTIONS",
    "CATEGORY_NAMES",
    "REPORT_FILENAME",
    "BondNewsResult",
    "Settings",
    "build_agent",
    "build_subagents",
    "make_news_search_tool",
    "run_agent",
]
