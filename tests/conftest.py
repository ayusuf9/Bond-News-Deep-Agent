"""Shared fixtures: ensure test runs never touch the real .env file."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run every test from a clean tmp directory with controlled env vars.

    - chdir into a tmp directory so `.env` discovery never picks up a real file
    - clear all known secret-bearing env vars
    """
    monkeypatch.chdir(tmp_path)
    for key in (
        "GOOGLE_API_KEY",
        "TAVILY_API_KEY",
        "BOND_NEWS_MODEL_NAME",
        "BOND_NEWS_MAX_SEARCH_RESULTS",
        "BOND_NEWS_DEFAULT_DAYS_BACK",
        "BOND_NEWS_REPORTS_DIR",
        "BOND_NEWS_RECURSION_LIMIT",
        "BOND_NEWS_LOG_LEVEL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate the minimum required env vars with safe placeholders."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key-123456")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-tavily-key-123456")
