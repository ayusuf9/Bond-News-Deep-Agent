"""Settings validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from bond_news_agent.config import Settings


def test_missing_required_keys_raises() -> None:
    with pytest.raises(ValidationError):
        Settings()


def test_defaults_are_applied(fake_env: None) -> None:
    s = Settings()
    assert s.model_name.startswith("google_genai:")
    assert 1 <= s.max_search_results <= 20
    assert 1 <= s.default_days_back <= 30
    assert s.log_level == "INFO"
    assert s.reports_dir == Path("reports")


def test_overrides_via_env(fake_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOND_NEWS_MODEL_NAME", "google_genai:gemini-3.5-pro")
    monkeypatch.setenv("BOND_NEWS_MAX_SEARCH_RESULTS", "12")
    monkeypatch.setenv("BOND_NEWS_DEFAULT_DAYS_BACK", "14")
    s = Settings()
    assert s.model_name == "google_genai:gemini-3.5-pro"
    assert s.max_search_results == 12
    assert s.default_days_back == 14


def test_model_name_must_have_colon(fake_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOND_NEWS_MODEL_NAME", "no-colon-here")
    with pytest.raises(ValidationError):
        Settings()


def test_max_results_out_of_range_rejected(
    fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BOND_NEWS_MAX_SEARCH_RESULTS", "999")
    with pytest.raises(ValidationError):
        Settings()


def test_days_out_of_range_rejected(
    fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BOND_NEWS_DEFAULT_DAYS_BACK", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_secrets_are_not_in_repr(fake_env: None) -> None:
    s = Settings()
    repr_str = repr(s)
    assert "test-google-key-123456" not in repr_str
    assert "tvly-test-tavily-key-123456" not in repr_str


def test_ensure_reports_dir_creates_path(fake_env: None, tmp_path: Path) -> None:
    custom = tmp_path / "out"
    s = Settings(reports_dir=custom)
    assert not custom.exists()
    resolved = s.ensure_reports_dir()
    assert resolved.exists()
    assert resolved == custom.resolve()
