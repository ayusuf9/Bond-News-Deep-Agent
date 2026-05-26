"""Search-tool factory: validation, retry, normalization, redaction."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError

from bond_news_agent.config import Settings
from bond_news_agent.tools import make_news_search_tool


def _settings() -> Settings:
    return Settings()


def _sample_tavily_response() -> dict[str, Any]:
    return {
        "query": "investment grade corporate bond issuance",
        "answer": "Issuance was robust this week.",
        "results": [
            {
                "title": "IG Issuance Hits Record",
                "url": "https://example.com/article-1",
                "published_date": "2026-05-22",
                "score": 0.95,
                "content": "  Investment-grade issuance hit a record... ",
                "raw_content": None,
            },
            {
                "title": "",
                "url": None,  # should be filtered out
                "score": 0.5,
                "content": "missing url",
            },
        ],
    }


def _invoke_tool(tool: Any, **kwargs: Any) -> Any:
    """Invoke a LangChain @tool with both decorated and undecorated paths."""
    if hasattr(tool, "invoke"):
        return tool.invoke(kwargs)
    return tool(**kwargs)


def test_validates_empty_query(fake_env: None) -> None:
    client = MagicMock()
    tool = make_news_search_tool(_settings(), client=client)
    with pytest.raises(ValueError, match="empty"):
        _invoke_tool(tool, query="   ")
    client.search.assert_not_called()


def test_validates_max_results_range(fake_env: None) -> None:
    client = MagicMock()
    tool = make_news_search_tool(_settings(), client=client)
    with pytest.raises(ValueError, match="max_results"):
        _invoke_tool(tool, query="treasuries", max_results=99, days=7)


def test_validates_days_range(fake_env: None) -> None:
    client = MagicMock()
    tool = make_news_search_tool(_settings(), client=client)
    with pytest.raises(ValueError, match="days"):
        _invoke_tool(tool, query="treasuries", max_results=5, days=999)


def test_normalizes_response(fake_env: None) -> None:
    client = MagicMock()
    client.search.return_value = _sample_tavily_response()
    tool = make_news_search_tool(_settings(), client=client)

    out = _invoke_tool(tool, query="investment grade issuance", max_results=5, days=7)

    assert out["query"] == "investment grade corporate bond issuance"
    assert out["result_count"] == 1  # url=None entry stripped
    only = out["results"][0]
    assert only["title"] == "IG Issuance Hits Record"
    assert only["content"].startswith("Investment-grade issuance")
    assert only["url"] == "https://example.com/article-1"

    client.search.assert_called_once()
    kwargs = client.search.call_args.kwargs
    assert kwargs["topic"] == "news"
    assert kwargs["search_depth"] == "advanced"
    assert kwargs["query"] == "investment grade issuance"


def test_retries_on_transient_error(fake_env: None) -> None:
    client = MagicMock()
    client.search.side_effect = [
        UsageLimitExceededError("rate limited"),
        UsageLimitExceededError("rate limited"),
        _sample_tavily_response(),
    ]
    tool = make_news_search_tool(_settings(), client=client)

    out = _invoke_tool(tool, query="treasuries", max_results=5, days=7)
    assert out["result_count"] == 1
    assert client.search.call_count == 3


def test_no_retry_on_invalid_api_key(fake_env: None) -> None:
    client = MagicMock()
    client.search.side_effect = InvalidAPIKeyError("bad key")
    tool = make_news_search_tool(_settings(), client=client)

    with pytest.raises(InvalidAPIKeyError):
        _invoke_tool(tool, query="treasuries", max_results=5, days=7)
    assert client.search.call_count == 1


def test_secret_redacted_in_error_message(
    fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-supersecretvalue123456")
    client = MagicMock()
    client.search.side_effect = InvalidAPIKeyError(
        "auth failed for tvly-supersecretvalue123456"
    )
    tool = make_news_search_tool(Settings(), client=client)

    with pytest.raises(InvalidAPIKeyError) as exc_info:
        _invoke_tool(tool, query="treasuries", max_results=5, days=7)
    assert "tvly-supersecretvalue123456" not in str(exc_info.value)
    assert "REDACTED" in str(exc_info.value)
