"""Bond-news search tool factory.

A factory function returns a LangChain ``@tool`` that wraps Tavily's news
search. The factory pattern lets us inject a configured ``Settings`` object
(API key, default result count, default recency window) without globals.

Hardening:
- Strict input validation (clear ``ValueError`` for empty / out-of-range args)
- Transient-error retry with exponential backoff via ``tenacity``
- Hard-fail on auth / quota errors (no point retrying)
- Normalized result schema so the LLM always sees the same fields
- Error messages are scrubbed of secrets before propagation
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.tools import tool
from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings
from .logging_utils import get_logger

logger = get_logger("tools")

NewsTopic = Literal["news", "finance"]
"""Tavily topics relevant to bond reporting; ``general`` is intentionally excluded."""

# Errors that indicate a permanent issue (don't retry).
_NON_RETRYABLE: tuple[type[Exception], ...] = (
    InvalidAPIKeyError,
    MissingAPIKeyError,
    BadRequestError,
    ForbiddenError,
    ValueError,
)

# Quota / rate-limit and transient transport issues are worth retrying briefly.
_RETRYABLE: tuple[type[Exception], ...] = (
    UsageLimitExceededError,
    TimeoutError,
    ConnectionError,
)

_MAX_QUERY_LEN = 500
_MIN_RESULTS = 1
_MAX_RESULTS = 20
_MIN_DAYS = 1
_MAX_DAYS = 30


def _validate(query: str, max_results: int, days: int) -> str:
    """Validate and normalize inputs; raise ``ValueError`` on failure."""
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query must not be empty")
    if len(cleaned) > _MAX_QUERY_LEN:
        raise ValueError(f"query must be <= {_MAX_QUERY_LEN} characters")
    if not _MIN_RESULTS <= max_results <= _MAX_RESULTS:
        raise ValueError(
            f"max_results must be between {_MIN_RESULTS} and {_MAX_RESULTS}"
        )
    if not _MIN_DAYS <= days <= _MAX_DAYS:
        raise ValueError(f"days must be between {_MIN_DAYS} and {_MAX_DAYS}")
    return cleaned


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Project Tavily's verbose response down to a stable, LLM-friendly shape."""
    raw_results = raw.get("results") or []
    cleaned = [
        {
            "title": (r.get("title") or "").strip(),
            "url": r.get("url"),
            "published_date": r.get("published_date"),
            "score": r.get("score"),
            "content": (r.get("content") or "").strip(),
            "raw_content": r.get("raw_content"),
        }
        for r in raw_results
        if r.get("url")
    ]
    return {
        "query": raw.get("query"),
        "answer": raw.get("answer"),
        "result_count": len(cleaned),
        "results": cleaned,
    }


def _scrub(message: str, settings: Settings) -> str:
    """Last-line-of-defense secret redaction for error messages."""
    secrets = (
        settings.google_api_key.get_secret_value(),
        settings.tavily_api_key.get_secret_value(),
    )
    redacted = message
    for s in secrets:
        if s and len(s) >= 8 and s in redacted:
            redacted = redacted.replace(s, "***REDACTED***")
    return redacted


def make_news_search_tool(
    settings: Settings,
    *,
    client: TavilyClient | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build the ``search_bond_news`` tool, bound to the given settings.

    Args:
        settings: Application settings (provides API key + defaults).
        client: Optional pre-built ``TavilyClient`` (used for testing).

    Returns:
        A LangChain ``StructuredTool`` ready to be passed into
        ``create_deep_agent(tools=[...])``.
    """
    tavily = client or TavilyClient(api_key=settings.tavily_api_key.get_secret_value())
    default_max_results = settings.max_search_results
    default_days = settings.default_days_back

    @tool("search_bond_news")
    def search_bond_news(
        query: str,
        max_results: int = default_max_results,
        days: int = default_days,
        topic: NewsTopic = "news",
        include_raw_content: bool = False,
    ) -> dict[str, Any]:
        """Search recent fixed-income bond news via Tavily.

        Use this tool to find news articles, press releases, ratings actions,
        issuance announcements, central-bank decisions, default headlines, and
        macro commentary relevant to government, corporate, high-yield,
        sovereign / EM, or municipal bond markets.

        Args:
            query: Specific search query. Prefer narrow, well-formed queries
                (e.g. ``"investment grade corporate bond issuance December"``)
                over broad terms.
            max_results: Number of articles to return (1-20). Smaller values
                are faster and cheaper.
            days: Recency window in days (1-30). The Tavily API will only
                return articles published within this window.
            topic: ``"news"`` (default, broad news index) or ``"finance"``
                (curated financial sources).
            include_raw_content: Set to True only when you need full-article
                text. Defaults to False to keep responses small.

        Returns:
            A dict with ``query``, ``answer``, ``result_count`` and a list of
            ``results`` where each item has ``title``, ``url``,
            ``published_date``, ``score``, ``content`` and (optionally)
            ``raw_content``.
        """
        cleaned_query = _validate(query, max_results, days)

        def _do_search() -> dict[str, Any]:
            return tavily.search(
                query=cleaned_query,
                topic=topic,
                days=days,
                max_results=max_results,
                include_raw_content=include_raw_content,
                search_depth="advanced",
            )

        try:
            for attempt in Retrying(
                reraise=True,
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=8),
                retry=retry_if_exception_type(_RETRYABLE),
            ):
                with attempt:
                    raw = _do_search()
        except _NON_RETRYABLE as e:
            msg = _scrub(str(e), settings)
            logger.warning("search_bond_news non-retryable error: %s", msg)
            raise type(e)(msg) from None
        except RetryError as e:  # pragma: no cover - tenacity reraises original
            msg = _scrub(str(e), settings)
            logger.error("search_bond_news exhausted retries: %s", msg)
            raise RuntimeError(f"Tavily search failed after retries: {msg}") from None
        except Exception as e:
            msg = _scrub(str(e), settings)
            logger.exception("search_bond_news unexpected error")
            raise RuntimeError(f"Tavily search failed: {msg}") from None

        normalized = _normalize(raw)
        logger.info(
            "search_bond_news ok: query=%r results=%d topic=%s days=%d",
            cleaned_query,
            normalized["result_count"],
            topic,
            days,
        )
        return normalized

    return search_bond_news
