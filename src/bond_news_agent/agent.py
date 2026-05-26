"""Compose the Deep Agent and provide a high-level ``run_agent`` invocation."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent

from .config import Settings
from .logging_utils import configure_logging, get_logger
from .prompts import MAIN_PROMPT, REPORT_FILENAME
from .subagents import CATEGORY_NAMES, build_subagents
from .tools import make_news_search_tool

logger = get_logger("agent")

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(slots=True)
class BondNewsResult:
    """Outcome of a single ``run_agent`` invocation.

    Attributes:
        report_md: The final markdown report (empty string if the orchestrator
            did not produce one).
        report_path: Absolute path of the persisted report on disk, or
            ``None`` when ``save=False``.
        chat_summary: The orchestrator's final assistant message (the short
            summary, *not* the full report).
        files: Snapshot of every virtual-FS file the agent wrote, keyed by
            virtual path. Useful for debugging and tests.
    """

    report_md: str
    report_path: Path | None
    chat_summary: str
    files: dict[str, str] = field(default_factory=dict)


def _set_provider_env(settings: Settings) -> None:
    """Push the Google API key into the environment so ``init_chat_model`` can pick it up.

    deepagents resolves ``"google_genai:..."`` via ``init_chat_model``, which
    expects ``GOOGLE_API_KEY`` to be set in the process environment. We use a
    SecretStr in :class:`Settings`, so the user only sees the key once at
    construction time; here we propagate it to the env var the SDK reads.
    Existing env var values take precedence so callers can override.
    """
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key.get_secret_value()
    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key.get_secret_value()


def build_agent(
    settings: Settings,
    *,
    only_categories: Sequence[str] | None = None,
):
    """Build a compiled Deep Agent ready to ``invoke``.

    Args:
        settings: Validated runtime settings.
        only_categories: When provided, only these specialist subagents are
            wired in (must be a subset of ``CATEGORY_NAMES``). Useful for the
            CLI ``--category`` flag.

    Returns:
        A compiled LangGraph state graph.
    """
    _set_provider_env(settings)

    search_tool = make_news_search_tool(settings)
    subagents = build_subagents(search_tool, only=only_categories)

    logger.info(
        "Building bond-news agent: model=%s subagents=%d",
        settings.model_name,
        len(subagents),
    )

    return create_deep_agent(
        model=settings.model_name,
        tools=[search_tool],
        subagents=subagents,
        system_prompt=MAIN_PROMPT,
    )


def _extract_message_text(message: Any) -> str:
    """Extract a human-readable string from any LangChain message-like object."""
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict) and chunk.get("type") == "text":
                parts.append(str(chunk.get("text", "")))
        return "".join(parts)
    return str(content) if content is not None else ""


def _extract_files(state: Any) -> dict[str, str]:
    """Collapse the agent's virtual filesystem state into ``{path: content}``."""
    files = state.get("files") if isinstance(state, dict) else None
    if not files:
        return {}
    out: dict[str, str] = {}
    for path, file_data in files.items():
        if isinstance(file_data, dict):
            content = file_data.get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(c) for c in content)
            out[path] = content if isinstance(content, str) else str(content)
        elif isinstance(file_data, str):
            out[path] = file_data
    return out


def _find_report(files: dict[str, str], target: str = REPORT_FILENAME) -> tuple[str | None, str]:
    """Find the orchestrator's report in the virtual FS, regardless of path style.

    Models occasionally write to relative (``bond_news_report.md``) or absolute
    (``/bond_news_report.md``) paths, sometimes with a leading ``./``. This
    matches by basename so any of those wins. Exact matches are preferred.

    Returns ``(matched_path, content)``. ``matched_path`` is ``None`` when no
    candidate was found.
    """
    if not files:
        return None, ""
    if target in files:
        return target, files[target]
    target_base = target.rsplit("/", 1)[-1]
    for path, content in files.items():
        if path.rsplit("/", 1)[-1] == target_base:
            return path, content
    return None, ""


def _safe_slug(text: str, *, max_len: int = 50) -> str:
    slug = _SAFE_FILENAME_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len] if slug else "report"


def _build_output_path(reports_dir: Path, query: str, override: Path | None) -> Path:
    """Compute the on-disk path for the persisted report."""
    if override is not None:
        path = override.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"bond_news_{timestamp}_{_safe_slug(query)}.md"
    return reports_dir / name


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query must not be empty or whitespace")
    if len(cleaned) > 2_000:
        raise ValueError("query must be <= 2000 characters")
    return cleaned


def run_agent(
    query: str,
    settings: Settings | None = None,
    *,
    only_categories: Iterable[str] | None = None,
    save: bool = True,
    output_path: Path | str | None = None,
) -> BondNewsResult:
    """Run a bond-news research request end-to-end.

    Args:
        query: Natural-language research request from the user.
        settings: Optional pre-built settings; constructed from env if omitted.
        only_categories: Optional iterable of specialist names to enable.
        save: Persist the consolidated report to disk under ``reports_dir``.
        output_path: Override the on-disk path for the report.

    Returns:
        A :class:`BondNewsResult` with the report, on-disk path, and chat
        summary.

    Raises:
        ValueError: If ``query`` is empty or ``only_categories`` is invalid.
        RuntimeError: If the agent invocation itself fails.
    """
    cleaned_query = _validate_query(query)
    settings = settings or Settings()
    configure_logging(settings.log_level)

    cats = tuple(only_categories) if only_categories is not None else None
    if cats is not None:
        unknown = [c for c in cats if c not in CATEGORY_NAMES]
        if unknown:
            raise ValueError(
                f"Unknown categories: {unknown}. Valid: {list(CATEGORY_NAMES)}"
            )

    agent = build_agent(settings, only_categories=cats)

    invocation = {"messages": [{"role": "user", "content": cleaned_query}]}
    config = {"recursion_limit": settings.recursion_limit}

    logger.info("Invoking bond-news agent (categories=%s)", cats or "all")
    try:
        state = agent.invoke(invocation, config=config)
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise RuntimeError(f"Agent invocation failed: {e}") from e

    files = _extract_files(state)
    matched_path, report_md = _find_report(files)
    if matched_path and matched_path != REPORT_FILENAME:
        logger.info(
            "Located report at virtual path %r (expected %r)", matched_path, REPORT_FILENAME
        )

    messages = state.get("messages", []) if isinstance(state, dict) else []
    chat_summary = _extract_message_text(messages[-1]) if messages else ""

    report_path: Path | None = None
    if save and report_md:
        reports_dir = settings.ensure_reports_dir()
        path_override = Path(output_path).expanduser() if output_path else None
        report_path = _build_output_path(reports_dir, cleaned_query, path_override)
        report_path.write_text(report_md, encoding="utf-8")
        logger.info("Saved bond-news report to %s", report_path)
    elif save and not report_md:
        logger.warning(
            "Agent did not write %s (or any file with that basename); "
            "nothing to save. Files seen: %s. Inspect chat_summary.",
            REPORT_FILENAME,
            sorted(files.keys()) or "<none>",
        )

    return BondNewsResult(
        report_md=report_md,
        report_path=report_path,
        chat_summary=chat_summary,
        files=files,
    )
