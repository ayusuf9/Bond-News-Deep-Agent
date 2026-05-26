"""Command-line entry point for the bond-news Deep Agent.

Run with::

    bond-news-agent --query "Top fixed-income bond news this week"

Or, equivalently::

    python -m bond_news_agent --query "..."
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agent import run_agent
from .config import Settings
from .logging_utils import configure_logging, get_logger
from .prompts import CATEGORY_DESCRIPTIONS
from .subagents import CATEGORY_NAMES

logger = get_logger("cli")

_EXIT_OK = 0
_EXIT_USAGE = 2
_EXIT_RUNTIME = 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bond-news-agent",
        description=(
            "Fetch and synthesize fixed-income bond news using a Deep Agent "
            "(Google Gemini + Tavily)."
        ),
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help=(
            "Research request (e.g. 'Top fixed-income bond news this week'). "
            "Required unless --list-categories is used."
        ),
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=list(CATEGORY_NAMES),
        help=(
            "Restrict research to one or more specialist categories. "
            "Repeatable. If omitted, all five specialists are wired in."
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=(
            "Override the default Tavily news recency window (1-30 days). "
            "Surfaces in the orchestrator prompt as guidance."
        ),
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Override the default per-search max results (1-20).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write the consolidated report to this path instead of reports/bond_news_<ts>.md.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist the report to disk.",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List the available specialist categories and exit.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Override BOND_NEWS_LOG_LEVEL.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to a .env file. Defaults to .env in the current directory.",
    )
    return parser


def _print_categories(console: Console) -> None:
    table = Table(title="Available specialist subagents")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    for name in CATEGORY_NAMES:
        table.add_row(name, CATEGORY_DESCRIPTIONS[name])
    console.print(table)


def _build_settings(args: argparse.Namespace) -> Settings:
    """Construct settings, applying CLI overrides."""
    overrides: dict[str, object] = {}
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if args.max_results is not None:
        overrides["max_search_results"] = args.max_results
    if args.days is not None:
        overrides["default_days_back"] = args.days

    return Settings(**overrides)


def _augment_query(query: str, args: argparse.Namespace) -> str:
    """Inline CLI hints (days / max_results) into the orchestrator query."""
    hints: list[str] = []
    if args.days is not None:
        hints.append(f"recency window: last {args.days} days")
    if args.max_results is not None:
        hints.append(f"max results per search: {args.max_results}")
    if not hints:
        return query
    return f"{query}\n\n[Operator hints: {'; '.join(hints)}]"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    console = Console()

    if args.env_file is not None:
        load_dotenv(args.env_file, override=False)
    else:
        load_dotenv(override=False)

    if args.list_categories:
        _print_categories(console)
        return _EXIT_OK

    if not args.query or not args.query.strip():
        parser.error("--query is required (or pass --list-categories)")
        return _EXIT_USAGE

    try:
        settings = _build_settings(args)
    except ValidationError as e:
        console.print(
            Panel.fit(
                f"[red]Configuration error[/red]\n\n{e}\n\n"
                "Make sure GOOGLE_API_KEY and TAVILY_API_KEY are set "
                "(see .env.example).",
                title="bond-news-agent",
            )
        )
        return _EXIT_USAGE

    configure_logging(settings.log_level)

    augmented_query = _augment_query(args.query, args)

    console.print(
        Panel.fit(
            f"[bold]Query:[/bold] {args.query}\n"
            f"[bold]Model:[/bold] {settings.model_name}\n"
            f"[bold]Categories:[/bold] {args.category or 'all'}\n"
            f"[bold]Recency:[/bold] {settings.default_days_back} days\n"
            f"[bold]Max results / search:[/bold] {settings.max_search_results}",
            title="bond-news-agent",
        )
    )

    try:
        result = run_agent(
            augmented_query,
            settings=settings,
            only_categories=args.category,
            save=not args.no_save,
            output_path=args.output,
        )
    except ValueError as e:
        console.print(f"[red]Invalid input:[/red] {e}")
        return _EXIT_USAGE
    except RuntimeError as e:
        console.print(f"[red]Runtime error:[/red] {e}")
        return _EXIT_RUNTIME
    except Exception as e:
        logger.exception("Unhandled error in CLI")
        console.print(f"[red]Unexpected error:[/red] {e}")
        return _EXIT_RUNTIME

    if result.chat_summary:
        console.print(Panel(Markdown(result.chat_summary), title="Summary"))

    if result.report_md:
        if not args.no_save and result.report_path:
            console.print(
                f"\n[green]Report saved:[/green] {result.report_path}"
            )
        console.print(Panel(Markdown(result.report_md), title="Bond News Report"))
    else:
        console.print(
            "[yellow]The agent did not produce a final report file. "
            "Check the chat summary above.[/yellow]"
        )

    return _EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
