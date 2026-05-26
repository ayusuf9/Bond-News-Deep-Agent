"""CLI parsing + dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bond_news_agent import cli as cli_module
from bond_news_agent.agent import BondNewsResult


def test_list_categories(
    fake_env: None, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli_module.main(["--list-categories"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "treasuries-rates" in out
    assert "munis" in out


def test_query_required(fake_env: None, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_module.main([])
    assert exc.value.code == 2  # argparse error


def test_runs_with_query(
    fake_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run_agent(query: str, **kwargs: Any) -> BondNewsResult:
        captured["query"] = query
        captured["kwargs"] = kwargs
        return BondNewsResult(
            report_md="# Test Report\nA bullet.",
            report_path=tmp_path / "out.md",
            chat_summary="Wrote report.",
            files={"bond_news_report.md": "# Test Report\nA bullet."},
        )

    monkeypatch.setattr(cli_module, "run_agent", _fake_run_agent)

    rc = cli_module.main(
        [
            "--query",
            "Top bond headlines",
            "--category",
            "ig-corporates",
            "--days",
            "5",
            "--max-results",
            "6",
            "--no-save",
        ]
    )

    assert rc == 0
    assert "Top bond headlines" in captured["query"]
    assert captured["kwargs"]["only_categories"] == ["ig-corporates"]
    assert captured["kwargs"]["save"] is False


def test_invalid_category_rejected_by_argparse(
    fake_env: None, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        cli_module.main(["--query", "x", "--category", "not-real"])


def test_runtime_error_returns_nonzero(
    fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_: Any, **__: Any) -> BondNewsResult:
        raise RuntimeError("agent broke")

    monkeypatch.setattr(cli_module, "run_agent", _boom)
    rc = cli_module.main(["--query", "treasuries", "--no-save"])
    assert rc == 1
