"""Agent factory + run_agent invocation helper.

These tests patch ``deepagents.create_deep_agent`` so the network is never
touched and no LLM call is made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bond_news_agent import agent as agent_module
from bond_news_agent.agent import BondNewsResult, _find_report, run_agent
from bond_news_agent.prompts import REPORT_FILENAME


def _fake_state(
    report_md: str = "# fake report\n",
    *,
    file_path: str = REPORT_FILENAME,
) -> dict[str, Any]:
    summary_msg = MagicMock()
    summary_msg.content = "Done. Wrote report to bond_news_report.md."
    return {
        "messages": [summary_msg],
        "files": {
            file_path: {
                "content": report_md,
                "encoding": "utf-8",
            }
        },
    }


def test_run_agent_persists_report(
    fake_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_compiled = MagicMock()
    fake_compiled.invoke.return_value = _fake_state()
    monkeypatch.setattr(agent_module, "create_deep_agent", lambda **_: fake_compiled)

    monkeypatch.setenv("BOND_NEWS_REPORTS_DIR", str(tmp_path / "out"))

    result = run_agent("Top fixed-income bond news this week")

    assert isinstance(result, BondNewsResult)
    assert result.report_md.startswith("# fake report")
    assert result.report_path is not None
    assert result.report_path.exists()
    assert result.report_path.read_text(encoding="utf-8") == result.report_md
    assert "bond_news_report.md" in result.chat_summary

    fake_compiled.invoke.assert_called_once()
    invoke_kwargs = fake_compiled.invoke.call_args
    payload = invoke_kwargs.args[0]
    assert payload["messages"][0]["role"] == "user"


def test_run_agent_validates_query(
    fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_module, "create_deep_agent", lambda **_: MagicMock())
    with pytest.raises(ValueError, match="empty"):
        run_agent("   ")


def test_run_agent_unknown_category(
    fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(agent_module, "create_deep_agent", lambda **_: MagicMock())
    with pytest.raises(ValueError, match="Unknown categories"):
        run_agent("treasuries", only_categories=["does-not-exist"])


def test_run_agent_no_save(
    fake_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_compiled = MagicMock()
    fake_compiled.invoke.return_value = _fake_state("# no save\n")
    monkeypatch.setattr(agent_module, "create_deep_agent", lambda **_: fake_compiled)

    monkeypatch.setenv("BOND_NEWS_REPORTS_DIR", str(tmp_path / "out"))

    result = run_agent("treasuries", save=False)
    assert result.report_md == "# no save\n"
    assert result.report_path is None
    assert not (tmp_path / "out").exists()


def test_run_agent_handles_invocation_failure(
    fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_compiled = MagicMock()
    fake_compiled.invoke.side_effect = RuntimeError("boom")
    monkeypatch.setattr(agent_module, "create_deep_agent", lambda **_: fake_compiled)

    with pytest.raises(RuntimeError, match="Agent invocation failed"):
        run_agent("treasuries")


def test_run_agent_missing_report_file_does_not_save(
    fake_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_compiled = MagicMock()
    fake_compiled.invoke.return_value = {
        "messages": [MagicMock(content="No report written.")],
        "files": {},
    }
    monkeypatch.setattr(agent_module, "create_deep_agent", lambda **_: fake_compiled)
    monkeypatch.setenv("BOND_NEWS_REPORTS_DIR", str(tmp_path / "out"))

    result = run_agent("treasuries")
    assert result.report_md == ""
    assert result.report_path is None


def test_find_report_exact_match() -> None:
    matched, content = _find_report({REPORT_FILENAME: "# r"})
    assert matched == REPORT_FILENAME
    assert content == "# r"


def test_find_report_absolute_path() -> None:
    matched, content = _find_report({"/" + REPORT_FILENAME: "# abs"})
    assert matched == "/" + REPORT_FILENAME
    assert content == "# abs"


def test_find_report_nested_path() -> None:
    matched, content = _find_report({f"reports/2026/{REPORT_FILENAME}": "# nested"})
    assert matched == f"reports/2026/{REPORT_FILENAME}"
    assert content == "# nested"


def test_find_report_prefers_exact_over_basename() -> None:
    files = {
        "/" + REPORT_FILENAME: "# wrong-place",
        REPORT_FILENAME: "# right-place",
    }
    matched, content = _find_report(files)
    assert matched == REPORT_FILENAME
    assert content == "# right-place"


def test_find_report_missing_returns_none() -> None:
    matched, content = _find_report({"unrelated.md": "noise"})
    assert matched is None
    assert content == ""


@pytest.mark.parametrize(
    "virtual_path",
    [
        REPORT_FILENAME,
        "/" + REPORT_FILENAME,
        "./" + REPORT_FILENAME,
        f"agent_workspace/{REPORT_FILENAME}",
    ],
)
def test_run_agent_recovers_report_from_any_path(
    fake_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    virtual_path: str,
) -> None:
    """Regression: model may write the file under a leading slash, ./, or
    a nested directory. The extractor should still find it by basename."""
    fake_compiled = MagicMock()
    fake_compiled.invoke.return_value = _fake_state(
        f"# from {virtual_path}\n", file_path=virtual_path
    )
    monkeypatch.setattr(agent_module, "create_deep_agent", lambda **_: fake_compiled)
    monkeypatch.setenv("BOND_NEWS_REPORTS_DIR", str(tmp_path / "out"))

    result = run_agent("treasuries")
    assert result.report_path is not None
    assert result.report_path.exists()
    assert result.report_md == f"# from {virtual_path}\n"


def test_build_agent_passes_expected_kwargs(
    fake_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(agent_module, "create_deep_agent", _capture)
    agent_module.build_agent(agent_module.Settings())

    assert captured["model"].startswith("google_genai:")
    assert captured["system_prompt"]
    assert len(captured["subagents"]) == 5
    assert len(captured["tools"]) == 1
