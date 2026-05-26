# Bond News Deep Agent

A production grade agentic workflow that fetches and synthesizes **fixed-income bond news** using Google Gemini and Tavily. A main orchestrator delegates to five specialist subagents (treasuries / rates, IG corporates, high-yield, sovereign & EM, munis), then writes a consolidated markdown report.

## Architecture

```
                    ┌────────────────────────────────────────┐
   user query ────► │  Main Deep Agent (orchestrator)│
                    └──┬──────────────────────────────────┬──┘
                       │ delegate                          │ write_file
                       ▼                                   ▼
   ┌────────────────────────────────────────┐    reports/bond_news_*.md
   │ subagents:                             │
   │  • treasuries-rates                    │
   │  • ig-corporates                       │
   │  • high-yield                          │
   │  • sovereign-em                        │
   │  • munis                               │
   └────────────┬───────────────────────────┘
                │ search_bond_news (Tavily, topic="news")
                ▼
        ┌──────────────────┐
        │   Tavily News    │
        └──────────────────┘
```

## Prerequisites

- Python 3.10+
- A Google Gemini API key: <https://aistudio.google.com/app/apikey>
- A Tavily API key: <https://app.tavily.com/>

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Configure

```bash
cp .env.example .env
```

All other settings are optional and documented inline in `.env.example`.

## Run

### CLI

```bash
# Top-level fixed-income headlines for the last 7 days
bond-news-agent --query "Top fixed-income bond news this week" --days 7

# Drill into a single specialist (skip orchestrator delegation)
bond-news-agent --query "Latest IG corporate issuance" --category ig-corporates

# Save report to a specific path
bond-news-agent --query "EM sovereign distress watchlist" --output reports/em.md

# List available subagent categories
bond-news-agent --list-categories
```

### Library

```python
from bond_news_agent import Settings, run_agent

settings = Settings()  # loads from env / .env
result = run_agent(
    "Summarize this week's investment-grade corporate bond news",
    settings,
)
print(result.report_md)
print(f"Saved to: {result.report_path}")
```

## How it works

The agent uses the [Deep Agents harness](https://docs.langchain.com/oss/python/deepagents/customization) which provides:

1. **Planning** – the orchestrator drafts a todo list before fetching.
2. **Tools** – a single Tavily-backed `search_bond_news` tool (validated, retried, normalized).
3. **Subagents** – five specialist agents to parallelize and isolate context.
4. **Virtual filesystem** – the orchestrator writes the final report to a virtual file, which `run_agent` extracts and persists to `reports/`.

## Disclaimer

This tool is for **informational and research purposes only** and does **not** constitute investment advice. Always verify primary sources and consult a licensed professional before making investment decisions.

## Development

```bash
ruff check src tests
pytest
```

## Project layout

```
src/bond_news_agent/
├── __init__.py        # public API
├── __main__.py        # `python -m bond_news_agent`
├── agent.py           # build_agent / run_agent
├── cli.py             # CLI entry point
├── config.py          # Settings (pydantic-settings)
├── logging_utils.py   # structured logging + secret redaction
├── prompts.py         # main + subagent prompts
├── subagents.py       # specialist subagent specs
└── tools.py           # Tavily-backed search tool factory
```
