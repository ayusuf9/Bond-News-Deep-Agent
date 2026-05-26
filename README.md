# Bond News Deep Agent

A production grade agentic workflow that fetches and synthesizes **fixed-income bond news**. A main orchestrator delegates to five specialist subagents (treasuries / rates, IG corporates, high-yield, sovereign & EM, munis), then writes a consolidated markdown report.

The agent is **model-agnostic**: it ships defaulting to Google Gemini, but works with any LangChain-supported chat model (OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, Vertex AI, OpenRouter, Fireworks, Ollama, …). See [Model selection](#model-selection) below.

## Architecture

```
                    ┌────────────────────────────────────────┐
   user query ────► │   Main Deep Agent (orchestrator LLM)   │
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
- A Tavily API key: <https://app.tavily.com/>
- An API key for **one** LLM provider (defaults to Google Gemini — see [Model selection](#model-selection) for alternatives)

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

## Model selection

The agent passes its `model_name` straight through to LangChain's [`init_chat_model`](https://python.langchain.com/docs/integrations/chat/), so any provider supported there works here. Pick one of the following patterns; you only need an API key for the provider you choose.

### Option A: switch via `.env` (no code changes)

```bash
# .env
TAVILY_API_KEY=tvly-...

# Pick ONE provider key:
GOOGLE_API_KEY=AIza...                  # Gemini (default)
# OPENAI_API_KEY=sk-...                 # OpenAI
# ANTHROPIC_API_KEY=sk-ant-...          # Anthropic
# OPENROUTER_API_KEY=sk-or-...          # OpenRouter (proxy to many providers)

# And select the model identifier (provider:model format):
BOND_NEWS_MODEL_NAME=google_genai:gemini-2.5-pro
# BOND_NEWS_MODEL_NAME=openai:gpt-4o
# BOND_NEWS_MODEL_NAME=anthropic:claude-sonnet-4-6
# BOND_NEWS_MODEL_NAME=openrouter:anthropic/claude-sonnet-4-6
# BOND_NEWS_MODEL_NAME=azure_openai:gpt-4o
# BOND_NEWS_MODEL_NAME=bedrock:anthropic.claude-sonnet-4
# BOND_NEWS_MODEL_NAME=ollama:llama3.1:70b
```

You may also need to install the corresponding LangChain integration package (e.g. `pip install "langchain[openai]"` or `pip install "langchain[anthropic]"`). The default install only includes `langchain-google-genai`.

### Option B: pass an initialized model instance

For full control over temperature, max tokens, base URLs, retries, etc., build the model yourself and inject it:

```python
from langchain_anthropic import ChatAnthropic
from bond_news_agent import Settings, build_agent, run_agent

model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.2, max_tokens=8000)
settings = Settings(model_name="anthropic:claude-sonnet-4-6")  # placeholder; ignored when model is passed

# Edit src/bond_news_agent/agent.py:build_agent to accept an injected `model=`
# (one-line change), or call create_deep_agent directly with model=model.
```

### Option C: use a self-hosted / private model

Same pattern — point at Ollama, vLLM, or LM Studio:

```bash
BOND_NEWS_MODEL_NAME=ollama:llama3.1:70b   # requires Ollama running locally
```

For air-gapped / on-prem deployments, swap Gemini for Vertex AI (`google_vertexai:...`) so prompt data stays in your own GCP tenant.

### Picking a model

The orchestrator + 5 specialists need **strong tool-calling** and **instruction following**. Verified to work well: `google_genai:gemini-2.5-pro`, `openai:gpt-4o` / `gpt-4.1`, `anthropic:claude-sonnet-4-6`. Smaller / faster tiers (`gemini-2.5-flash`, `gpt-4o-mini`, `claude-haiku-4`) work for `--category`-scoped runs but can struggle with the full 5-subagent orchestration on long queries.

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
