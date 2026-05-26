"""Application configuration loaded from environment variables / .env file.

All non-secret knobs are namespaced with the ``BOND_NEWS_`` prefix so they
cannot collide with provider SDK environment variables (``GOOGLE_API_KEY``,
``TAVILY_API_KEY``) which the provider SDKs themselves read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Strongly-typed runtime settings.

    Required secrets (``google_api_key``, ``tavily_api_key``) are read from
    their canonical environment variables so that the upstream Google and
    Tavily SDKs continue to work unmodified. All other knobs are prefixed with
    ``BOND_NEWS_`` to avoid namespace pollution.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    google_api_key: SecretStr = Field(
        ...,
        validation_alias="GOOGLE_API_KEY",
        description="Google Gemini API key.",
    )
    tavily_api_key: SecretStr = Field(
        ...,
        validation_alias="TAVILY_API_KEY",
        description="Tavily Search API key.",
    )

    model_name: str = Field(
        default="google_genai:gemini-2.5-pro",
        validation_alias="BOND_NEWS_MODEL_NAME",
        description=(
            "LangChain provider:model identifier resolved by deepagents "
            "(e.g. 'google_genai:gemini-2.5-pro')."
        ),
    )
    max_search_results: int = Field(
        default=8,
        ge=1,
        le=20,
        validation_alias="BOND_NEWS_MAX_SEARCH_RESULTS",
        description="Default max number of Tavily results per search call.",
    )
    default_days_back: int = Field(
        default=7,
        ge=1,
        le=30,
        validation_alias="BOND_NEWS_DEFAULT_DAYS_BACK",
        description="Default Tavily news recency window in days.",
    )
    reports_dir: Path = Field(
        default=Path("reports"),
        validation_alias="BOND_NEWS_REPORTS_DIR",
        description="Directory where consolidated markdown reports are written.",
    )
    recursion_limit: int = Field(
        default=100,
        ge=10,
        le=500,
        validation_alias="BOND_NEWS_RECURSION_LIMIT",
        description="LangGraph recursion limit for the compiled agent.",
    )
    log_level: LogLevel = Field(
        default="INFO",
        validation_alias="BOND_NEWS_LOG_LEVEL",
    )

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, v: str) -> str:
        if ":" not in v or not v.split(":", 1)[1].strip():
            raise ValueError(
                "model_name must be in 'provider:model' format (e.g. "
                "'google_genai:gemini-2.5-pro')"
            )
        return v.strip()

    @field_validator("reports_dir", mode="before")
    @classmethod
    def _coerce_reports_dir(cls, v: object) -> Path:
        if isinstance(v, Path):
            return v
        if isinstance(v, str):
            return Path(v.strip()).expanduser()
        raise TypeError("reports_dir must be a string or Path")

    def ensure_reports_dir(self) -> Path:
        """Create the reports directory if needed and return its absolute path."""
        path = self.reports_dir.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
