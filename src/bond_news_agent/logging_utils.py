"""Logging configuration with secret redaction.

A single ``configure_logging`` entry point sets up a structured stdlib logger
and installs a filter that scrubs known API keys from every record before it
hits a handler. This is a defense-in-depth measure on top of pydantic's
``SecretStr`` wrapping.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable

_REDACTED = "***REDACTED***"

_ENV_VARS_TO_REDACT: tuple[str, ...] = (
    "GOOGLE_API_KEY",
    "TAVILY_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)

_KEY_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"tvly-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
)


class _SecretRedactor(logging.Filter):
    """Replace known secrets in log message and args with a marker."""

    def __init__(self, env_vars: Iterable[str] = _ENV_VARS_TO_REDACT) -> None:
        super().__init__()
        self._env_vars = tuple(env_vars)

    def _redact(self, value: str) -> str:
        for var in self._env_vars:
            secret = os.environ.get(var)
            if secret and len(secret) >= 8 and secret in value:
                value = value.replace(secret, _REDACTED)
        for pattern in _KEY_PATTERNS:
            value = pattern.sub(_REDACTED, value)
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = self._redact(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: self._redact(v) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                elif isinstance(record.args, tuple):
                    record.args = tuple(
                        self._redact(a) if isinstance(a, str) else a for a in record.args
                    )
        except Exception:
            # Logging must never break the application.
            return True
        return True


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure the root logger for the package and return its logger.

    Idempotent: repeated calls do not stack handlers.
    """
    package_logger = logging.getLogger("bond_news_agent")
    package_logger.setLevel(level)

    if getattr(package_logger, "_bna_configured", False):
        return package_logger

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler.addFilter(_SecretRedactor())
    package_logger.addHandler(handler)
    package_logger.propagate = False
    package_logger._bna_configured = True  # type: ignore[attr-defined]
    return package_logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the package logger."""
    return logging.getLogger(f"bond_news_agent.{name}")
