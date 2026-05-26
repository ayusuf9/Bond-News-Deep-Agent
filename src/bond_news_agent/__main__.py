"""Allow ``python -m bond_news_agent`` to invoke the CLI."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
