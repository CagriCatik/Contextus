"""Simple logging helpers."""

from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler

from .console import console


def configure_logging(level: str = "INFO") -> None:
    """Configure a Rich-powered logging formatter for command-line tools."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=False, markup=True)],
        force=True,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "ragstack")
