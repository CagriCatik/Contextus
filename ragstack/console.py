"""Rich console utilities shared across command-line tools."""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "green",
        "prompt": "bold green",
    }
)

console = Console(theme=_THEME)

__all__ = ["console"]
