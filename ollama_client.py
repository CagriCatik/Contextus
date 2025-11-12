"""Compatibility wrapper around :mod:`ragstack.ollama`."""

from __future__ import annotations

from typing import Optional

from ragstack.config import OllamaConfig
from ragstack.ollama import (
    OllamaClient as CoreClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaModel,
    OllamaModelNotFoundError,
)


class OllamaClient(CoreClient):
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        config = OllamaConfig()
        if host is not None:
            config.host = host
        if timeout is not None:
            config.timeout = timeout
        super().__init__(config, model=model)


__all__ = [
    "OllamaClient",
    "OllamaClientError",
    "OllamaConnectionError",
    "OllamaModel",
    "OllamaModelNotFoundError",
]
