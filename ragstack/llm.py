"""Abstract interfaces and shared models for chat providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Sequence


@dataclass
class ModelInfo:
    """Basic metadata describing an available model."""

    name: str


class ModelClientError(RuntimeError):
    """Base exception raised for chat provider errors."""


class ModelConnectionError(ModelClientError):
    """Raised when the provider cannot be reached."""


class ModelNotFoundError(ModelClientError):
    """Raised when the requested model is unavailable."""


class SupportsModels(Protocol):
    """Protocol describing the operations the chat session expects."""

    model: str | None

    def list_models(self) -> List[ModelInfo]:
        ...

    def ensure_model(self, model_name: str, installed_models: Sequence[ModelInfo] | None = None) -> None:
        ...

    def chat(self, messages: Sequence[dict]) -> str:
        ...
