"""Embedding utilities with configurable backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Dict, Iterable, Type

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .config import EmbeddingConfig
from .logging import get_logger


class EmbeddingBackend(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self._logger = get_logger(__name__)

    @abstractmethod
    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        """Return embeddings for each item in *texts*."""

    def embed_query(self, text: str) -> np.ndarray:
        """Convenience helper that embeds a single string."""

        return self.embed_documents([text])[0]


class SentenceTransformerEmbeddings(EmbeddingBackend):
    """Wrapper around :class:`SentenceTransformer` with sensible defaults."""

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)
        self._model: SentenceTransformer | None = None
        self._resolved_device: str | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            device = self._resolve_device()
            self._model = SentenceTransformer(self.config.model_name, device=device)
            self._resolved_device = device
            self._logger.info(
                "Loaded SentenceTransformer '%s' on device '%s'", self.config.model_name, device
            )
        return self._model

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors

    def _resolve_device(self) -> str:
        configured = (self.config.device or "auto").strip()
        target = configured.lower()
        if target == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        if target.startswith("cuda") and not torch.cuda.is_available():
            self._logger.warning(
                "CUDA was requested for embeddings but is unavailable. Falling back to CPU."
            )
            return "cpu"

        if target == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                self._logger.warning(
                    "MPS was requested for embeddings but is unavailable. Falling back to CPU."
                )
                return "cpu"

        return configured


_EMBEDDING_BACKENDS: Dict[str, Type[EmbeddingBackend]] = {
    "sentence_transformer": SentenceTransformerEmbeddings,
}


def register_embedding_backend(name: str, backend: Type[EmbeddingBackend]) -> None:
    """Register a custom embedding backend under *name*."""

    _EMBEDDING_BACKENDS[name.lower()] = backend


def _import_backend(identifier: str) -> Type[EmbeddingBackend]:
    module_name, _, class_name = identifier.rpartition(".")
    if not module_name:
        raise ValueError(
            "Custom embedding backends must be provided as a dotted path, e.g. 'pkg.module.CustomClass'."
        )
    module = import_module(module_name)
    backend = getattr(module, class_name)
    if not issubclass(backend, EmbeddingBackend):
        raise TypeError(
            f"Embedding backend '{identifier}' must subclass EmbeddingBackend."
        )
    return backend


def create_embeddings(config: EmbeddingConfig) -> EmbeddingBackend:
    """Instantiate the configured embedding backend."""

    backend_id = (config.backend or "sentence_transformer").strip()
    backend_cls: Type[EmbeddingBackend]
    if backend_id.lower() in _EMBEDDING_BACKENDS:
        backend_cls = _EMBEDDING_BACKENDS[backend_id.lower()]
    else:
        backend_cls = _import_backend(backend_id)
    return backend_cls(config)
