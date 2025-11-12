"""Vector store abstractions and FAISS-backed implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple, Type, TYPE_CHECKING

import faiss
import numpy as np

from .config import PathConfig
from .logging import get_logger


if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .config import AppConfig


LOGGER = get_logger(__name__)


@dataclass
class VectorRecord:
    text: str
    source_file: str
    chunk_id: int


@dataclass
class IndexMetadata:
    records: List[VectorRecord] = field(default_factory=list)

    def to_json(self) -> List[Dict[str, object]]:
        return [
            {"text": record.text, "source_file": record.source_file, "chunk_id": record.chunk_id}
            for record in self.records
        ]

    @classmethod
    def from_json(cls, payload: Iterable[Dict[str, object]]) -> "IndexMetadata":
        records = [
            VectorRecord(
                text=str(item.get("text", "")),
                source_file=str(item.get("source_file", "")),
                chunk_id=int(item.get("chunk_id", 0)),
            )
            for item in payload
        ]
        return cls(records=records)


class VectorStore(ABC):
    """Common protocol for vector stores."""

    def __init__(self, *, name: str, paths: PathConfig, dim: int | None = None, **_: object) -> None:
        self.name = name
        self.paths = paths
        self.dim = dim

    @abstractmethod
    def add(self, vectors: np.ndarray, records: Sequence[VectorRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: np.ndarray, *, k: int) -> List[Tuple[float, VectorRecord]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        raise NotImplementedError


class FaissVectorStore(VectorStore):
    """Thin wrapper around :class:`faiss.IndexFlatIP` with metadata persistence."""

    def __init__(self, *, name: str, paths: PathConfig, dim: int | None = None, **_: object) -> None:
        super().__init__(name=name, paths=paths, dim=dim)
        self.index_path = self.paths.index_dir / f"{name}.faiss"
        self.meta_path = self.paths.index_dir / f"{name}_meta.json"
        self._legacy_meta_path = self.paths.index_dir / f"{name}.json"
        self.metadata = IndexMetadata()

        if self.index_path.exists():
            self._load()
        else:
            if dim is None:
                raise ValueError("Dimension must be provided when creating a new index")
            self.index = faiss.IndexFlatIP(dim)

    def _load(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self.dim = self.index.d
        meta_path = self.meta_path
        if not meta_path.exists() and self._legacy_meta_path.exists():
            meta_path = self._legacy_meta_path
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.metadata = IndexMetadata.from_json(payload)
        elif self.metadata.records:
            return
        else:
            LOGGER.warning(
                "Metadata for index '%s' was not found. Only similarity scores will be available.",
                self.name,
            )

        records = len(self.metadata.records)
        if records < self.index.ntotal:
            LOGGER.warning(
                "Metadata for index '%s' is missing %s entr%s (metadata=%s, vectors=%s).",
                self.name,
                self.index.ntotal - records,
                "ies" if self.index.ntotal - records != 1 else "y",
                records,
                self.index.ntotal,
            )
        elif records > self.index.ntotal:
            LOGGER.warning(
                "Metadata for index '%s' has %s extra entr%s (metadata=%s, vectors=%s).",
                self.name,
                records - self.index.ntotal,
                "ies" if records - self.index.ntotal != 1 else "y",
                records,
                self.index.ntotal,
            )

    def add(self, vectors: np.ndarray, records: Sequence[VectorRecord]) -> None:
        if vectors.shape[0] != len(records):
            raise ValueError("Number of vectors must match number of records")
        if self.index.d != vectors.shape[1]:
            raise ValueError("Vector dimensionality does not match index")
        self.index.add(vectors.astype("float32"))
        self.metadata.records.extend(records)

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata.to_json(), handle, ensure_ascii=False, indent=2)

    def search(self, query: np.ndarray, *, k: int) -> List[Tuple[float, VectorRecord]]:
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape[1] != self.index.d:
            raise ValueError("Query dimensionality does not match index")
        if self.index.ntotal == 0:
            return []

        query = query.astype("float32")
        scores, indices = self.index.search(query, k)
        results: List[Tuple[float, VectorRecord]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if idx >= len(self.metadata.records):
                LOGGER.warning(
                    "Index '%s' returned result %s but only %s metadata record(s) are available. Skipping entry.",
                    self.name,
                    idx,
                    len(self.metadata.records),
                )
                continue
            results.append((float(score), self.metadata.records[idx]))
        return results

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0


_VECTOR_STORE_BACKENDS: Dict[str, Type[VectorStore]] = {
    "faiss": FaissVectorStore,
}


def register_vector_store(name: str, backend: Type[VectorStore]) -> None:
    """Register a custom vector store backend under *name*."""

    _VECTOR_STORE_BACKENDS[name.lower()] = backend


def _import_vector_store(identifier: str) -> Type[VectorStore]:
    module_name, _, class_name = identifier.rpartition(".")
    if not module_name:
        raise ValueError(
            "Custom vector store backends must be provided as a dotted path, e.g. 'pkg.module.CustomStore'."
        )
    module = import_module(module_name)
    backend = getattr(module, class_name)
    if not issubclass(backend, VectorStore):
        raise TypeError(
            f"Vector store backend '{identifier}' must subclass VectorStore."
        )
    return backend


def create_vector_store(config: "AppConfig", name: str, *, dim: int | None) -> VectorStore:
    """Instantiate the configured vector store backend."""

    backend_id = (config.vector_store.backend or "faiss").strip()
    backend_cls: Type[VectorStore]
    if backend_id.lower() in _VECTOR_STORE_BACKENDS:
        backend_cls = _VECTOR_STORE_BACKENDS[backend_id.lower()]
    else:
        backend_cls = _import_vector_store(backend_id)

    extra_params = dict(config.vector_store.parameters or {})
    if "name" in extra_params or "paths" in extra_params:
        raise ValueError("Vector store parameters cannot override 'name' or 'paths'.")
    if "dim" not in extra_params:
        extra_params["dim"] = dim
    elif dim is not None and extra_params["dim"] is None:
        extra_params["dim"] = dim

    return backend_cls(name=name, paths=config.paths, **extra_params)
