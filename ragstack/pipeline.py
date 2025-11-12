"""Index building pipeline that ties together corpus, embeddings, and vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import AppConfig
from .corpus import DocumentChunk, MarkdownCorpus
from .embedding import EmbeddingBackend, create_embeddings
from .logging import get_logger
from .store import VectorRecord, VectorStore, create_vector_store


LOGGER = get_logger(__name__)


@dataclass
class IngestionStats:
    documents: int
    chunks: int
    dimension: int


class IndexBuilder:
    def __init__(self, config: AppConfig, *, index_name: str = "markdown_rag") -> None:
        self.config = config
        self.index_name = index_name
        self.corpus = MarkdownCorpus(config.paths, config.chunking, config.corpus)
        self.embeddings: EmbeddingBackend = create_embeddings(config.embeddings)
        self.store: VectorStore | None = None

    def _ensure_store(self, dim: int) -> VectorStore:
        if self.store is None:
            self.store = create_vector_store(self.config, self.index_name, dim=dim)
        return self.store

    def build(self) -> IngestionStats:
        chunks: List[DocumentChunk] = self.corpus.build()
        if not chunks:
            raise RuntimeError(
                f"No documents found in {self.config.paths.data_dir}. Add files and retry."
            )

        texts = [chunk.text for chunk in chunks]
        LOGGER.info("Embedding %s chunks with %s", len(texts), self.config.embeddings.model_name)
        vectors = self.embeddings.embed_documents(texts)

        store = self._ensure_store(vectors.shape[1])
        records = [
            VectorRecord(text=chunk.text, source_file=chunk.source_file, chunk_id=chunk.chunk_id)
            for chunk in chunks
        ]

        LOGGER.info("Adding vectors to index %s", store.name)
        store.add(vectors, records)
        store.save()

        return IngestionStats(
            documents=len(set(record.source_file for record in records)),
            chunks=len(records),
            dimension=vectors.shape[1],
        )
