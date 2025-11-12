"""Utilities for retrieving relevant context from the configured vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import RetrievalConfig
from .embedding import EmbeddingBackend
from .store import VectorRecord, VectorStore


@dataclass
class RetrievedChunk:
    score: float
    record: VectorRecord

    @property
    def formatted(self) -> str:
        return (
            f"[source: {self.record.source_file}#{self.record.chunk_id} | score: {self.score:.3f}]\n"
            f"{self.record.text}"
        )


class ContextBuilder:
    def __init__(
        self,
        embeddings: EmbeddingBackend,
        store: VectorStore,
        retrieval: RetrievalConfig,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.retrieval = retrieval

    def retrieve(self, query: str, *, top_k: int | None = None) -> List[RetrievedChunk]:
        k = top_k or self.retrieval.top_k
        vector = self.embeddings.embed_query(query)
        results = self.store.search(vector, k=k)
        return [RetrievedChunk(score=score, record=record) for score, record in results]

    def build_context(self, query: str, *, max_chars: int | None = None, top_k: int | None = None) -> str:
        max_chars = max_chars or self.retrieval.max_context_chars
        chunks = self.retrieve(query, top_k=top_k)
        text_blocks: List[str] = []
        current_length = 0
        for chunk in chunks:
            block = chunk.formatted
            if current_length + len(block) > max_chars:
                remaining = max_chars - current_length
                if remaining <= 0:
                    break
                block = block[:remaining]
            text_blocks.append(block)
            current_length += len(block)
            if current_length >= max_chars:
                break
        return "\n\n---\n\n".join(text_blocks)
