"""High-level retrieval helper used by the command-line tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ragstack.config import AppConfig
from ragstack.embedding import create_embeddings
from ragstack.retrieval import ContextBuilder
from ragstack.store import create_vector_store


@dataclass
class RetrievedDocument:
    score: float
    text: str
    source_file: str
    chunk_id: int


class MarkdownRag:
    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        index_name: str = "markdown_rag",
        task: str | None = None,
    ) -> None:
        self.config = config or AppConfig.load()
        self.retrieval, _ = self.config.resolve_task(task)
        self.embeddings = create_embeddings(self.config.embeddings)
        try:
            self.store = create_vector_store(self.config, index_name, dim=None)
        except ValueError as exc:
            raise RuntimeError(
                f"Vector index '{index_name}' was not found. Run ingest_markdown.py to build it."
            ) from exc
        self.context_builder = ContextBuilder(self.embeddings, self.store, self.retrieval)

    def retrieve(self, query: str, k: int | None = None) -> List[RetrievedDocument]:
        chunks = self.context_builder.retrieve(query, top_k=k)
        return [
            RetrievedDocument(
                score=chunk.score,
                text=chunk.record.text,
                source_file=chunk.record.source_file,
                chunk_id=chunk.record.chunk_id,
            )
            for chunk in chunks
        ]

    def build_context(self, query: str, k: int | None = None, max_chars: int | None = None) -> str:
        return self.context_builder.build_context(query, top_k=k, max_chars=max_chars)
