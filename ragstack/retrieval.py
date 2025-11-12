"""Utilities for retrieving relevant context from the configured vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import RetrievalConfig
from .embedding import EmbeddingBackend
from .store import VectorRecord, VectorStore
from .tokenization import TokenCounter


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
        self._token_counter = TokenCounter(
            encoding_name=retrieval.token_encoder,
            fallback_chars_per_token=retrieval.fallback_chars_per_token,
        )

    def retrieve(self, query: str, *, top_k: int | None = None) -> List[RetrievedChunk]:
        config = self.retrieval
        requested_k = top_k or config.top_k
        rerank_k = max(config.rerank_top_k, requested_k)
        vector = self.embeddings.embed_query(query)
        results = self.store.search(vector, k=rerank_k)
        chunks = [RetrievedChunk(score=score, record=record) for score, record in results]

        if config.rerank_top_k > 0 and chunks:
            head_count = min(config.rerank_top_k, len(chunks))
            head = chunks[:head_count]
            texts = [chunk.record.text for chunk in head]
            doc_vectors = self.embeddings.embed_documents(texts)
            query_vec = vector.reshape(1, -1)
            scores = doc_vectors @ query_vec.T
            reranked = [
                RetrievedChunk(score=float(score), record=chunk.record)
                for score, chunk in zip(scores[:, 0], head)
            ]
            chunks = (
                sorted(reranked, key=lambda item: item.score, reverse=True)
                + chunks[head_count:]
            )

        if config.min_score is not None:
            chunks = [chunk for chunk in chunks if chunk.score >= config.min_score]

        return chunks[:requested_k]

    def build_context(
        self,
        query: str,
        *,
        max_chars: int | None = None,
        max_tokens: int | None = None,
        top_k: int | None = None,
    ) -> str:
        config = self.retrieval
        max_chars = config.max_context_chars if max_chars is None else max_chars
        if max_chars is not None and max_chars <= 0:
            max_chars = None
        token_budget = config.max_context_tokens if max_tokens is None else max_tokens
        if token_budget is not None:
            token_budget = max(0, token_budget - config.token_overhead)
            if token_budget == 0:
                token_budget = None

        chunks = self.retrieve(query, top_k=top_k)
        text_blocks: List[str] = []
        current_chars = 0
        current_tokens = 0
        separator = config.context_separator

        for chunk in chunks:
            block = chunk.formatted
            block_tokens = None

            if token_budget is not None:
                block_tokens = self._token_counter.count(block)
                if current_tokens + block_tokens > token_budget:
                    remaining_tokens = token_budget - current_tokens
                    if remaining_tokens <= 0:
                        break
                    block, block_tokens = self._token_counter.truncate(block, remaining_tokens)
                    if not block.strip():
                        break

            if max_chars is not None:
                if current_chars + len(block) > max_chars:
                    remaining_chars = max_chars - current_chars
                    if remaining_chars <= 0:
                        break
                    block = block[:remaining_chars]
            text_blocks.append(block)
            current_chars += len(block)

            if token_budget is not None:
                if block_tokens is None:
                    block_tokens = self._token_counter.count(block)
                current_tokens += block_tokens
                if current_tokens >= token_budget:
                    break

            if max_chars is not None and current_chars >= max_chars:
                break

        return separator.join(text_blocks)
