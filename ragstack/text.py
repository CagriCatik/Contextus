"""Utilities for cleaning and chunking markdown documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List

from .config import ChunkingConfig


_MARKDOWN_CODE_BLOCK = re.compile(r"```.*?```", flags=re.DOTALL)
_MARKDOWN_INLINE_CODE = re.compile(r"`([^`]+)`")
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MARKDOWN_HEADING = re.compile(r"^#+\s*", flags=re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")


def strip_markdown(md_text: str) -> str:
    """Convert markdown content to a whitespace-normalised plain text string."""

    text = _MARKDOWN_CODE_BLOCK.sub("", md_text)
    text = _MARKDOWN_INLINE_CODE.sub(r"\1", text)
    text = _MARKDOWN_IMAGE.sub("", text)
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _MARKDOWN_HEADING.sub("", text)
    text = text.replace("*", " ").replace("_", " ").replace("#", " ")
    text = _WHITESPACE.sub(" ", text).strip()
    return text


@dataclass
class Chunk:
    text: str
    index: int


class Chunker:
    """Character-based chunker suitable for lightweight RAG pipelines."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def iter_chunks(self, text: str) -> Iterator[Chunk]:
        if not text:
            return
        max_chars = self.config.max_chars
        min_chars = self.config.min_chars
        overlap = self.config.overlap

        start = 0
        doc_length = len(text)
        chunk_id = 0

        while start < doc_length:
            end = min(start + max_chars, doc_length)
            chunk_text = text[start:end]

            while len(chunk_text) < min_chars and end < doc_length:
                end = min(end + (min_chars - len(chunk_text)), doc_length)
                chunk_text = text[start:end]

            yield Chunk(text=chunk_text, index=chunk_id)

            if end == doc_length:
                break

            start = max(0, end - overlap)
            chunk_id += 1

    def chunk_document(self, text: str) -> List[Chunk]:
        return list(self.iter_chunks(text))

    def chunk_documents(self, texts: Iterable[str]) -> List[List[Chunk]]:
        return [self.chunk_document(text) for text in texts]
