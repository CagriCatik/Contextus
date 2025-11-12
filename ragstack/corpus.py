"""Corpus building utilities for heterogeneous document sources."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from tqdm import tqdm

from .config import ChunkingConfig, CorpusConfig, PathConfig
from .logging import get_logger
from .text import Chunker, strip_markdown


LOGGER = get_logger(__name__)


@dataclass
class DocumentChunk:
    text: str
    source_file: str
    chunk_id: int


class MarkdownCorpus:
    """Load documents, normalise them, and produce RAG-ready chunks."""

    def __init__(
        self,
        paths: PathConfig,
        chunking: ChunkingConfig,
        corpus_config: CorpusConfig,
    ) -> None:
        self.paths = paths
        self.chunker = Chunker(chunking)
        self.config = corpus_config
        self._converter = self._init_converter() if corpus_config.use_markitdown else None

    def _init_converter(self):
        try:
            from markitdown import MarkItDown
        except ImportError:  # pragma: no cover - optional dependency guard
            LOGGER.warning(
                "MarkItDown integration requested but the package is missing."
                " Install markitdown to enable rich document conversion."
            )
            return None

        return MarkItDown()

    def list_documents(self) -> List[Path]:
        includes = self.config.include or ["**/*.md"]
        candidates = set()
        for pattern in includes:
            candidates.update(path for path in self.paths.data_dir.glob(pattern) if path.is_file())

        excludes = self.config.exclude or []
        if excludes:
            filtered = []
            for path in candidates:
                rel = path.relative_to(self.paths.data_dir).as_posix()
                if any(fnmatch.fnmatch(rel, pattern) for pattern in excludes):
                    continue
                filtered.append(path)
            candidates = set(filtered)

        return sorted(candidates)

    def _load_text(self, path: Path) -> Optional[str]:
        if self._converter is not None:
            try:
                result = self._converter.convert(path)
            except Exception as exc:  # pragma: no cover - third-party failure
                LOGGER.warning("MarkItDown failed for %s: %s", path, exc)
            else:
                text = result.text_content or result.markdown
                if text:
                    return text

        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            LOGGER.warning("Skipping %s: unable to decode as UTF-8", path)
        except OSError as exc:  # pragma: no cover - unexpected filesystem errors
            LOGGER.warning("Skipping %s: %s", path, exc)
        return None

    def iter_chunks(self) -> Iterator[DocumentChunk]:
        files = self.list_documents()
        if not files:
            LOGGER.warning("No documents found in %s", self.paths.data_dir)
            return

        for md_path in tqdm(files, desc="Processing documents", unit="file"):
            raw_text = self._load_text(md_path)
            if not raw_text:
                continue
            plain_text = strip_markdown(raw_text)
            for chunk in self.chunker.iter_chunks(plain_text):
                yield DocumentChunk(
                    text=chunk.text,
                    source_file=str(md_path.relative_to(self.paths.data_dir)),
                    chunk_id=chunk.index,
                )

    def build(self) -> List[DocumentChunk]:
        return list(self.iter_chunks() or [])
