"""Persistent conversational memory and adaptive context management utilities."""

from __future__ import annotations

import json
import textwrap
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import AppConfig, RetrievalConfig
from .embedding import EmbeddingBackend
from .logging import get_logger
from .store import VectorRecord, VectorStore, create_vector_store
from .tokenization import TokenCounter

LOGGER = get_logger(__name__)


@dataclass
class MemoryEntry:
    """Representation of a persisted conversational turn."""

    entry_id: int
    created_at: datetime
    query: str
    answer: str
    summary: str
    session_id: str

    def to_json(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "created_at": self.created_at.isoformat(),
            "query": self.query,
            "answer": self.answer,
            "summary": self.summary,
            "session_id": self.session_id,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "MemoryEntry":
        created = payload.get("created_at")
        if isinstance(created, str):
            created_at = datetime.fromisoformat(created)
        else:
            created_at = datetime.utcnow()
        return cls(
            entry_id=int(payload.get("entry_id", 0)),
            created_at=created_at,
            query=str(payload.get("query", "")),
            answer=str(payload.get("answer", "")),
            summary=str(payload.get("summary", "")),
            session_id=str(payload.get("session_id", "")) or "unknown",
        )


class ConversationMemory:
    """Manages persistent long-term memory and adaptive recall for chat sessions."""

    def __init__(
        self,
        config: AppConfig,
        embeddings: EmbeddingBackend,
        retrieval: RetrievalConfig,
    ) -> None:
        self.config = config
        self.cfg = config.memory
        self.embeddings = embeddings
        self.retrieval = retrieval
        self.enabled = self.cfg.enabled
        self.session_id = uuid.uuid4().hex
        self._entries: List[MemoryEntry] = []
        self._store: VectorStore | None = None
        self._load_paths()
        self._token_counter = TokenCounter(
            encoding_name=self.cfg.token_encoder or retrieval.token_encoder,
            fallback_chars_per_token=self.cfg.fallback_chars_per_token,
        )
        self._summary_counter = TokenCounter(
            encoding_name=self.cfg.token_encoder or retrieval.token_encoder,
            fallback_chars_per_token=self.cfg.fallback_chars_per_token,
        )
        self._summary_text: str = ""
        if self.enabled:
            self._load_history()
            self._ensure_store()
            self._refresh_summary(save=False)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _load_paths(self) -> None:
        base_index = self.config.paths.index_dir
        summary_path = self.cfg.summary_path
        log_path = self.cfg.log_path
        self.summary_path = (
            Path(summary_path).expanduser().resolve()
            if summary_path
            else base_index / f"{self.cfg.index_name}_summary.json"
        )
        self.log_path = (
            Path(log_path).expanduser().resolve()
            if log_path
            else base_index / f"{self.cfg.index_name}_log.jsonl"
        )
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_history(self) -> None:
        if not self.log_path.exists():
            return
        try:
            with self.log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        LOGGER.warning("Skipping malformed memory entry: %s", line)
                        continue
                    self._entries.append(MemoryEntry.from_json(payload))
        except OSError as exc:
            LOGGER.warning("Failed to read memory log '%s': %s", self.log_path, exc)
        if self.summary_path.exists():
            try:
                with self.summary_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                    self._summary_text = str(payload.get("summary", ""))
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Failed to load memory summary '%s': %s", self.summary_path, exc)

    def _ensure_store(self) -> None:
        if self._store is not None:
            return
        probe_vector = self.embeddings.embed_documents(["memory-dimension-probe"])
        dim = int(probe_vector.shape[1]) if probe_vector.ndim == 2 else int(probe_vector.shape[0])
        self._store = create_vector_store(self.config, self.cfg.index_name, dim=dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def maybe_answer(self, query: str) -> Optional[str]:
        """Return a cached response if a high-confidence semantic match exists."""

        if not self.enabled or not self.cfg.cache_enabled:
            return None
        store = self._store
        if store is None or store.is_empty:
            return None
        vector = self.embeddings.embed_query(query)
        results = store.search(vector, k=max(1, self.cfg.max_memory_items))
        threshold = self.cfg.cache_min_score or 0.0
        for score, record in results:
            if score < threshold:
                continue
            answer = self._extract_answer(record.text)
            if answer:
                return answer
        return None

    def build_memory_context(self, query: str) -> Tuple[str, int]:
        """Return a contextual memory block and the estimated token usage."""

        if not self.enabled:
            return "", 0
        sections: List[str] = []
        token_budget = self.cfg.max_memory_tokens
        used_tokens = 0

        summary_text = self._summary_text.strip()
        if summary_text:
            block = f"Persistent summary:\n{summary_text}" if summary_text else ""
            if block:
                summary_tokens = self._token_counter.count(block)
                if token_budget is None or summary_tokens <= token_budget:
                    sections.append(block)
                    used_tokens += summary_tokens
                    if token_budget is not None:
                        token_budget = max(0, token_budget - summary_tokens)

        episodic_block, episodic_tokens = self._retrieve_episodic_memories(
            query, token_budget
        )
        if episodic_block:
            sections.append(episodic_block)
            used_tokens += episodic_tokens

        context_text = "\n\n".join(part for part in sections if part.strip())
        return context_text, used_tokens

    def remember(self, query: str, answer: str, *, from_cache: bool = False) -> None:
        """Persist a new conversational turn for future recall."""

        if not self.enabled:
            return
        entry_id = self._next_entry_id()
        summary = self._summarise_turn(query, answer)
        entry = MemoryEntry(
            entry_id=entry_id,
            created_at=datetime.utcnow(),
            query=query,
            answer=answer,
            summary=summary,
            session_id=self.session_id,
        )
        self._entries.append(entry)
        self._append_log(entry)
        formatted = self._format_vector_text(entry, from_cache=from_cache)
        if not from_cache:
            vector = self.embeddings.embed_documents([formatted])
            store = self._store
            if store is None:
                self._ensure_store()
                store = self._store
            if store is None:
                LOGGER.warning("Memory store could not be initialised; skipping persistence.")
                return
            store.add(
                vector,
                [VectorRecord(text=formatted, source_file="memory", chunk_id=entry.entry_id)],
            )
            store.save()
        self._refresh_summary(save=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _retrieve_episodic_memories(
        self, query: str, token_budget: Optional[int]
    ) -> Tuple[str, int]:
        store = self._store
        if store is None or store.is_empty:
            return "", 0
        vector = self.embeddings.embed_query(query)
        search_k = max(self.cfg.search_top_k, self.cfg.max_memory_items)
        results = store.search(vector, k=search_k)
        min_score = self.cfg.min_score or 0.0
        selected: List[str] = []
        used_tokens = 0
        for score, record in results:
            if score < min_score:
                continue
            block = self._format_context_block(record, score)
            block_tokens = self._token_counter.count(block)
            if token_budget is not None and used_tokens + block_tokens > token_budget:
                remaining = token_budget - used_tokens
                if remaining <= 0:
                    break
                truncated, truncated_tokens = self._token_counter.truncate(block, remaining)
                if not truncated.strip():
                    break
                block = truncated
                block_tokens = truncated_tokens
            selected.append(block)
            used_tokens += block_tokens
            if len(selected) >= self.cfg.max_memory_items:
                break
            if token_budget is not None and used_tokens >= token_budget:
                break
        return "\n\n".join(selected), used_tokens

    def _extract_answer(self, text: str) -> Optional[str]:
        if "Answer:" not in text:
            return None
        answer_section = text.split("Answer:", 1)[1]
        summary_split = answer_section.split("\nSummary:", 1)
        if summary_split:
            answer = summary_split[0]
        else:
            answer = answer_section
        return answer.strip() or None

    def _summarise_turn(self, query: str, answer: str) -> str:
        snippet = f"Q: {query.strip()} | A: {answer.strip()}"
        return textwrap.shorten(snippet, width=self.cfg.summary_max_chars, placeholder="…")

    def _format_vector_text(self, entry: MemoryEntry, *, from_cache: bool) -> str:
        origin = "cached" if from_cache else "fresh"
        return (
            f"Turn: {entry.entry_id}\n"
            f"Session: {entry.session_id}\n"
            f"Created: {entry.created_at.isoformat()}\n"
            f"Origin: {origin}\n"
            f"Question: {entry.query.strip()}\n"
            f"Answer: {entry.answer.strip()}\n"
            f"Summary: {entry.summary.strip()}"
        )

    def _format_context_block(self, record: VectorRecord, score: float) -> str:
        header = f"[memory turn={record.chunk_id} | score={score:.3f}]"
        return f"{header}\n{record.text.strip()}"

    def _next_entry_id(self) -> int:
        if not self._entries:
            return 1
        return max(entry.entry_id for entry in self._entries) + 1

    def _append_log(self, entry: MemoryEntry) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")
        except OSError as exc:
            LOGGER.warning("Failed to append memory entry to '%s': %s", self.log_path, exc)

    def _refresh_summary(self, *, save: bool) -> None:
        if not self._entries:
            self._summary_text = ""
            if save:
                self._persist_summary()
            return
        window = max(1, self.cfg.rolling_window)
        recent_entries = self._entries[-window:]
        bullets = [
            f"- ({entry.created_at.strftime('%Y-%m-%d %H:%M')}) {entry.summary}"
            for entry in recent_entries
        ]
        summary_text = "\n".join(bullets)
        if self.cfg.summary_tokens is not None:
            summary_tokens = self._summary_counter.count(summary_text)
            while summary_tokens > self.cfg.summary_tokens and len(bullets) > 1:
                bullets.pop(0)
                summary_text = "\n".join(bullets)
                summary_tokens = self._summary_counter.count(summary_text)
        self._summary_text = summary_text
        if save:
            self._persist_summary()

    def _persist_summary(self) -> None:
        payload = {"summary": self._summary_text}
        try:
            with self.summary_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            LOGGER.warning("Failed to persist memory summary '%s': %s", self.summary_path, exc)

    # ------------------------------------------------------------------
    # Cache maintenance helpers
    # ------------------------------------------------------------------
    def prune_cache(self) -> None:
        if not self.enabled or not self.cfg.cache_enabled:
            return
        ttl_minutes = max(1, self.cfg.cache_ttl_minutes)
        cutoff = datetime.utcnow() - timedelta(minutes=ttl_minutes)
        original = len(self._entries)
        self._entries = [entry for entry in self._entries if entry.created_at >= cutoff]
        if len(self._entries) != original:
            LOGGER.info(
                "Pruned %s cached memory entr%s older than %s minutes.",
                original - len(self._entries),
                "ies" if original - len(self._entries) != 1 else "y",
                ttl_minutes,
            )
            self._refresh_summary(save=True)

    def all_entries(self) -> Sequence[MemoryEntry]:
        return tuple(self._entries)
