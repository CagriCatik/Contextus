"""Token counting utilities with optional model-aware encoders."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Optional, Sequence


_TIKTOKEN_AVAILABLE = importlib.util.find_spec("tiktoken") is not None
if _TIKTOKEN_AVAILABLE:
    import tiktoken  # type: ignore[import-not-found]
else:  # pragma: no cover - runtime fallback when tiktoken is absent
    tiktoken = None  # type: ignore[assignment]


@dataclass
class TokenCounter:
    """Lightweight helper that estimates token usage for context budgeting."""

    encoding_name: Optional[str] = None
    fallback_chars_per_token: float = 4.0

    def __post_init__(self) -> None:
        self._encoding = None
        if _TIKTOKEN_AVAILABLE and self.encoding_name:
            self._encoding = self._resolve_encoding(self.encoding_name)

    @staticmethod
    def _resolve_encoding(name: str):  # type: ignore[no-untyped-def]
        if not _TIKTOKEN_AVAILABLE:
            return None
        try:
            return tiktoken.encoding_for_model(name)  # type: ignore[attr-defined]
        except KeyError:
            if not name:
                return tiktoken.get_encoding("cl100k_base")  # type: ignore[attr-defined]
            try:
                return tiktoken.get_encoding(name)  # type: ignore[attr-defined]
            except KeyError:
                return tiktoken.get_encoding("cl100k_base")  # type: ignore[attr-defined]
        except ValueError:
            return tiktoken.get_encoding("cl100k_base")  # type: ignore[attr-defined]

    def _ensure_encoding(self):  # type: ignore[no-untyped-def]
        if not _TIKTOKEN_AVAILABLE:
            return None
        if self._encoding is not None:
            return self._encoding
        if self.encoding_name:
            self._encoding = self._resolve_encoding(self.encoding_name)
        return self._encoding

    def count(self, text: str) -> int:
        encoding = self._ensure_encoding()
        if encoding is not None:
            return len(encoding.encode(text))  # type: ignore[attr-defined]
        if not text:
            return 0
        if self.fallback_chars_per_token <= 0:
            return len(text)
        return max(1, int(len(text) / self.fallback_chars_per_token))

    def truncate(self, text: str, max_tokens: int) -> tuple[str, int]:
        if max_tokens <= 0:
            return "", 0
        encoding = self._ensure_encoding()
        if encoding is not None:
            tokens = encoding.encode(text)  # type: ignore[attr-defined]
            if len(tokens) <= max_tokens:
                return text, len(tokens)
            truncated = tokens[:max_tokens]
            decoded = encoding.decode(truncated)  # type: ignore[attr-defined]
            return decoded, len(truncated)
        if not text:
            return "", 0
        approx_chars = int(max_tokens * self.fallback_chars_per_token)
        if approx_chars >= len(text):
            return text, self.count(text)
        truncated_text = text[:approx_chars]
        return truncated_text, max_tokens

    def accumulate(self, blocks: Sequence[str]) -> int:
        total = 0
        for block in blocks:
            total += self.count(block)
        return total
