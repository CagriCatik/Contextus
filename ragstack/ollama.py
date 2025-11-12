"""HTTP client for interacting with the Ollama REST API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import requests

from .config import OllamaConfig
from .llm import ModelClientError, ModelConnectionError, ModelInfo, ModelNotFoundError


class OllamaClientError(ModelClientError):
    """Base exception raised for Ollama client errors."""


class OllamaConnectionError(OllamaClientError, ModelConnectionError):
    """Raised when the Ollama HTTP API cannot be reached."""


class OllamaModelNotFoundError(OllamaClientError, ModelNotFoundError):
    """Raised when the requested model is not installed on the Ollama host."""


@dataclass
class OllamaModel(ModelInfo):
    pass


class OllamaClient:
    def __init__(
        self,
        config: OllamaConfig,
        *,
        model: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config
        self.model = model
        self._session = session or requests.Session()

    @property
    def base_url(self) -> str:
        return self.config.host.rstrip("/")

    # ------------------------------------------------------------------
    # Model discovery helpers
    # ------------------------------------------------------------------
    def list_models(self) -> List[OllamaModel]:
        url = f"{self.base_url}/api/tags"
        try:
            response = self._session.get(url, timeout=self.config.timeout)
        except requests.RequestException as exc:
            raise OllamaConnectionError(f"Unable to reach Ollama at {self.config.host}") from exc

        if response.status_code == 404:
            raise OllamaClientError(
                "Ollama server responded with 404 for /api/tags. "
                "Ensure the Ollama HTTP API is running."
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise OllamaClientError(
                f"Unexpected Ollama response ({response.status_code}): {response.text}"
            ) from exc

        payload = response.json()
        models = payload.get("models", [])
        return [OllamaModel(name=model.get("name", "")) for model in models if model.get("name")]

    def ensure_model(self, model_name: str, installed_models: Sequence[ModelInfo] | None = None) -> None:
        models = installed_models or self.list_models()
        for model in models:
            if model.name == model_name:
                self.model = model_name
                return
        available = ", ".join(model.name for model in models) or "<none>"
        raise OllamaModelNotFoundError(
            f"Model '{model_name}' is not installed on Ollama. Installed models: {available}."
        )

    # ------------------------------------------------------------------
    # Chat interaction
    # ------------------------------------------------------------------
    def chat(self, messages: Sequence[dict]) -> str:
        if not self.model:
            raise OllamaClientError("No model configured. Call 'ensure_model' or set 'model'.")

        url = f"{self.base_url}/api/chat"
        payload = {"model": self.model, "messages": list(messages), "stream": False}

        try:
            response = self._session.post(url, json=payload, timeout=self.config.timeout)
        except requests.RequestException as exc:
            raise OllamaConnectionError("Failed to connect to Ollama while calling /api/chat.") from exc

        if response.status_code == 404:
            raise OllamaModelNotFoundError(f"Model '{self.model}' is not available on Ollama.")

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise OllamaClientError(
                f"Ollama responded with status {response.status_code}: {response.text}"
            ) from exc

        data = response.json()
        message = data.get("message", {})
        return str(message.get("content", ""))

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)
