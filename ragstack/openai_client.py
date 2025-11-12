"""Client for interacting with the OpenAI Chat Completions API."""

from __future__ import annotations

from typing import List, Optional, Sequence

from openai import OpenAI
from openai import AuthenticationError, NotFoundError, OpenAIError

from .config import OpenAIConfig
from .llm import ModelClientError, ModelInfo, ModelNotFoundError


class OpenAIClientError(ModelClientError):
    """Base exception raised for OpenAI client errors."""


class OpenAIAuthenticationError(OpenAIClientError):
    """Raised when authentication with OpenAI fails."""


class OpenAIModelNotFoundError(OpenAIClientError, ModelNotFoundError):
    """Raised when the requested OpenAI model is unavailable."""


class OpenAIClient:
    def __init__(self, config: OpenAIConfig, *, model: Optional[str] = None) -> None:
        if not config.api_key:
            raise OpenAIAuthenticationError(
                "OPENAI_API_KEY is not configured. Add it to your .env file or environment."
            )
        client_kwargs = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        if config.organization:
            client_kwargs["organization"] = config.organization

        self.config = config
        self.model = model or config.model
        self._client = OpenAI(**client_kwargs)

    def list_models(self) -> List[ModelInfo]:
        try:
            response = self._client.models.list()
        except AuthenticationError as exc:
            raise OpenAIAuthenticationError("Authentication with OpenAI failed.") from exc
        except OpenAIError as exc:
            raise OpenAIClientError(f"Failed to list OpenAI models: {exc}") from exc

        data = getattr(response, "data", [])
        return [ModelInfo(name=item.id) for item in data if getattr(item, "id", None)]

    def ensure_model(self, model_name: str, installed_models: Sequence[ModelInfo] | None = None) -> None:
        models = installed_models or self.list_models()
        for model in models:
            if model.name == model_name:
                self.model = model_name
                return
        available = ", ".join(model.name for model in models) or "<none>"
        raise OpenAIModelNotFoundError(
            f"Model '{model_name}' is not available in the current OpenAI account. Known models: {available}."
        )

    def chat(self, messages: Sequence[dict]) -> str:
        if not self.model:
            raise OpenAIClientError("No model configured. Provide --model or set llm.default_model.")
        try:
            response = self._client.chat.completions.create(model=self.model, messages=list(messages))
        except AuthenticationError as exc:
            raise OpenAIAuthenticationError("Authentication with OpenAI failed.") from exc
        except NotFoundError as exc:
            raise OpenAIModelNotFoundError(f"Model '{self.model}' is not available.") from exc
        except OpenAIError as exc:
            raise OpenAIClientError(f"OpenAI chat completion failed: {exc}") from exc

        choices = getattr(response, "choices", [])
        if not choices:
            return ""
        message = choices[0].message
        content = getattr(message, "content", "")
        return str(content)
