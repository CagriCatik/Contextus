"""Configuration models and helpers for the RAG stack."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import copy

from dotenv import load_dotenv
import json
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    env_file = os.getenv("ENV_FILE", ".env")
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_file
    if env_path.exists():
        load_dotenv(env_path, override=False)
    _DOTENV_LOADED = True


def _env(key: str) -> Optional[str]:
    value = os.getenv(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _update_dataclass(instance: Any, values: Dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(instance, key):
            setattr(instance, key, value)


@dataclass
class PathConfig:
    """Filesystem layout for the project."""

    base_dir: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "index")

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        if "base_dir" in overrides:
            self.base_dir = Path(overrides["base_dir"]).expanduser().resolve()
        if "data_dir" in overrides:
            self.data_dir = Path(overrides["data_dir"]).expanduser().resolve()
        if "index_dir" in overrides:
            self.index_dir = Path(overrides["index_dir"]).expanduser().resolve()

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class CorpusConfig:
    """Controls how source documents are discovered and normalised."""

    include: List[str] = field(
        default_factory=lambda: ["**/*.md", "**/*.mdx", "**/*.txt", "**/*.pdf"]
    )
    exclude: List[str] = field(default_factory=lambda: ["**/.ipynb_checkpoints/**"])
    use_markitdown: bool = True


@dataclass
class EmbeddingConfig:
    """SentenceTransformer configuration."""

    backend: str = "sentence_transformer"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"


@dataclass
class ChunkingConfig:
    """Parameters for converting markdown into overlapping text chunks."""

    max_chars: int = 800
    min_chars: int = 200
    overlap: int = 100


@dataclass
class RetrievalConfig:
    """Runtime retrieval configuration."""

    top_k: int = 5
    max_context_chars: int = 4000


@dataclass
class VectorStoreConfig:
    """Backend selection and parameters for the vector store layer."""

    backend: str = "faiss"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OllamaConfig:
    """Connection details for the Ollama HTTP API."""

    host: str = "http://localhost:11434"
    timeout: float = 30.0


@dataclass
class OpenAIConfig:
    """Settings for the OpenAI Chat Completions API."""

    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = "https://api.openai.com/v1"
    organization: Optional[str] = None


@dataclass
class LLMConfig:
    """Provider selection and defaults for the chat layer."""

    provider: str = "ollama"
    default_model: Optional[str] = None


@dataclass
class PromptConfig:
    """Customisable prompt templates for chat generation."""

    system_prompt: str = (
        "You are a helpful assistant that answers questions using the provided context. "
        "If the context is insufficient, say you do not know and suggest how to expand the knowledge base."
    )
    context_template: str = (
        "Here is the retrieved context from the knowledge base:\n\n{context}\n\n"
        "When answering, cite the relevant [source: ...] blocks if needed."
    )
    question_template: str = "User question: {query}"


@dataclass
class DocumentationConfig:
    """Settings that control automated documentation generation workflows."""

    test_case_id_prefix: str = "TC-"
    starting_index: int = 1
    system_prompt: str = (
        "You are a senior QA engineer who writes thorough and actionable test specifications. "
        "Use the supplied requirements context to derive precise test cases and reference every "
        "requirement you rely on."
    )


@dataclass
class AppConfig:
    """Aggregate configuration container used throughout the project."""

    paths: PathConfig = field(default_factory=PathConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    documentation: DocumentationConfig = field(default_factory=DocumentationConfig)
    task_profiles: Dict[str, "TaskProfile"] = field(default_factory=dict)

    @classmethod
    def load(cls, *, config_path: Optional[Path] = None) -> "AppConfig":
        """Create an :class:`AppConfig` from YAML/JSON and environment overrides."""

        _load_dotenv_once()
        instance = cls()

        # File overrides (YAML preferred, JSON supported for backwards compatibility)
        file_path = config_path or _env("APP_CONFIG_FILE")
        if file_path is None:
            # default to config.yaml if present, otherwise fall back to legacy config.json
            yaml_path = PROJECT_ROOT / "config.yaml"
            json_path = PROJECT_ROOT / "config.json"
            file_path = yaml_path if yaml_path.exists() or not json_path.exists() else json_path
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = PROJECT_ROOT / file_path
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as handle:
                if file_path.suffix.lower() in {".yaml", ".yml"}:
                    payload = yaml.safe_load(handle) or {}
                else:
                    payload = json.load(handle)
            instance.apply_mapping(payload)

        # Environment overrides
        instance.apply_environment()
        instance.paths.ensure_directories()
        return instance

    # ------------------------------------------------------------------
    # Override helpers
    # ------------------------------------------------------------------
    def apply_mapping(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return

        if "paths" in payload:
            self.paths.apply_overrides(payload["paths"])
        if "corpus" in payload:
            _update_dataclass(self.corpus, payload["corpus"])
        if "embeddings" in payload:
            _update_dataclass(self.embeddings, payload["embeddings"])
        if "chunking" in payload:
            _update_dataclass(self.chunking, payload["chunking"])
        if "retrieval" in payload:
            _update_dataclass(self.retrieval, payload["retrieval"])
        if "vector_store" in payload:
            _update_dataclass(self.vector_store, payload["vector_store"])
        if "ollama" in payload:
            _update_dataclass(self.ollama, payload["ollama"])
        if "openai" in payload:
            _update_dataclass(self.openai, payload["openai"])
        if "llm" in payload:
            _update_dataclass(self.llm, payload["llm"])
        if "prompts" in payload:
            _update_dataclass(self.prompts, payload["prompts"])
        if "documentation" in payload:
            _update_dataclass(self.documentation, payload["documentation"])
        if "tasks" in payload:
            self.task_profiles = {
                name: TaskProfile.from_mapping(name, data)
                for name, data in payload["tasks"].items()
            }

    def apply_environment(self) -> None:
        embedding_model = _env("EMBEDDING_MODEL_NAME")
        if embedding_model:
            self.embeddings.model_name = embedding_model

        embedding_device = _env("EMBEDDING_DEVICE")
        if embedding_device:
            self.embeddings.device = embedding_device

        max_chars = _env("MAX_CHARS_PER_CHUNK")
        if max_chars:
            self.chunking.max_chars = int(max_chars)
        min_chars = _env("MIN_CHARS_PER_CHUNK")
        if min_chars:
            self.chunking.min_chars = int(min_chars)
        overlap = _env("CHUNK_OVERLAP")
        if overlap:
            self.chunking.overlap = int(overlap)

        top_k = _env("RAG_TOP_K")
        if top_k:
            self.retrieval.top_k = int(top_k)
        max_context = _env("RAG_MAX_CONTEXT_CHARS")
        if max_context:
            self.retrieval.max_context_chars = int(max_context)

        ollama_host = _env("OLLAMA_HOST")
        if ollama_host:
            self.ollama.host = ollama_host
        ollama_timeout = _env("OLLAMA_TIMEOUT")
        if ollama_timeout:
            self.ollama.timeout = float(ollama_timeout)

        provider = _env("LLM_PROVIDER")
        if provider:
            self.llm.provider = provider.lower()
        default_model = _env("LLM_MODEL")
        if default_model:
            self.llm.default_model = default_model

        vector_backend = _env("VECTOR_STORE_BACKEND")
        if vector_backend:
            self.vector_store.backend = vector_backend

        openai_key = _env("OPENAI_API_KEY")
        if openai_key:
            self.openai.api_key = openai_key
        openai_model = _env("OPENAI_MODEL")
        if openai_model:
            self.openai.model = openai_model
        openai_base = _env("OPENAI_BASE_URL")
        if openai_base:
            self.openai.base_url = openai_base
        openai_org = _env("OPENAI_ORG")
        if openai_org:
            self.openai.organization = openai_org

        system_prompt = _env("SYSTEM_PROMPT")
        if system_prompt:
            self.prompts.system_prompt = system_prompt
        context_template = _env("CONTEXT_TEMPLATE")
        if context_template:
            self.prompts.context_template = context_template
        question_template = _env("QUESTION_TEMPLATE")
        if question_template:
            self.prompts.question_template = question_template

        test_id_prefix = _env("TEST_CASE_ID_PREFIX")
        if test_id_prefix:
            self.documentation.test_case_id_prefix = test_id_prefix

        test_start_index = _env("TEST_CASE_START_INDEX")
        if test_start_index:
            self.documentation.starting_index = int(test_start_index)

        test_system_prompt = _env("TEST_SPEC_SYSTEM_PROMPT")
        if test_system_prompt:
            self.documentation.system_prompt = test_system_prompt

    # ------------------------------------------------------------------
    # Task profile helpers
    # ------------------------------------------------------------------
    def resolve_task(self, task_name: Optional[str]) -> Tuple[RetrievalConfig, PromptConfig]:
        if not task_name:
            return self.retrieval, self.prompts
        profile = self.task_profiles.get(task_name)
        if profile is None:
            raise KeyError(f"Task profile '{task_name}' is not defined in the configuration.")
        return profile.build(self.retrieval, self.prompts)


@dataclass
class TaskProfile:
    """Per-task overrides for retrieval and prompt behaviour."""

    name: str
    retrieval_overrides: Dict[str, Any] = field(default_factory=dict)
    prompt_overrides: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, payload: Dict[str, Any]) -> "TaskProfile":
        retrieval = payload.get("retrieval", {}) or {}
        prompts = payload.get("prompts", {}) or {}
        return cls(name=name, retrieval_overrides=dict(retrieval), prompt_overrides=dict(prompts))

    def build(
        self, base_retrieval: RetrievalConfig, base_prompts: PromptConfig
    ) -> Tuple[RetrievalConfig, PromptConfig]:
        retrieval = copy.deepcopy(base_retrieval)
        prompts = copy.deepcopy(base_prompts)
        for key, value in self.retrieval_overrides.items():
            if hasattr(retrieval, key):
                setattr(retrieval, key, value)
        for key, value in self.prompt_overrides.items():
            if hasattr(prompts, key):
                setattr(prompts, key, value)
        return retrieval, prompts
