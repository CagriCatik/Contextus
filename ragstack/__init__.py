"""High-level public API for the provider-agnostic RAG toolkit."""

from .config import (
    AppConfig,
    ChunkingConfig,
    CorpusConfig,
    DocumentationConfig,
    EmbeddingConfig,
    TaskProfile,
    LLMConfig,
    OllamaConfig,
    OpenAIConfig,
    PathConfig,
    PromptConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from .corpus import DocumentChunk, MarkdownCorpus
from .embedding import (
    EmbeddingBackend,
    SentenceTransformerEmbeddings,
    create_embeddings,
    register_embedding_backend,
)
from .llm import ModelClientError, ModelConnectionError, ModelInfo, ModelNotFoundError
from .ollama import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaModel,
    OllamaModelNotFoundError,
)
from .openai_client import (
    OpenAIClient,
    OpenAIAuthenticationError,
    OpenAIClientError,
    OpenAIModelNotFoundError,
)
from .pipeline import IndexBuilder, IngestionStats
from .retrieval import ContextBuilder
from .store import (
    FaissVectorStore,
    VectorRecord,
    VectorStore,
    create_vector_store,
    register_vector_store,
)
from .chat import ChatSession
from .docgen import TestCase, TestSpecGenerator
from .tokenization import TokenCounter

__all__ = [
    "AppConfig",
    "ChunkingConfig",
    "CorpusConfig",
    "DocumentationConfig",
    "EmbeddingConfig",
    "TaskProfile",
    "LLMConfig",
    "OllamaConfig",
    "OpenAIConfig",
    "PathConfig",
    "PromptConfig",
    "RetrievalConfig",
    "VectorStoreConfig",
    "DocumentChunk",
    "MarkdownCorpus",
    "EmbeddingBackend",
    "SentenceTransformerEmbeddings",
    "create_embeddings",
    "register_embedding_backend",
    "ModelClientError",
    "ModelConnectionError",
    "ModelInfo",
    "ModelNotFoundError",
    "OllamaClient",
    "OllamaClientError",
    "OllamaConnectionError",
    "OllamaModel",
    "OllamaModelNotFoundError",
    "OpenAIClient",
    "OpenAIClientError",
    "OpenAIAuthenticationError",
    "OpenAIModelNotFoundError",
    "IndexBuilder",
    "IngestionStats",
    "FaissVectorStore",
    "VectorRecord",
    "VectorStore",
    "create_vector_store",
    "register_vector_store",
    "ContextBuilder",
    "ChatSession",
    "TestCase",
    "TestSpecGenerator",
    "TokenCounter",
]
