"""Interactive chat CLI backed by the modular RAG stack."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from ragstack.chat import ChatSession
from ragstack.config import AppConfig
from ragstack.embedding import create_embeddings
from ragstack.console import console
from ragstack.logging import configure_logging, get_logger
from ragstack.llm import ModelClientError, ModelNotFoundError
from ragstack.ollama import OllamaClient
from ragstack.openai_client import OpenAIClient
from ragstack.retrieval import ContextBuilder
from ragstack.store import create_vector_store

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a FAISS-backed RAG index using configurable providers."
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Path to a config file (YAML or JSON, default: config.yaml in project root).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Path to a .env file with secrets such as OPENAI_API_KEY (default: .env).",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        help="Override the language model provider configured in the config file.",
    )
    parser.add_argument("--model", help="Model to use for the selected provider.")
    parser.add_argument(
        "--embedding-device",
        help="Torch device for embeddings (auto, cpu, cuda, cuda:0, mps).",
    )
    parser.add_argument(
        "--host",
        help="Ollama HTTP host (default: OLLAMA_HOST env or value from the config file).",
    )
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds for Ollama requests.")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List installed models for the selected provider and exit.",
    )
    parser.add_argument(
        "--task",
        help="Name of a task profile from config.yaml to override retrieval and prompts.",
    )
    parser.add_argument("--top-k", type=int, help="Number of context chunks to retrieve per query.")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        help="Maximum character budget for the context block.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        help="Maximum token budget for the context block (post-separator).",
    )
    parser.add_argument(
        "--token-encoder",
        help="Tokenizer identifier used for estimating context tokens (e.g. cl100k_base).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum similarity score threshold for retrieved chunks.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        help="Re-embed the top-N results and rerank them with the query embedding.",
    )
    parser.add_argument("--index-dir", type=Path, help="Directory containing the FAISS index files.")
    parser.add_argument("--index-name", default="markdown_rag", help="Name of the FAISS index to query.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO).")
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.provider:
        config.llm.provider = args.provider
    if args.model:
        config.llm.default_model = args.model
    if args.embedding_device:
        config.embeddings.device = args.embedding_device
    if args.host:
        config.ollama.host = args.host
    if args.timeout:
        config.ollama.timeout = args.timeout
    if args.top_k:
        config.retrieval.top_k = args.top_k
    if args.max_context_chars:
        config.retrieval.max_context_chars = args.max_context_chars
    if args.max_context_tokens is not None:
        config.retrieval.max_context_tokens = args.max_context_tokens
    if args.token_encoder:
        config.retrieval.token_encoder = args.token_encoder
    if args.min_score is not None:
        config.retrieval.min_score = args.min_score
    if args.rerank_top_k is not None:
        config.retrieval.rerank_top_k = args.rerank_top_k
    if args.index_dir:
        config.paths.index_dir = args.index_dir.resolve()
        config.paths.index_dir.mkdir(parents=True, exist_ok=True)


def build_session(config: AppConfig, args: argparse.Namespace) -> ChatSession:
    try:
        retrieval_cfg, prompt_cfg = config.resolve_task(args.task)
    except KeyError as exc:
        message = exc.args[0] if exc.args else str(exc)
        raise RuntimeError(message) from exc

    embeddings = create_embeddings(config.embeddings)
    try:
        store = create_vector_store(config, args.index_name, dim=None)
    except ValueError as exc:
        raise RuntimeError(
            f"Vector index '{args.index_name}' not found in {config.paths.index_dir}. "
            "Run ingest_markdown.py before starting the chat."
        ) from exc

    context_builder = ContextBuilder(embeddings, store, retrieval_cfg)
    client, provider_name = build_client(config, args)
    return ChatSession(client, context_builder, retrieval_cfg, prompt_cfg, provider_name)


def build_client(config: AppConfig, args: argparse.Namespace):
    provider = (config.llm.provider or "ollama").lower()
    model_override = args.model or config.llm.default_model

    if provider == "openai":
        client = OpenAIClient(config.openai, model=model_override)
        provider_name = "openai"
    else:
        client = OllamaClient(config.ollama, model=model_override)
        provider_name = "ollama"
    if not config.llm.default_model and getattr(client, "model", None):
        config.llm.default_model = client.model
    return client, provider_name


def list_models(session: ChatSession) -> int:
    try:
        models = session.list_models()
    except ModelClientError as exc:
        LOGGER.error("Failed to query models: %s", exc)
        return 1

    if not models:
        console.print(
            Panel(
                f"No models are currently available for the {session.provider_name} provider.",
                title="Models Unavailable",
                style="warning",
            )
        )
        return 0

    table = Table(title=f"{session.provider_name.title()} Models", box=None, highlight=True)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Name", style="magenta")
    for idx, model in enumerate(models, start=1):
        table.add_row(str(idx), model.name)
    console.print(table)
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    if args.env_file:
        os.environ["ENV_FILE"] = str(args.env_file)

    config = AppConfig.load(config_path=args.config_file)
    apply_overrides(config, args)

    try:
        session = build_session(config, args)
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        return 1
    except ModelClientError as exc:
        LOGGER.error("Failed to initialise provider client: %s", exc)
        return 1

    if args.list_models:
        return list_models(session)

    target_model = args.model or config.llm.default_model
    try:
        session.ensure_model(target_model)
    except ModelNotFoundError as exc:
        console.print(Panel(str(exc), title="Model Not Found", style="error"))
        return 1
    except ModelClientError as exc:
        console.print(
            Panel(
                f"Failed to contact {session.provider_name}: {exc}",
                title="Provider Error",
                style="error",
            )
        )
        return 1

    session.run_cli()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
