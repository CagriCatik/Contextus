"""Command-line utility to build the FAISS index from local documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.panel import Panel

from ragstack.config import AppConfig
from ragstack.console import console
from ragstack.logging import configure_logging, get_logger
from ragstack.pipeline import IndexBuilder

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert documents with MarkItDown and ingest them into the FAISS index."
    )
    parser.add_argument("--data-dir", type=Path, help="Directory containing source documents.")
    parser.add_argument("--index-dir", type=Path, help="Directory where the FAISS index is stored.")
    parser.add_argument("--index-name", default="markdown_rag", help="Name of the FAISS index to create.")
    parser.add_argument(
        "--model",
        help="SentenceTransformers embedding model to use (default: from EMBEDDING_MODEL_NAME env var).",
    )
    parser.add_argument(
        "--device",
        help="Torch device for embeddings (auto, cpu, cuda, cuda:0, mps).",
    )
    parser.add_argument("--max-chars", type=int, help="Maximum characters per chunk.")
    parser.add_argument("--min-chars", type=int, help="Minimum characters per chunk.")
    parser.add_argument("--overlap", type=int, help="Character overlap between consecutive chunks.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO).")
    return parser.parse_args()


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    if args.data_dir:
        config.paths.data_dir = args.data_dir.resolve()
        config.paths.data_dir.mkdir(parents=True, exist_ok=True)
    if args.index_dir:
        config.paths.index_dir = args.index_dir.resolve()
        config.paths.index_dir.mkdir(parents=True, exist_ok=True)
    if args.model:
        config.embeddings.model_name = args.model
    if args.device:
        config.embeddings.device = args.device
    if args.max_chars:
        config.chunking.max_chars = args.max_chars
    if args.min_chars:
        config.chunking.min_chars = args.min_chars
    if args.overlap:
        config.chunking.overlap = args.overlap


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    config = AppConfig.load()
    apply_overrides(config, args)

    builder = IndexBuilder(config, index_name=args.index_name)

    try:
        with console.status("[info]Building FAISS index...[/info]"):
            stats = builder.build()
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        return 1

    console.print(
        Panel(
            f"Created index '[bold]{args.index_name}[/]' with [bold]{stats.documents}[/] documents\n"
            f"([bold]{stats.chunks}[/] chunks, {stats.dimension} dimensions).",
            title="Ingestion Complete",
            style="success",
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
