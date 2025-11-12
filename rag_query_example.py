"""Example script that queries the FAISS index without calling Ollama."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from ragstack.config import AppConfig
from ragstack.console import console
from ragstack.embedding import create_embeddings
from ragstack.store import create_vector_store


def main() -> None:
    config = AppConfig.load()
    embeddings = create_embeddings(config.embeddings)

    try:
        store = create_vector_store(config, name="markdown_rag", dim=None)
    except ValueError:
        raise SystemExit("Index not found. Run ingest_markdown.py before querying.")

    query = "Give me an overview of the documentation content."
    vector = embeddings.embed_query(query)
    results = store.search(vector, k=config.retrieval.top_k)

    console.print(Panel(f"[bold]Query[/]: {query}", style="info"))
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    table.add_column("Chunk", justify="right")
    table.add_column("Preview", overflow="fold")
    for score, record in results:
        table.add_row(
            f"{score:.4f}",
            record.source_file,
            str(record.chunk_id),
            record.text[:400] + ("..." if len(record.text) > 400 else ""),
        )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover - sample script
    main()
