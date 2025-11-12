"""Inspect nearest neighbors in the FAISS index for manual validation."""

from __future__ import annotations

import random

import numpy as np
from rich.panel import Panel
from rich.table import Table

from ragstack.config import AppConfig
from ragstack.console import console
from ragstack.store import create_vector_store


def extract_vectors(index) -> np.ndarray:
    ntotal = index.ntotal
    dim = index.d
    xb = np.zeros((ntotal, dim), dtype="float32")
    for i in range(ntotal):
        xb[i] = index.reconstruct(i)
    return xb


def main() -> None:
    config = AppConfig.load()

    try:
        store = create_vector_store(config, name="markdown_rag", dim=None)
    except ValueError:
        raise SystemExit("Index not found. Run ingest_markdown.py before inspecting neighbors.")

    index = store.index
    metadata = store.metadata.records

    vectors = extract_vectors(index)
    num_samples = min(5, index.ntotal)
    k = 5

    console.print(
        Panel(
            f"Vectors: [bold]{index.ntotal}[/]\nDimension: [bold]{index.d}[/]",
            title="Index Summary",
            style="info",
        )
    )

    for _ in range(num_samples):
        i = random.randint(0, index.ntotal - 1)
        query_vec = vectors[i].astype("float32").reshape(1, -1)

        scores, indices = index.search(query_vec, k)
        anchor = metadata[i]
        console.print(
            Panel(
                f"Source: [bold]{anchor.source_file}[/]\n"
                f"Chunk ID: [bold]{anchor.chunk_id}[/]\n"
                f"Preview: {anchor.text[:200]}...",
                title=f"Anchor #{i}",
                style="magenta",
            )
        )

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Idx", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Source")
        table.add_column("Chunk", justify="right")
        table.add_column("Preview", overflow="fold")
        for score, idx in zip(scores[0], indices[0]):
            neighbor = metadata[idx]
            table.add_row(
                str(idx),
                f"{score:.4f}",
                neighbor.source_file,
                str(neighbor.chunk_id),
                neighbor.text[:200] + ("..." if len(neighbor.text) > 200 else ""),
            )
        console.print(table)
        console.print()


if __name__ == "__main__":  # pragma: no cover - diagnostic script
    main()
