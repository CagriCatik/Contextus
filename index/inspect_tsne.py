"""Visualise embeddings in the FAISS index using t-SNE."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from rich.panel import Panel

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
        raise SystemExit("Index not found. Run ingest_markdown.py before visualising.")

    index = store.index
    metadata = store.metadata.records

    vectors = extract_vectors(index)
    console.print(
        Panel(
            f"Vectors: [bold]{index.ntotal}[/]\nDimension: [bold]{index.d}[/]\n"
            f"Tensor shape: [bold]{vectors.shape}[/]",
            title="Index Overview",
            style="info",
        )
    )

    max_points = 1000
    if vectors.shape[0] > max_points:
        idx = np.random.choice(vectors.shape[0], max_points, replace=False)
        vectors_sub = vectors[idx]
        metadata_sub = [metadata[i] for i in idx]
    else:
        vectors_sub = vectors
        metadata_sub = metadata

    console.print(f"Using [bold]{vectors_sub.shape[0]}[/] points for t-SNE")

    tsne = TSNE(n_components=2, perplexity=30, init="random", verbose=1)
    emb_2d = tsne.fit_transform(vectors_sub)

    files = [record.source_file for record in metadata_sub]
    unique_files = sorted(set(files))
    file_to_id = {name: idx for idx, name in enumerate(unique_files)}
    colors = [file_to_id[name] for name in files]

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab20")
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=10, c=colors, cmap=cmap)

    handles = []
    labels = []
    for name, idx in list(file_to_id.items())[:10]:
        handles.append(
            plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=cmap(idx % cmap.N))
        )
        labels.append(name)
    if handles:
        plt.legend(handles, labels, title="source_file (subset)", loc="best")

    plt.title("t-SNE visualization of markdown_rag embeddings")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  # pragma: no cover - visualization script
    main()
