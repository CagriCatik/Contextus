"""Utility script to inspect the FAISS index and metadata."""

from __future__ import annotations

import faiss
import numpy as np
from rich.panel import Panel
from rich.table import Table

from ragstack.config import AppConfig
from ragstack.console import console
from ragstack.store import create_vector_store


INDEX_NAME = "markdown_rag"


def section_title(title: str) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def describe_basic(index) -> None:
    section_title("FAISS INDEX INFO")

    info_panel = Panel(
        f"Type: [bold]{type(index).__name__}[/]\n"
        f"Dimension: [bold]{index.d}[/]\n"
        f"Vectors: [bold]{index.ntotal}[/]\n"
        f"Trained: [bold]{getattr(index, 'is_trained', 'N/A')}[/]",
        title="Basic Properties",
        style="info",
    )
    console.print(info_panel)

    summary_panel = Panel(
        str(index),
        title="Index Summary (faiss pretty printer)",
        style="info",
    )
    console.print(summary_panel)


def describe_metric(index) -> None:
    section_title("METRIC INFO")

    metric_type = getattr(index, "metric_type", None)
    if metric_type is None:
        console.print("[yellow]metric_type: <not available on this index>[/yellow]")
        return

    candidates = [
        ("METRIC_L2", "L2 (squared Euclidean)"),
        ("METRIC_INNER_PRODUCT", "Inner product / cosine-compatible"),
        ("METRIC_L1", "L1"),
        ("METRIC_Linf", "Linf"),
        ("METRIC_Canberra", "Canberra"),
        ("METRIC_BrayCurtis", "BrayCurtis"),
        ("METRIC_Jaccard", "Jaccard"),
        ("METRIC_Hamming", "Hamming"),
        ("METRIC_Substructure", "Substructure"),
        ("METRIC_Superstructure", "Superstructure"),
    ]

    metric_map: dict[int, str] = {}
    for attr_name, label in candidates:
        value = getattr(faiss, attr_name, None)
        if value is not None:
            metric_map[value] = label

    readable = metric_map.get(metric_type, "<unknown>")

    table = Table(show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("metric_type (raw)", str(metric_type))
    table.add_row("metric_type (label)", readable)

    console.print(table)


def describe_ivf(index) -> None:
    if not isinstance(index, faiss.IndexIVF):
        return

    section_title("IVF SPECIFIC")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("nlist (cells)", str(index.nlist))
    table.add_row("nprobe", str(index.nprobe))
    table.add_row("code_size", str(getattr(index, "code_size", None)))

    try:
        quantizer = index.quantizer
        table.add_row("quantizer type", repr(type(quantizer)))
        table.add_row("quantizer d", str(quantizer.d))
    except Exception as e:
        table.add_row("quantizer info", f"<error> {repr(e)}")

    console.print(table)


def describe_hnsw(index) -> None:
    if not hasattr(index, "hnsw"):
        return

    section_title("HNSW SPECIFIC")

    hnsw = index.hnsw
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("M (neighbors)", str(getattr(hnsw, "M", None)))
    table.add_row("efConstruction", str(getattr(hnsw, "efConstruction", None)))
    table.add_row("efSearch", str(getattr(hnsw, "efSearch", None)))

    console.print(table)


def describe_pq(index) -> None:
    if not hasattr(index, "pq"):
        return

    section_title("PQ SPECIFIC")

    pq = index.pq
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("m (subvectors)", str(getattr(pq, "M", None)))
    table.add_row("ksub (centers per subvector)", str(getattr(pq, "ksub", None)))
    table.add_row("code_size (bytes)", str(getattr(pq, "code_size", None)))
    table.add_row("dsub (dims per subvector)", str(getattr(pq, "dsub", None)))

    console.print(table)


def inspect_sample_vectors(index, n: int = 3) -> None:
    if index.ntotal == 0:
        return

    section_title("SAMPLE VECTORS (RECONSTRUCT)")

    n = min(n, index.ntotal)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Index", style="bold", justify="right")
    table.add_column("First 8 dims")

    for i in range(n):
        try:
            vec = index.reconstruct(i)
        except Exception as e:
            console.print(
                Panel(
                    f"Could not reconstruct vector {i}: {repr(e)}",
                    border_style="red",
                    title="Reconstruct Error",
                )
            )
            break
        first_dims = ", ".join(f"{x:.4f}" for x in vec[:8])
        table.add_row(str(i), first_dims)

    console.print(table)


def test_random_search(index, k: int = 5) -> None:
    section_title("TEST SEARCH (RANDOM QUERY)")

    if index.ntotal == 0:
        console.print("[yellow]Index is empty, skipping search.[/yellow]")
        return

    d = index.d
    xq = np.random.randn(1, d).astype("float32")

    metric_ip = getattr(faiss, "METRIC_INNER_PRODUCT", -1)
    if getattr(index, "metric_type", None) == metric_ip:
        faiss.normalize_L2(xq)

    distances, ids = index.search(xq, k)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Query shape", str(xq.shape))
    table.add_row("Top k", str(k))
    table.add_row("IDs", str(ids.tolist()))
    table.add_row("Distances", str(distances.tolist()))

    console.print(table)


def show_metadata(store) -> None:
    section_title("METADATA SAMPLE")

    metadata = store.metadata.records
    console.print(f"Total metadata entries: [bold]{len(metadata)}[/]")

    if metadata:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", justify="right")
        table.add_column("Source")
        table.add_column("Chunk ID", justify="right")
        table.add_column("Preview", overflow="fold")

        for i, record in enumerate(metadata[:3]):
            preview = record.text[:120] + ("..." if len(record.text) > 120 else "")
            table.add_row(
                str(i),
                record.source_file,
                str(record.chunk_id),
                preview,
            )

        console.print(table)
    else:
        console.print("[yellow]No metadata records found.[/yellow]")


def check_metadata_consistency(store, index) -> None:
    metadata_len = len(store.metadata.records)
    if metadata_len != index.ntotal:
        console.print(
            Panel(
                f"Metadata length ({metadata_len}) does not match ntotal ({index.ntotal}).",
                title="Warning",
                style="warning",
            )
        )
    else:
        console.print(Panel("Metadata and index size are consistent.", style="success"))


def main() -> None:
    config = AppConfig.load()

    try:
        store = create_vector_store(config, INDEX_NAME, dim=None)
    except ValueError:
        raise SystemExit(
            f"Index '{INDEX_NAME}' not found in {config.paths.index_dir}. Run ingest_markdown.py first."
        )

    index = store.index

    describe_basic(index)
    describe_metric(index)
    describe_ivf(index)
    describe_hnsw(index)
    describe_pq(index)
    inspect_sample_vectors(index, n=3)
    test_random_search(index, k=5)
    show_metadata(store)
    check_metadata_consistency(store, index)


if __name__ == "__main__":  # pragma: no cover - diagnostic script
    main()
