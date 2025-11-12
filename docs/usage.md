# Usage Guide

This guide outlines how to ingest content, query the RAG index, and tune behaviour for different scenarios.

## 1. Prepare the Environment

1. Install dependencies: `pip install -r requirements.txt`.
2. Populate a `.env` file with provider credentials (e.g., `OPENAI_API_KEY`) and optional overrides such as `OLLAMA_HOST`.
3. Place source documents inside the directory referenced by `paths.data_dir` (default: `data/`). Supported formats include
   Markdown, text, and—when MarkItDown is available—PDF or DOCX.

## 2. Build the Index

```bash
python ingest_markdown.py --data-dir ./data --index-dir ./index --index-name markdown_rag
```

Key flags:

- `--model` / `--device` – override the embedding model and execution device.
- `--max-chars`, `--min-chars`, `--overlap` – control chunking behaviour.
- `--log-level DEBUG` – inspect ingestion progress, embedding device selection, and FAISS metadata output.

## 3. Chat with the Index

```bash
python chat_cli.py --index-dir ./index --index-name markdown_rag --provider ollama --model llama3.2:latest
```

Additional tips:

- Use `--task long_form_answering` (defined in `config.yaml`) to pull deeper context for expansive answers.
- Pass `--list-models` to enumerate installed provider models before starting the session.
- Override retrieval depth temporarily with `--top-k`, `--max-context-chars`, and `--max-context-tokens` to experiment with
  answer quality while staying under provider limits.
- Combine `--token-encoder` with `--max-context-tokens` when working with hosted models that expose strict token windows.
- Use `--min-score` and `--rerank-top-k` to filter noisy matches and prioritise high-confidence evidence at query time.

## 4. Generate Documentation

If the optional documentation tooling is present, generate structured test cases using:

```bash
python generate_test_spec.py --task default --output docs/test_plan.md
```

The generator shares the same retrieval pipeline as chat, ensuring consistent grounding across artefacts.

## 5. Troubleshooting Checklist

- **Missing index files:** Re-run ingestion and confirm that `index/<name>.faiss` and `<name>_meta.json` are created.
- **Provider authentication errors:** Verify `.env` entries and ensure environment variables are exported when running scripts.
- **Low-quality retrieval:** Inspect `[source: ... | score: ...]` headers in responses. If scores are uniformly low, consider
  raising `--min-score`, increasing `--rerank-top-k`, re-ingesting with larger `max_chars`, or upgrading the embedding model.
- **Latency spikes:** Check whether embedding device auto-detection fell back to CPU; specify `--embedding-device cuda` when
  GPUs are available.

## 6. Extending the Stack

- Implement custom embedding or vector store backends and register them via `ragstack.embedding.register_embedding_backend`
  or `ragstack.store.register_vector_store`.
- Create new task profiles in `config.yaml` to experiment with prompt tone, retrieval budgets, or context formatting for
  downstream applications.
- For production services, wrap `ChatSession` in an API layer and cache the loaded embedding model and FAISS index to amortise
  initialisation cost across requests.
