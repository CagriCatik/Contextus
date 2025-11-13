# Contextus

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![RAG](https://img.shields.io/badge/RAG-modular%20stack-6E40C9)](https://github.com/CagriCatik/Contextus)
[![FAISS](https://img.shields.io/badge/vector%20store-FAISS-009688)](https://github.com/facebookresearch/faiss)
[![Embeddings](https://img.shields.io/badge/embeddings-SentenceTransformers-FF6F00)](https://www.sbert.net/)
[![Ollama](https://img.shields.io/badge/inference-Ollama-000000?logo=ollama)](https://ollama.com)
[![OpenAI](https://img.shields.io/badge/inference-OpenAI-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Issues](https://img.shields.io/github/issues/CagriCatik/Contextus)](https://github.com/CagriCatik/Contextus/issues)
[![Last commit](https://img.shields.io/github/last-commit/CagriCatik/Contextus)](https://github.com/CagriCatik/Contextus/commits/main)

> Contextus is a modular retrieval-augmented generation stack for grounded question answering over local document collections.

It separates the workflow into:

1. **Ingestion** – use [MarkItDown](https://github.com/openai/markitdown) to normalise heterogeneous documents to Markdown/plain text, split them into overlapping chunks, embed each chunk with SentenceTransformers, and persist the vectors inside FAISS.
2. **Retrieval** – efficiently locate the most relevant chunks for a user question.
3. **Generation** – call either a locally hosted [Ollama](https://ollama.com) model or an [OpenAI](https://platform.openai.com) chat model with the retrieved context to produce grounded answers.

All moving pieces live in the `ragstack/` package so configuration, ingestion, and chat experiences stay modular and testable.

---

## 1. Prerequisites

- Python 3.10 or later.
- **For local inference:** [Ollama](https://ollama.com) installed and running (`ollama serve`).
- **For hosted inference:** an OpenAI account and API key with access to your preferred model.
- At least one LLM pulled locally (`ollama pull llama3`) or available in OpenAI.
- Optional GPU support for faster embedding generation.

Install the Python dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (or copy `.env.example`) to store secrets:

```bash
OPENAI_API_KEY=sk-your-openai-key
```

`.env` is loaded automatically before configuration is constructed.

---

## 2. Project structure

```bash
.
├── config.yaml            # Primary runtime configuration (provider, prompts, chunking…)
├── .env / .env.example    # Environment overrides (.env is ignored by Git)
├── ragstack/              # Core package: config, ingestion, retrieval, chat, providers
├── ingest_markdown.py     # Build the FAISS index from local documents via MarkItDown
├── chat_cli.py            # Interactive chat client with provider/model discovery
├── rag_core.py            # Thin wrapper around the retrieval stack
├── data/                  # Place documents here (Markdown, PDFs, text… auto-created)
└── index/                 # Persisted FAISS index and metadata (auto-created)
```

The public API is exposed through `ragstack/__init__.py`, making it easy to integrate the components into other projects or notebooks.

---

## 3. Configure the system

Settings live in `config.yaml` and can be overridden with environment variables, `.env`, or
CLI flags. The default configuration ships with sensible values:

```yaml
corpus:
  include:
    - "**/*.md"
    - "**/*.mdx"
    - "**/*.txt"
    - "**/*.pdf"
  exclude:
    - "**/.ipynb_checkpoints/**"
  use_markitdown: true
embeddings:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  device: auto
  # auto prefers CUDA > MPS > CPU; set to "cuda", "cuda:0", "mps", or "cpu" to pin the device
chunking:
  max_chars: 800
  min_chars: 200
  overlap: 100
retrieval:
  top_k: 5
  max_context_chars: 4000
llm:
  provider: ollama
  default_model: deepseek-r1:1.5b
ollama:
  host: http://localhost:11434
  timeout: 30
openai:
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
prompts:
  system_prompt: >-
    You are a focused assistant that answers using the retrieved knowledge base context.
    If the context is insufficient, admit it and suggest follow-up ingestion steps when appropriate.
  context_template: |-
    Here is the retrieved context from the knowledge base:

    {context}

    When answering, cite the relevant [source: ...] blocks if needed.
  question_template: "User question: {query}"
```

Key overrides and their environment variables:

| Setting | Environment variable | Description |
| --- | --- | --- |
| `corpus.include` / `corpus.exclude` | — | Control which files MarkItDown ingests (glob patterns). |
| `corpus.use_markitdown` | — | Disable to fall back to plain UTF-8 reads when MarkItDown is unavailable. |
| `embeddings.model_name` | `EMBEDDING_MODEL_NAME` | Swap the embedding model (e.g. to `sentence-transformers/all-mpnet-base-v2`). |
| `embeddings.device` | `EMBEDDING_DEVICE` | Force a specific Torch device (`cuda`, `cuda:0`, `mps`, `cpu`) or keep `auto` detection. |
| `chunking.max_chars` | `MAX_CHARS_PER_CHUNK` | Increase / decrease chunk size for longer or shorter passages. |
| `chunking.min_chars` | `MIN_CHARS_PER_CHUNK` | Ensure a minimum length for small documents. |
| `chunking.overlap` | `CHUNK_OVERLAP` | Control how much neighbouring chunks overlap. |
| `retrieval.top_k` | `RAG_TOP_K` | Number of chunks retrieved per query. |
| `retrieval.max_context_chars` | `RAG_MAX_CONTEXT_CHARS` | Character budget for the concatenated context. |
| `llm.provider` | `LLM_PROVIDER` | Choose `ollama` or `openai`. |
| `llm.default_model` | `LLM_MODEL` | Default model to preselect for the chosen provider. |
| `ollama.host` | `OLLAMA_HOST` | Base URL of the Ollama HTTP API. |
| `ollama.timeout` | `OLLAMA_TIMEOUT` | Request timeout (seconds). |
| `openai.model` | `OPENAI_MODEL` | OpenAI chat model (e.g. `gpt-4o-mini`). |
| `openai.base_url` | `OPENAI_BASE_URL` | Custom endpoint for compatible OpenAI deployments. |
| `prompts.*` | `SYSTEM_PROMPT`, `CONTEXT_TEMPLATE`, `QUESTION_TEMPLATE` | Customise the chat prompt templates. |

You can point the CLI to alternative configuration files with `--config-file` or to a different `.env` via `--env-file`.

---

## 4. Vectorise your documents

1. Copy or symlink all supported sources into the `data/` directory (or provide `--data-dir` to the ingestion CLI). By default MarkItDown handles Markdown, MDX, plain text, and PDFs; extend the `corpus.include` globs for other formats.
2. Run the ingestion pipeline:

   ```bash
   python ingest_markdown.py --data-dir ./data --index-dir ./index --index-name markdown_rag
   ```

   Optional overrides:

   - `--model`: select a different SentenceTransformers model.
   - `--device`: force embeddings to run on `cuda`, `cpu`, `mps`, or `auto`.
   - `--max-chars`, `--min-chars`, `--overlap`: tune chunk sizes on the command line.
   - `--log-level`: change verbosity (`DEBUG`, `INFO`, …).

   The script prints a summary similar to:

   ```bash
   [INFO] Embedding 128 chunks with sentence-transformers/all-MiniLM-L6-v2
   [INFO] Adding vectors to index markdown_rag
   [INFO] Created index 'markdown_rag' with 5 documents (128 chunks, 384 dimensions).
   ```

   Resulting artefacts:

- `index/markdown_rag.faiss` – FAISS index storing the vectors (loaded at query time).
- `index/markdown_rag_meta.json` – metadata with chunk text + provenance.

---

## 5. Explore the index (optional)

Diagnostic scripts help verify the ingestion results:

- `python inspect_index.py` – print index statistics and sample metadata rows.
- `python inspect_neighbors.py` – sample chunks and list their nearest neighbours.
- `python visualize_tsne.py` – launch a t-SNE plot (opens a Matplotlib window).

All scripts rely on the shared configuration, so you can point them at alternative index
locations with the same flags used by the main CLIs.

---

## 6. Chat with your Knowledge Base

**List models for the configured provider:**

```bash
# Use provider from config.yaml (default: Ollama)
python chat_cli.py --list-models

# Explicitly query OpenAI models (requires OPENAI_API_KEY)
python chat_cli.py --provider openai --list-models
```

**Start an interactive chat session:**

```bash
# Local Ollama example
python chat_cli.py --index-dir ./index --index-name markdown_rag --provider ollama --model deepseek-r1:1.5b

# Hosted OpenAI example
python chat_cli.py --provider openai --model gpt-4o-mini
```

**Highlights:**

- When `--model` is omitted the CLI lists available models and prompts for a choice (the first model is auto-selected for non-interactive shells).
- `--host` and `--timeout` override Ollama connectivity details; OpenAI uses `.env`.
- Retrieval depth and context size can be tuned per run via `--top-k` and `--max-context-chars`.
- Swap prompt + retrieval presets with `--task <profile>` to load overrides from `config.yaml`.
- Override embedding placement on the fly with `--embedding-device` (defaults to the configuration / auto-detection).
- The CLI is powered by [Rich](https://rich.readthedocs.io/), so model discovery, status messages, and responses render with colourful tables and panels.
- Customise system, context, and question prompts directly in `config.yaml` or via environment variables without touching code.

**Example conversation (Ollama provider):**

```bash
$ python chat_cli.py --index-dir ./index --index-name markdown_rag --provider ollama --model deepseek-r1:1.5b
────────────── RAG chat session. Press Enter on an empty line or Ctrl+D to exit. ──────────────

[prompt]You:[/prompt] What does the ingestion pipeline do?
[info]Retrieving context from the vector store...[/info]
[info]Calling ollama...[/info]

╭─ Assistant ─╮
│ The ingestion pipeline uses MarkItDown to convert documents, splits them into │
│ overlapping chunks, embeds them with sentence-transformers/all-MiniLM-L6-v2,  │
│ and stores the vectors in a FAISS index for fast retrieval.                   │
╰─────────────╯
```

Provider errors (missing models, authentication problems, connectivity) are surfaced as actionable messages instead of raw stack traces.

---

## 7. Custom prompts, embeddings, and chunking strategies

- **Chunking:** adjust `max_chars`, `min_chars`, and `overlap` in `config.yaml` (or via
  environment variables) to match the structure of your documents. Shorter chunks with more
  overlap help tight Q&A pairs; longer chunks preserve narrative flow.
- **Prompts:** edit the `prompts` block in `config.yaml` (or set
  `SYSTEM_PROMPT` / `CONTEXT_TEMPLATE` / `QUESTION_TEMPLATE`) to shape model behaviour, or
  create named `tasks` with prompt overrides for specific workflows (e.g. summarisation vs.
  troubleshooting).
- **Embedding backends:** set `embeddings.backend` to `sentence_transformer` (default) or to the
  dotted path of a custom subclass of `EmbeddingBackend`. Override `embeddings.model_name` or
  `EMBEDDING_MODEL_NAME` as needed.
- **Vector stores:** configure the `vector_store` block to switch backends or pass backend-specific
  parameters. Use `VECTOR_STORE_BACKEND` to override the backend from the environment.

---

## 9. Troubleshooting

| Symptom | Remedy |
| --- | --- |
| `Vector index 'markdown_rag' not found` | Run `python ingest_markdown.py` to build the index or point the CLI at the correct `--index-dir`. |
| `Authentication with OpenAI failed` | Ensure `OPENAI_API_KEY` is set in `.env` or the environment and that the key has access to the chosen model. |
| `Model 'xyz' is not installed` | Install it with `ollama pull xyz`, pick another model when prompted, or switch providers. |
| Embeddings still Contextusrt `device_name: cpu` | Set `embeddings.device: cuda` in `config.yaml` (or export `EMBEDDING_DEVICE=cuda`) and ensure PyTorch sees your GPU (`python -c "import torch; print(torch.cuda.is_available())"`). |
| Empty / irrelevant responses | Increase `--top-k`, adjust chunk sizes, or inspect the index with the diagnostic scripts. |

---

## 10. Extending the stack

- Plug in alternative embedding backends by subclassing `EmbeddingBackend` (or
  `SentenceTransformerEmbeddings`) and pointing `embeddings.backend` at the new class.
- Swap FAISS for another vector store by subclassing `VectorStore`, registering it via
  `register_vector_store`, and updating `vector_store.backend`.
- Add more providers by following the patterns in `ragstack/ollama.py` and
  `ragstack/openai_client.py` and wiring them through `ragstack/llm.py`.
- Tune prompts and retrieval parameters via named `tasks` in `config.yaml`, then select them with
  `--task` in the CLIs or `AppConfig.resolve_task()` in code.
