# Contextus

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![RAG](https://img.shields.io/badge/RAG-modular%20stack-6E40C9)](https://github.com/CagriCatik/Contextus)
[![FAISS](https://img.shields.io/badge/vector%20store-FAISS-009688)](https://github.com/facebookresearch/faiss)
[![Embeddings](https://img.shields.io/badge/embeddings-SentenceTransformers-FF6F00)](https://www.sbert.net/)
[![Ollama](https://img.shields.io/badge/inference-Ollama-000000?logo=ollama)](https://ollama.com)
[![OpenAI](https://img.shields.io/badge/inference-OpenAI-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Type hints](https://img.shields.io/badge/type%20hints-mypy-informational)](https://mypy-lang.org/)
[![Pre-commit](https://img.shields.io/badge/hooks-pre--commit-FAB040?logo=pre-commit)](https://pre-commit.com/)
[![Issues](https://img.shields.io/github/issues/CagriCatik/Contextus)](https://github.com/CagriCatik/Contextus/issues)
[![Last commit](https://img.shields.io/github/last-commit/CagriCatik/Contextus)](https://github.com/CagriCatik/Contextus/commits/main)

> Contextus is a modular retrieval-augmented generation stack for grounded question answering over local document collections.

It separates the workflow into:

1. **Ingestion** – use [MarkItDown](https://github.com/openai/markitdown) to normalise
   heterogeneous documents to Markdown/plain text, split them into overlapping chunks,
   embed each chunk with SentenceTransformers, and persist the vectors inside FAISS.
2. **Retrieval** – efficiently locate the most relevant chunks for a user question with score-aware reranking, adaptive token budgeting, and configurable similarity thresholds.
3. **Generation** – call either a locally hosted [Ollama](https://ollama.com) model or an
   [OpenAI](https://platform.openai.com) chat model with the retrieved context to produce
   grounded answers.
4. **Memory orchestration** – persist multi-session conversations in a semantic vector index, summarise long-term context, and reuse high-confidence answers through a semantic cache.

All moving pieces live in the `ragstack/` package so configuration, ingestion, and chat
experiences stay modular and testable.

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
├── rag_query_example.py   # Run retrieval without a chat model
├── data/                  # Place documents here (Markdown, PDFs, text… auto-created)
└── index/                 # Persisted FAISS index and metadata (auto-created)
```

The public API is exposed through `ragstack/__init__.py`, making it easy to integrate the
components into other projects or notebooks.

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
  max_context_tokens: 1200
  token_encoder: cl100k_base
  token_overhead: 200
  min_score: 0.2
  rerank_top_k: 8
memory:
  enabled: true
  max_memory_tokens: 512
  summary_tokens: 256
  search_top_k: 12
  max_memory_items: 6
  cache_min_score: 0.9
llm:
  provider: ollama
  default_model: llama3.2:latest
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
| `retrieval.max_context_tokens` | `RAG_MAX_CONTEXT_TOKENS` | Token budget (after subtracting `token_overhead`) used when building the context window. |
| `retrieval.token_encoder` | `RAG_TOKEN_ENCODER` | Tokenizer identifier (e.g. `cl100k_base`, or an LLM name supported by tiktoken). |
| `retrieval.fallback_chars_per_token` | `RAG_FALLBACK_CHARS_PER_TOKEN` | Character-to-token ratio used when tiktoken is unavailable. |
| `retrieval.token_overhead` | `RAG_TOKEN_OVERHEAD` | Reserve this many tokens to keep room for prompts and user input. |
| `retrieval.min_score` | `RAG_MIN_SCORE` | Discard retrieved chunks whose similarity falls below this threshold. |
| `retrieval.rerank_top_k` | `RAG_RERANK_TOP_K` | Re-embed and rerank the top-N chunks for better precision. |
| `retrieval.context_separator` | `RAG_CONTEXT_SEPARATOR` | Separator inserted between retrieved chunks. |
| `memory.enabled` | `MEMORY_ENABLED` | Enable (`true`) or disable (`false`) persistent conversational memory. |
| `memory.max_memory_tokens` | `MEMORY_MAX_TOKENS` | Token budget reserved for replaying persistent memory per query. |
| `memory.summary_tokens` | `MEMORY_SUMMARY_TOKENS` | Token budget for the rolling long-term memory summary. |
| `memory.search_top_k` | `MEMORY_TOP_K` | Maximum number of memory entries to retrieve before filtering. |
| `memory.max_memory_items` | `MEMORY_MAX_ITEMS` | Cap on the number of memory snippets injected into the context. |
| `memory.min_score` | `MEMORY_MIN_SCORE` | Filter out recalled memories whose similarity is below this threshold. |
| `memory.cache_min_score` | `MEMORY_CACHE_MIN_SCORE` | Minimum score required to reuse an answer directly from the semantic cache. |
| `memory.cache_ttl_minutes` | `MEMORY_CACHE_TTL_MINUTES` | Age (minutes) after which cached answers are pruned from the summary. |
| `memory.token_encoder` | `MEMORY_TOKEN_ENCODER` | Override the tokenizer used when budgeting persistent memory tokens. |
| `llm.provider` | `LLM_PROVIDER` | Choose `ollama` or `openai`. |
| `llm.default_model` | `LLM_MODEL` | Default model to preselect for the chosen provider. |
| `ollama.host` | `OLLAMA_HOST` | Base URL of the Ollama HTTP API. |
| `ollama.timeout` | `OLLAMA_TIMEOUT` | Request timeout (seconds). |
| `openai.model` | `OPENAI_MODEL` | OpenAI chat model (e.g. `gpt-4o-mini`). |
| `openai.base_url` | `OPENAI_BASE_URL` | Custom endpoint for compatible OpenAI deployments. |
| `prompts.*` | `SYSTEM_PROMPT`, `CONTEXT_TEMPLATE`, `QUESTION_TEMPLATE` | Customise the chat prompt templates. |
| `documentation.system_prompt` | `TEST_SPEC_SYSTEM_PROMPT` | System instruction used when creating test specs. |

You can point the CLI to alternative configuration files with `--config-file` or to a
different `.env` via `--env-file`.

---

## 4. Vectorise your documents

1. Copy or symlink all supported sources into the `data/` directory (or provide `--data-dir`
   to the ingestion CLI). By default MarkItDown handles Markdown, MDX, plain text, and PDFs;
   extend the `corpus.include` globs for other formats.
2. Run the ingestion pipeline:

   ```bash
   python ingest_markdown.py \
       --data-dir ./data \
       --index-dir ./index \
       --index-name markdown_rag
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

## 5. Persistent memory & adaptive context

Contextus now keeps track of multi-session conversations alongside the document index:

- **Vectorised episodic memory.** Each turn (`question → answer`) is embedded and stored in a
  dedicated FAISS index (`index/memory_context.faiss`), enabling semantic recall of previous
  exchanges alongside document retrieval.
- **Rolling long-term summary.** A compact Markdown summary is updated after every response and
  persisted to `index/memory_context_summary.json`, keeping the most recent `rolling_window`
  turns within the configured token budget.
- **Semantic answer cache.** High-confidence matches above `memory.cache_min_score` allow the
  chat layer to reuse answers instantly, improving latency for repeated questions while still
  logging the interaction (marked as `Origin: cached`).
- **Adaptive budgeting.** Memory snippets consume a reserved token budget before handing the
  remaining allowance to `ContextBuilder`, ensuring the prompt never exceeds the configured
  context window.

Tune the behaviour through `config.yaml` or CLI flags such as `--enable-memory`,
`--memory-max-tokens`, `--memory-top-k`, `--memory-max-items`, and
`--memory-cache-threshold`. Memory artefacts live next to the primary index and survive across
chat sessions.

---

## 6. Explore the index (optional)

Diagnostic scripts help verify the ingestion results:

- `python inspect_index.py` – print index statistics and sample metadata rows.
- `python inspect_neighbors.py` – sample chunks and list their nearest neighbours.
- `python visualize_tsne.py` – launch a t-SNE plot (opens a Matplotlib window).
- `python rag_query_example.py` – run a plain retrieval query without invoking an LLM.

Additional CLI switches expose the new retrieval optimisations:

- `--max-context-tokens`: enforce a model-aware token budget.
- `--token-encoder`: select the tokenizer used for estimation (falls back to heuristics when missing).
- `--min-score`: drop low-similarity results before building the context.
- `--rerank-top-k`: re-embed and rerank the top-N hits to prioritise high-confidence chunks.

All scripts rely on the shared configuration, so you can point them at alternative index
locations with the same flags used by the main CLIs.

---

## 7. Chat with your Knowledge Base

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
python chat_cli.py \
    --model llama3.2:latest \
    --embedding-device cuda \
    --top-k 6 \
    --max-context-chars 3500

# Hosted OpenAI example
python chat_cli.py \
    --provider openai \
    --model gpt-4o-mini
```

**Highlights:**

- When `--model` is omitted the CLI lists available models and prompts for a choice (the
  first model is auto-selected for non-interactive shells).
- `--host` and `--timeout` override Ollama connectivity details; OpenAI uses `.env`.
- Retrieval depth and context size can be tuned per run via `--top-k` and
  `--max-context-chars`.
- Swap prompt + retrieval presets with `--task <profile>` to load overrides from `config.yaml`.
- Override embedding placement on the fly with `--embedding-device` (defaults to
  the configuration / auto-detection).
- The CLI is powered by [Rich](https://rich.readthedocs.io/), so model discovery,
  status messages, and responses render with colourful tables and panels.
- Customise system, context, and question prompts directly in `config.yaml` or via
  environment variables without touching code.

**Example conversation (Ollama provider):**

```bash
$ python chat_cli.py --model llama3.2:latest
────────────── RAG chat session. Press Enter on an empty line or Ctrl+D to exit. ──────────────

[prompt]You:[/prompt] What does the ingestion pipeline do?
[info]Preparing augmented context...[/info]
[info]Calling ollama...[/info]

╭─ Assistant ─╮
│ The ingestion pipeline uses MarkItDown to convert documents, splits them into │
│ overlapping chunks, embeds them with sentence-transformers/all-MiniLM-L6-v2,  │
│ and stores the vectors in a FAISS index for fast retrieval.                   │
╰─────────────╯
```

Follow-up questions that cross the `memory.cache_min_score` threshold are served instantly
from persistent memory and labelled with a `Memory Hit` panel.

Provider errors (missing models, authentication problems, connectivity) are surfaced as
actionable messages instead of raw stack traces.

---

## 8. Custom prompts, embeddings, and chunking strategies

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
