# Model and Retrieval Pipeline

This document drills into the runtime flow that powers ingestion, retrieval, and LLM interaction. It is organised by phase so
operators can reason about throughput, latency, and potential scaling strategies.

## 1. Ingestion Phase

1. **Document discovery** – `MarkdownCorpus.list_documents()` finds source material under `paths.data_dir` using include/exclude
   glob patterns. Optional MarkItDown conversion ensures PDFs and other rich formats are normalised into Markdown before
   chunking.
2. **Plain-text normalisation** – `strip_markdown()` removes code blocks, inline code fences, images, headings, and redundant
   whitespace to keep embeddings focused on natural language.
3. **Chunk generation** – `Chunker.iter_chunks()` creates overlapping windows of `max_chars` with a minimum length guard and
   a configurable overlap to preserve cross-chunk continuity. The overlap and window size directly influence recall and
   downstream context length.
4. **Embedding computation** – `SentenceTransformerEmbeddings` loads the configured model on the requested device (auto detects
   CUDA/MPS/CPU) and outputs unit-normalised vectors for stability with inner-product FAISS indices.
5. **Vector persistence** – `FaissVectorStore.add()` stores embeddings and aligns them with `VectorRecord` metadata so queries can
   recover source filenames and chunk identifiers. Metadata is serialised to `<index>_meta.json` during `save()`.

## 2. Retrieval Phase

1. **Query embedding** – `ContextBuilder.retrieve()` calls `embed_query()` to transform natural-language questions into the same
   vector space used during ingestion.
2. **Similarity search + reranking** – `VectorStore.search()` performs inner-product similarity (FAISS `IndexFlatIP`) and
   returns top-K `(score, VectorRecord)` tuples. When `retrieval.rerank_top_k` is non-zero, the highest-scoring hits are
   re-embedded and re-scored against the query vector to tighten precision.
3. **Score filtering** – `retrieval.min_score` drops low-confidence chunks before they reach the prompt stage, ensuring the
   remaining budget emphasises relevant passages.
4. **Context assembly** – `ContextBuilder.build_context()` concatenates formatted blocks with configurable separators.
   Character budgets (`max_context_chars`) and token budgets (`max_context_tokens` with an optional `token_overhead`) prevent
   overruns while still allowing partial chunks when the final block would otherwise overflow. When persistent memory is enabled,
   retrieval only consumes the budget that remains after memory snippets are injected.

## 3. Generation Phase

1. **Prompt construction** – `ChatRequest.to_messages()` builds a three-message sequence: the system prompt, a context block, and
   the user query. Prompts are taken from `PromptConfig`, allowing task profiles to customise tone and behaviour without code
   changes.
2. **Provider invocation** – The chat session delegates to a provider client implementing the `SupportsModels` protocol. Ollama
   and OpenAI clients both expose `list_models()`, `ensure_model()`, and `chat()`.
3. **Response handling** – Rich console renderers display Markdown answers in the CLI. Downstream integrations (e.g. documentation
   generation) can reuse the same `ChatSession` mechanics to guarantee consistent prompt formatting.

## 4. Persistent Memory Phase

1. **Semantic cache check** – `ConversationMemory.maybe_answer()` probes the memory index for high-similarity matches. If the
   best score meets or exceeds `memory.cache_min_score`, the stored answer is returned immediately.
2. **Memory context budgeting** – `ConversationMemory.build_memory_context()` assembles a rolling summary and top episodic
   memories while staying within `memory.max_memory_tokens` and `memory.max_memory_items`.
3. **Turn persistence** – `ConversationMemory.remember()` appends the interaction to an on-disk log, refreshes the summary
   (bounded by `memory.summary_tokens`), and writes new vectors to the FAISS memory index unless the answer came from cache.

## Latency and Throughput Considerations

- **Batching ingestion** – For large corpora, split ingestion by directory and merge FAISS indices offline or stream chunk batches
  through `VectorStore.add()` to avoid loading all vectors into memory simultaneously.
- **Embedding hardware** – Set `embeddings.device` to GPU-backed options where available. Auto-detection falls back to CPU, which
  may impact build times for large document sets.
- **Similarity search scaling** – `IndexFlatIP` provides exact search. If corpus sizes grow into millions of vectors, consider
  swapping to an ANN backend (e.g., FAISS IVF/HNSW) via `vector_store.backend` to improve latency.
- **Cold-start caching** – Initialising embedding models and FAISS indices has non-trivial cost. Pre-load these components in long
  running services or reuse them across requests to avoid repeated initialisation.

## Context Window Strategies

- **Token-aware budgets** – Configure `retrieval.max_context_tokens` with a matching `token_encoder` (or rely on the heuristic
  fallback) to keep contexts within provider limits. Reserve space for prompts and user text via `retrieval.token_overhead`.
- **Summarisation fallback** – When contexts exceed provider limits, fall back to summarising low-ranked chunks before appending
  them, or use a two-pass retrieval (coarse recall, rerank, summarise) to stay within window constraints.
- **Persistent memory** – `ConversationMemory` maintains a semantic cache and rolling summary. Adjust `memory.max_memory_tokens`,
  `memory.max_memory_items`, and cache thresholds to balance recall quality against latency.

## Observability Hooks

- Enable debug logging (`--log-level DEBUG`) to emit embedding model/device information, retrieval diagnostics, and provider
  responses. Instrumentation points exist in `ragstack.logging` and at major pipeline steps in `IndexBuilder` and
  `ContextBuilder`.
- Track retrieval quality by sampling queries and inspecting `[source: ... | score: ...]` headers. Sudden score drops may indicate
  drift between embeddings and stored vectors (e.g., after changing models without re-ingesting data).
