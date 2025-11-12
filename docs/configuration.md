# Configuration Reference

Contextus centralises configuration in `config.yaml` and the `AppConfig` dataclasses located in `ragstack/config.py`. This
reference explains each section, environment overrides, and task profile usage.

## File Locations

- **Primary file:** `config.yaml` in the project root. JSON is also supported via `config.json`.
- **Environment overrides:** `.env` file (loaded once) or direct environment variables.
- **CLI overrides:** Command-line flags provided by scripts such as `ingest_markdown.py` and `chat_cli.py`.

## Core Sections

| Section | Key Fields | Description |
|---------|------------|-------------|
| `paths` | `data_dir`, `index_dir` | Controls where source documents live and where FAISS indices are written. Automatically created if missing. |
| `corpus` | `include`, `exclude`, `use_markitdown` | Glob patterns governing which files are ingested and whether MarkItDown is used for rich-document conversion. |
| `embeddings` | `backend`, `model_name`, `device` | Embedding backend selection. Default is SentenceTransformers with automatic device detection. |
| `chunking` | `max_chars`, `min_chars`, `overlap` | Character-based windowing parameters for document chunking. |
| `retrieval` | `top_k`, `max_context_chars`, `max_context_tokens`, `token_encoder`, `token_overhead`, `min_score`, `rerank_top_k` | Retrieval depth plus the budgets and filters applied when assembling the prompt context. |
| `vector_store` | `backend`, `parameters` | Vector store backend (default FAISS). Extra parameters are forwarded to the backend constructor. |
| `ollama` | `host`, `timeout` | Connection details for the Ollama HTTP API. |
| `openai` | `api_key`, `model`, `base_url`, `organization` | OpenAI client settings. `api_key` can be omitted in YAML and provided via environment variables. |
| `llm` | `provider`, `default_model` | Default chat provider and model name. |
| `prompts` | `system_prompt`, `context_template`, `question_template` | Prompt templates shared by chat and documentation workflows. |
| `documentation` | `test_case_id_prefix`, `starting_index`, `system_prompt` | Settings specific to automated test specification generation. |
| `tasks` | arbitrary mapping | Named overrides for retrieval/prompt fields, enabling persona-specific behaviour. |

## Environment Variable Overrides

Set any of the following environment variables (directly or via `.env`) to override YAML values:

- `EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE`
- `MAX_CHARS_PER_CHUNK`, `MIN_CHARS_PER_CHUNK`, `CHUNK_OVERLAP`
- `RAG_TOP_K`, `RAG_MAX_CONTEXT_CHARS`, `RAG_MAX_CONTEXT_TOKENS`, `RAG_TOKEN_ENCODER`, `RAG_FALLBACK_CHARS_PER_TOKEN`, `RAG_TOKEN_OVERHEAD`, `RAG_MIN_SCORE`, `RAG_RERANK_TOP_K`, `RAG_CONTEXT_SEPARATOR`
- `VECTOR_STORE_BACKEND`
- `OLLAMA_HOST`, `OLLAMA_TIMEOUT`
- `LLM_PROVIDER`, `LLM_MODEL`
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`, `OPENAI_ORG`
- `SYSTEM_PROMPT`, `CONTEXT_TEMPLATE`, `QUESTION_TEMPLATE`
- `TEST_CASE_ID_PREFIX`, `TEST_CASE_START_INDEX`, `TEST_SPEC_SYSTEM_PROMPT`

Environment overrides take effect when `AppConfig.load()` is called. CLI scripts automatically load `.env` once per process to
avoid redundant filesystem access.

## Task Profiles

Define persona-specific overrides under the `tasks` section:

```yaml
tasks:
  default:
    retrieval:
      top_k: 5
      max_context_chars: 4000
  long_form_answering:
    retrieval:
      top_k: 8
      max_context_chars: 6000
    prompts:
      system_prompt: >-
        You are an analytical assistant who writes exhaustive answers grounded in the provided context.
        Explicitly cite the most relevant sources and mention when information is missing.
```

CLI scripts expose a `--task` flag that calls `AppConfig.resolve_task()` to merge the base configuration with the selected
profile. Use this feature to tailor retrieval depth, tone, or formatting per integration, including token budgets
(`max_context_tokens`), reranking parameters, or similarity thresholds for specialised workloads.

## Secrets Management

Keep provider credentials out of source control by defining them in `.env` or runtime environments. Example `.env` entries:

```
OPENAI_API_KEY=sk-...
OLLAMA_HOST=http://ollama.example.com:11434
```

## Configuration Validation Tips

- Run `python ingest_markdown.py --log-level DEBUG` to confirm that data/index directories resolve correctly and that embedding
  devices match expectations.
- After updating embedding models or chunking parameters, re-run ingestion to avoid stale vectors.
- If `create_vector_store()` raises a `ValueError`, double-check that the FAISS index files exist at `paths.index_dir` and that
  the index dimensionality matches the embedding output.
