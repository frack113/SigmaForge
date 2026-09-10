# SigmaForge

A local RAG system for Sigma detection rules.

## WIP

This project is under active development. Some features may be incomplete.

## Architecture

- **Backend**: FastAPI + Jinja2 templates
- **Database**: DuckDB (metadata, worker state, config)
- **Vector Store**: Qdrant (auto-managed subprocess)
- **LLM**: llama.cpp (runs on `127.0.0.1:8080`)
- **RAG Pipeline**: LlamaIndex
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)

## Quick Start

```bash
# Start server (auto-initializes on first run)
uv run python main.py
```

Server starts on `http://localhost:8000` .


## Configuration

- Config managed via the web UI Config page (stored in DuckDB)

### AirGap / Offline Mode

Set `HF_HUB_OFFLINE=1` to prevent HuggingFace API requests (useful for fully
disconnected deployments):

```bash
HF_HUB_OFFLINE=1 uv run python main.py
```

By default, the app sets `HF_HUB_OFFLINE=1` (via `os.environ.setdefault` in
`src/main.py`) to avoid accidental network calls in an AirGap context.

Functions that **explicitly need online access** (model search, model info,
GGUF file listing) temporarily remove the env var during the API call, then
restore it — so searching and downloading models works even in AirGap mode.

## Testing

```bash
uv run pytest              # Run all tests
uv run pytest tests/path -v # Single test
uv run ruff check .        # Lint
uv run mypy .              # Typecheck
```



## Icon

Use Icône de Pense créatif l&#039;inspiration ampoule by Sumit Saengthong on <a href="https://icon-icons.com/fr/authors/940-sumit-saengthong">Icon-Icons.com</a>