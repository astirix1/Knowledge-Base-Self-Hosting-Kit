# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Knowledge Base Self-Hosting Kit (Community Edition)** - Production-ready RAG system combining Docling document processing with ChromaDB vector storage. Extracted from Mail Modul Alpha, this is a fully-featured RAG implementation for self-hosting.

## Core Architecture

### Service Layers

1. **FastAPI Backend** (`backend/src/`)
   - Single entry point: `main.py` with lifespan management
   - API endpoints in `api/v1/rag/` organized by function
   - ChromaDB connection managed via singleton pattern in `core/chroma_manager.py`
   - Uses **resilience patterns**: circuit breaker + retry logic for ChromaDB operations

2. **Deployment Architecture**
   - **nginx gateway** (port 8080) - single external port for all services
   - **backend** (internal port 8080) - FastAPI application
   - **chromadb** (internal port 8000) - vector database
   - All services communicate via internal Docker networks

3. **Core Components** (`backend/src/core/`)
   - `chroma_manager.py` - Singleton ChromaDB client with connection pooling, health checks, circuit breaker
   - `docling_loader.py` - Document parsing (PDF, DOCX, PPTX, XLSX, HTML, Markdown)
   - `query_engine.py` - Query operations with multi-collection support
   - `retrievers/` - Hybrid search (vector + BM25), parent-document retrieval, reranking
   - `config.py` - Multi-LLM configuration (Ollama, OpenAI, Anthropic, Gemini) with hot-reload support
   - `feature_limits.py` - Edition tiers (Community has unlimited features)

4. **Services** (`backend/src/services/`)
   - Document processing: classification, extraction, folder scanning
   - Generators: summaries, ragignore, collection configs
   - Ingestion pipeline v2: async document processing

### Key Design Patterns

**Singleton Pattern**: ChromaManager ensures single ChromaDB connection instance throughout app lifecycle
- Access via `get_chroma_manager()` function
- Configured during FastAPI lifespan startup
- Properly closed during shutdown to prevent CLOSE_WAIT connections

**Lifespan Management**: FastAPI `@asynccontextmanager` coordinates startup/shutdown
- Startup: Configure ChromaDB → Test connection → Yield to app
- Shutdown: Close connections → Cleanup resources
- See `main.py` lines 28-68

**Resilience Patterns**:
- Circuit breaker for ChromaDB operations (5 failure threshold, 30s recovery)
- Exponential backoff retry (1s, 2s, 4s delays)
- Connection pooling with health check caching (30s TTL)

**Hot-Reload Config**: Configuration reads from `.env` file via `config_service`
- Set `use_hot_reload=True` to read fresh config without restart
- Used for LLM provider switching without downtime

## Common Development Commands

### Docker Development (Recommended)

```bash
# Setup and start services
./setup.sh

# Or manually:
docker compose up -d --build

# View logs
docker compose logs -f backend
docker compose logs -f chromadb

# Restart after code changes (hot-reload enabled)
docker compose restart backend

# Full rebuild (only when changing requirements.txt)
docker compose down
docker compose up -d --build

# Health check
curl http://localhost:8080/health

# Stop services
docker compose down
```

### Local Development (without Docker)

```bash
cd backend
pip install -r requirements.txt

# Requires ChromaDB and Ollama running separately
uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload
```

### Running Ollama (Required for Embeddings)

```bash
# Start Ollama server
ollama serve

# Pull embedding model
ollama pull nomic-embed-text

# Pull LLM model (optional for query generation)
ollama pull llama3.2:latest

# Test Ollama connection
curl http://localhost:11434/api/tags
```

## API Structure

All endpoints under `/api/v1/rag/`:

**Collections** (`collections.py`)
- `POST /collections` - Create collection with embeddings config
- `GET /collections` - List all collections
- `GET /collections/{name}/stats` - Collection statistics
- `DELETE /collections/{name}` - Delete collection

**Documents** (`documents/`)
- `POST /documents/upload` - Upload files (single/batch)
- `GET /documents` - List documents in collection
- `DELETE /documents/{id}` - Delete document

**Query** (`query.py`)
- `POST /query` - RAG query with hybrid search

**Ingestion** (`ingestion/`)
- `POST /ingestion/scan-folder` - Scan folder for compatible files
- `POST /ingestion/ingest-batch` - Batch document ingestion
- `POST /ingestion/ingest-folder` - Synchronous folder ingestion
- `GET /ingestion/status/{task_id}` - Poll ingestion status

**Cockpit** (`cockpit.py`)
- `GET /cockpit/system-status` - System health and metrics

## Configuration

**Environment Variables** (`.env`)

Key settings:
- `LLM_PROVIDER`: ollama, openai, anthropic, gemini
- `LLM_MODEL`: Model name for selected provider
- `EMBEDDING_PROVIDER`: Usually matches LLM_PROVIDER
- `EMBEDDING_MODEL`: nomic-embed-text (recommended for Ollama)
- `OLLAMA_HOST`: http://host.docker.internal:11434 (Docker) or http://localhost:11434 (local)
- `CHROMA_HOST`: chromadb (Docker service name)
- `CHROMA_PORT`: 8000 (internal ChromaDB port)
- `PORT`: 8080 (external gateway port)
- `DOCS_DIR`: Local folder mounted to `/host_root` in container

**LLM Provider Setup**

For OpenAI/Anthropic/Gemini, add API keys to `.env`:
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=openai
```

## Troubleshooting

**ChromaDB Connection Issues**
- Check service health: `docker compose logs chromadb`
- Verify network: `docker network ls`
- Test connection from backend: `docker compose exec backend curl chromadb:8000/api/v2/heartbeat`

**Ollama Connection Issues**
- Ensure Ollama is running: `curl http://localhost:11434/api/tags`
- Check `OLLAMA_HOST` in `.env` matches your setup
- For Docker on Linux: Use `http://host.docker.internal:11434`
- For Docker on Mac/Windows: Should work automatically

**Module Import Errors**
- Restart backend: `docker compose restart backend`
- If persists, rebuild: `docker compose up -d --build`

**Port Conflicts**
- Check port 8080 is free: `lsof -i :8080`
- Change `PORT` in `.env` to different value

## Important Implementation Notes

### ChromaDB Operations

Always use the singleton instance:
```python
from src.core.chroma_manager import get_chroma_manager

chroma_manager = get_chroma_manager()
client = chroma_manager.get_client()
```

For async operations with resilience:
```python
collection = await chroma_manager.get_collection_with_resilience(collection_name)
```

### Document Loading

Use `DoclingLoader` for all document types:
```python
from src.core.docling_loader import DoclingLoader

loader = DoclingLoader(file_path)
documents = loader.load()  # Returns List[Document]
```

Supported formats: PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, CSV

### LLM Configuration

Get config with hot-reload enabled (default):
```python
from src.core.config import get_config, create_llm_instances

config = get_config(use_hot_reload=True)
instances = create_llm_instances(config)
llm = instances["llm"]
embeddings = instances["embeddings"]
```

### Query Operations

Use `QueryEngine` for hybrid search:
```python
from src.core.query_engine import QueryEngine, QueryConfig

engine = QueryEngine(chroma_client, embeddings)
results = await engine.query(
    query_text="What is RAG?",
    collection_names=["my_docs"],
    config=QueryConfig(n_results=5, min_relevance=0.5)
)
```

## Testing the System

```bash
# 1. Health check
curl http://localhost:8080/health

# 2. Create collection
curl -X POST http://localhost:8080/api/v1/rag/collections \
  -F "collection_name=test_docs" \
  -F "embedding_provider=ollama" \
  -F "embedding_model=nomic-embed-text"

# 3. Upload document
curl -X POST http://localhost:8080/api/v1/rag/documents/upload \
  -F "files=@document.pdf" \
  -F "collection_name=test_docs" \
  -F "chunk_size=512" \
  -F "chunk_overlap=128"

# 4. Query knowledge base
curl -X POST http://localhost:8080/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "collection": "test_docs",
    "k": 5
  }'
```

## Code Style and Conventions

- **Logging**: Use `loguru` with component binding: `logger.bind(component="MyComponent")`
- **Error Handling**: Use structured exceptions from `src.core.exceptions`
- **Async Operations**: Use `async/await` for I/O operations, especially ChromaDB
- **Type Hints**: Required for all function signatures
- **Docstrings**: Use Google-style docstrings for all public functions/classes

## Dependencies

**Core Stack**:
- FastAPI 0.115.0 - API framework
- Docling 2.13.0 - Document processing
- ChromaDB 0.5.23 - Vector database
- LlamaIndex 0.12.9 - RAG pipeline
- PyTorch (CPU-only) - For ML models
- Loguru - Logging

**Installing Dependencies**:
The Dockerfile uses BuildKit caching for pip packages. When adding dependencies:
1. Add to `requirements.txt`
2. Rebuild: `docker compose up -d --build`

**PyTorch Note**: CPU-only version used to avoid 4GB+ CUDA downloads. See `requirements.txt` line 12.

## Edition Tiers

Community Edition (this repository):
- Unlimited collections and documents
- All file formats supported
- Full hybrid search with reranking
- All advanced RAG features enabled
- No API rate limits for self-hosting

Professional/Enterprise editions available via sales contact (see README).

## Source Code Hot-Reload

Backend source mounted as volume in `docker-compose.yml`:
```yaml
volumes:
  - ./backend/src:/app/src:ro
```

This means code changes are immediately detected (uvicorn `--reload` flag). No rebuild needed for code changes, only for dependency changes.

## Documentation

- API Docs: http://localhost:8080/docs (Swagger)
- ReDoc: http://localhost:8080/redoc
- OpenAPI JSON: http://localhost:8080/openapi.json
