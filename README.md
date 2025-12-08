# Knowledge Base Self-Hosting Kit (Community Edition)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)

**Production-ready RAG system combining Docling document processing with ChromaDB vector storage.**

Extracted from  our AI email assistant. This Community Edition focuses purely on RAG functionality without email-specific features.

---

## 🎯 What You Get

- **🔥 Modern Document Processing**: Docling 2.13.0 (PDF, DOCX, PPTX, XLSX, HTML, Markdown)
- **🔍 Hybrid Search**: Vector similarity + BM25 keyword search with Reciprocal Rank Fusion
- **📦 ChromaDB 0.5.23**: Vector storage with connection pooling and health checks
- **🚀 LlamaIndex 0.12.9**: Advanced retrieval pipelines
- **🎛️ Multi-LLM Support**: Ollama (default), OpenAI, Anthropic, Gemini
- 🖥️ **Lightweight UI**: Zero-build, single-file HTML/JS dashboard
- **🐳 Docker-First**: Production-ready deployment with hot-reload support

---

## ⚡ Quick Start (5 minutes)

### Prerequisites

1. **Docker & Docker Compose** installed
2. **Ollama** running locally (for embeddings)
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.com/install.sh | sh

   # Start Ollama server
   ollama serve

   # Pull embedding model (in another terminal)
   ollama pull nomic-embed-text
   ```

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/self-hosting-kit.git
cd self-hosting-kit

# 2. Configure & Start
# Run the interactive setup script to set your document folder
./setup.sh

# (Alternative) Manual setup:
# cp .env.example .env
# docker compose up -d

# 4. Check health
curl http://localhost:8080/health
# Expected: {"status":"healthy","chromadb":"connected","collections_count":0}

# 5. Open the application
open http://localhost:8080
```

**Services (all through single nginx gateway):**
- Frontend UI: http://localhost:8080/
- API Docs: http://localhost:8080/docs
- Health Check: http://localhost:8080/health
- API Endpoints: http://localhost:8080/api/v1/rag/*

**Port Configuration:**
The application exposes a single port (default: 8080) configured via the `PORT` variable in `.env`. This prevents port conflicts and follows production best practices with nginx as a reverse proxy.

---

## 📖 Usage Examples

### Create a Collection

```bash
curl -X POST http://localhost:8080/api/v1/rag/collections \
  -F "collection_name=my_docs" \
  -F "embedding_provider=ollama" \
  -F "embedding_model=nomic-embed-text"
```

### Upload Documents

```bash
curl -X POST http://localhost:8080/api/v1/rag/documents/upload \
  -F "files=@document.pdf" \
  -F "collection_name=my_docs" \
  -F "chunk_size=512" \
  -F "chunk_overlap=128"
```

### Query Your Knowledge Base

```bash
curl -X POST http://localhost:8080/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this document about?",
    "collection": "my_docs",
    "k": 5
  }'
```

---

## 🚀 Using the API Without Frontend

The Web UI is great for quick testing, but you'll likely want to integrate this into your applications. Here's how to use the API directly:

### Python Example

```python
import requests

BASE_URL = "http://localhost:8080/api/v1/rag"

# 1. Create a collection
response = requests.post(
    f"{BASE_URL}/collections",
    files={
        "collection_name": (None, "my_knowledge"),
        "embedding_provider": (None, "ollama"),
        "embedding_model": (None, "nomic-embed-text")
    }
)
print(f"Collection created: {response.json()}")

# 2. Upload documents
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/documents/upload",
        files={"files": f},
        data={
            "collection_name": "my_knowledge",
            "chunk_size": 512,
            "chunk_overlap": 128
        }
    )
print(f"Upload status: {response.json()}")

# 3. Query the knowledge base
response = requests.post(
    f"{BASE_URL}/query",
    json={
        "query": "What are the main topics?",
        "collection": "my_knowledge",
        "k": 5
    }
)
result = response.json()
print(f"Answer: {result.get('answer')}")
print(f"Sources: {len(result.get('sources', []))}")
```

### Folder Ingestion Example

```python
import requests
import time

BASE_URL = "http://localhost:8080/api/v1/rag"

# Start folder ingestion
response = requests.post(
    f"{BASE_URL}/ingest-folder",
    json={
        "folder_path": "/host_root/path/to/your/docs",
        "collection_name": "my_docs",
        "profile": "documents",
        "recursive": True
    }
)

task_id = response.json()["task_id"]
print(f"Ingestion started: {task_id}")

# Poll for status
while True:
    status = requests.get(f"{BASE_URL}/ingest-status/{task_id}").json()

    if status["status"] == "completed":
        print(f"✅ Processed {status['processed_files']} files")
        break
    elif status["status"] == "failed":
        print(f"❌ Failed: {status['error']}")
        break
    else:
        print(f"⏳ Processing: {status.get('current_file')} ({status.get('processed')}/{status.get('total')})")
        time.sleep(2)
```

### cURL Examples

```bash
# Health check
curl http://localhost:8080/health

# List collections
curl http://localhost:8080/api/v1/rag/collections

# Get collection stats
curl http://localhost:8080/api/v1/rag/collections/my_docs/stats

# Query with specific parameters
curl -X POST http://localhost:8080/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the architecture",
    "collection": "my_docs",
    "k": 10,
    "similarity_threshold": 0.5
  }'

# Delete a collection
curl -X DELETE http://localhost:8080/api/v1/rag/collections/my_docs
```

### JavaScript/TypeScript Example

```javascript
const BASE_URL = "http://localhost:8080/api/v1/rag";

async function queryKnowledgeBase(question, collection = "my_docs") {
  const response = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: question,
      collection: collection,
      k: 5
    })
  });

  const result = await response.json();
  return {
    answer: result.answer,
    sources: result.sources
  };
}

// Usage
const result = await queryKnowledgeBase("What is RAG?");
console.log(result.answer);
```

### Full API Documentation

For complete API documentation including all endpoints, parameters, and response schemas:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI JSON**: http://localhost:8080/openapi.json

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│             FastAPI Backend (Port 8081)         │
│  ┌──────────────┐  ┌─────────────────────────┐ │
│  │ RAG API      │  │ Lifespan Management     │ │
│  │ - Query      │  │ - ChromaDB Connection   │ │
│  │ - Upload     │  │ - Singleton Patterns    │ │
│  │ - Collections│  │ - Circuit Breaker       │ │
│  └──────┬───────┘  └────────────┬────────────┘ │
└─────────┼──────────────────────┼───────────────┘
          │                      │
          ↓                      ↓
┌─────────────────┐    ┌──────────────────────┐
│  ChromaDB       │    │  Ollama / LLM        │
│  Vector Storage │    │  Embeddings & Chat   │
│  (Port 8001)    │    │  (Port 11434)        │
└─────────────────┘    └──────────────────────┘
```

### Key Components

**Backend (`backend/src/`):**
- `api/v1/rag/` - API endpoints (ingestion, query, collections, documents)
- `core/` - ChromaDB manager, Docling loader, retrievers, query engine
- `services/` - Document processing, classification, generators

**Core Patterns:**
- **Singleton**: ChromaManager for single connection instance
- **Resilience**: Circuit breaker + retry logic for ChromaDB
- **Lifespan**: Proper FastAPI startup/shutdown for clean connections
- **Hot-Reload**: Source code mounted as volume for development

---

## 🔧 Configuration

Environment variables set in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM provider (ollama, openai, anthropic, gemini) |
| `LLM_MODEL` | `llama3.2:latest` | Model name for selected provider |
| `EMBEDDING_PROVIDER` | `ollama` | Embedding provider (usually matches LLM) |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama connection URL |
| `CHROMA_HOST` | `chromadb` | ChromaDB service name (Docker) |
| `CHROMA_PORT` | `8000` | ChromaDB internal port |
| `DEBUG` | `false` | Enable debug logging |
| `LOG_LEVEL` | `INFO` | Logging level |

**For OpenAI/Anthropic/Gemini:**
Add API keys to `docker-compose.yml`:
```yaml
environment:
  - LLM_PROVIDER=openai
  - OPENAI_API_KEY=sk-...
  - EMBEDDING_PROVIDER=openai
```

---

## 📦 What's Inside

```
.
├── backend/
│   ├── src/
│   │   ├── api/v1/rag/         # RAG endpoints
│   │   │   ├── collections.py  # Collection CRUD
│   │   │   ├── documents/      # Upload, management
│   │   │   ├── query.py        # RAG queries
│   │   │   ├── ingestion/      # Folder scanning, batch processing
│   │   │   └── cockpit.py      # System status
│   │   ├── core/
│   │   │   ├── chroma_manager.py      # ChromaDB singleton
│   │   │   ├── docling_loader.py      # Document parser
│   │   │   ├── query_engine.py        # Query execution
│   │   │   ├── retrievers/            # Hybrid, BM25, reranker
│   │   │   ├── config.py              # Multi-LLM config
│   │   │   └── feature_limits.py      # Edition tiers
│   │   └── services/
│   │       ├── docling_service.py     # Central doc processing
│   │       ├── classification.py      # Doc classification
│   │       └── generators/            # Summaries, configs
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html              # Zero-build dashboard (Vanilla JS)
├── docker-compose.yml          # Full stack orchestration
├── CLAUDE.md                   # Development guide
└── README.md
```

---

## 🎓 Development

### Local Development (without Docker)

```bash
cd backend
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload
```

**Note:** You'll need ChromaDB and Ollama running separately.

### Docker Development (with hot-reload)

Code changes are automatically detected (source mounted as volume):

```bash
# Edit code in backend/src/
# Changes reflect immediately, no rebuild needed

# View logs
docker compose logs -f backend

# Restart if needed
docker compose restart backend
```

### Rebuild (only when changing dependencies)

```bash
docker compose down
docker compose up -d --build
```

---

## 🚨 Troubleshooting

### App won't start

```bash
# Check all services
docker compose ps

# View backend logs
docker compose logs backend

# Check ChromaDB connection
docker compose logs chromadb
```

### "Failed to connect to Ollama"

```bash
# Ensure Ollama is running
ollama serve

# Pull embedding model
ollama pull nomic-embed-text

# Test Ollama
curl http://localhost:11434/api/tags
```

### "ChromaDB client not available"

```bash
# Check ChromaDB service
docker compose logs chromadb

# Restart ChromaDB
docker compose restart chromadb
```

### Import errors after code changes

```bash
# Restart backend to reload modules
docker compose restart backend
```

---

## 📚 API Endpoints

Full API documentation available at http://localhost:8081/docs

**Collections:**
- `POST /api/v1/rag/collections` - Create collection
- `GET /api/v1/rag/collections` - List collections
- `DELETE /api/v1/rag/collections/{name}` - Delete collection

**Documents:**
- `POST /api/v1/rag/documents/upload` - Upload documents
- `GET /api/v1/rag/documents` - List documents
- `DELETE /api/v1/rag/documents/{id}` - Delete document

**Query:**
- `POST /api/v1/rag/query` - Query knowledge base

**Ingestion:**
- `POST /api/v1/rag/ingestion/scan-folder` - Scan folder for documents
- `POST /api/v1/rag/ingestion/ingest-batch` - Batch ingestion
- `POST /api/v1/rag/ingestion/ingest-folder` - Ingest folder synchronously

---

## 🏢 Edition Comparison

### Community Edition (This Repository)

**Free & Open Source (Self-Hosted)**

- ✅ **Collections: Unlimited**
- ✅ **Documents: Unlimited**
- ✅ Formats: PDF, Markdown, TXT
- ✅ Hybrid Search: Vector + BM25
- ✅ Basic Classification: Heuristic-based
- ✅ Full source code access
- ❌ No advanced reranking (can be added via code)
- ❌ No multi-collection search routing
- ❌ No ML-powered features

**Perfect for:**
- Personal Knowledge Bases
- Internal Company Documentation
- Development and testing
- Understanding RAG architecture

### Professional Edition

**Contact Sales**

- 🚀 Collections: 10, 5000 docs each
- 🚀 Formats: Extended (DOCX, HTML, PPTX, XLSX)
- 🚀 Advanced Reranking: Cross-encoder models
- 🚀 Multi-Collection Search: Intelligent routing
- 🚀 ML Classification: Confidence calibration
- 🚀 Analytics & Monitoring
- 🚀 Priority Support

### Enterprise Edition

**Contact Sales**

- 💼 Unlimited collections & documents
- 💼 Custom fine-tuned models
- 💼 SSO & RBAC integration
- 💼 Advanced analytics dashboard
- 💼 Dedicated support & SLA
- 💼 Custom feature development
- 💼 On-premise deployment assistance

---

## 🤝 Contributing

Contributions welcome! This is the Community Edition - we encourage:

- 🐛 Bug reports and fixes
- 📝 Documentation improvements
- 💡 Feature suggestions
- ⚡ Performance optimizations

**Please note:** Advanced features (ML classification, reranking, multi-collection) are part of paid editions. Community contributions focus on core RAG functionality.

---

## 📜 License

MIT License - Use freely in commercial and open-source projects.
Validated Table Extractor
Copyright (c) 2025 2dogsandanerd

This product includes software developed by IBM (Docling) and other open source contributors.

Docling: https://github.com/DS4SD/docling (MIT License)
Copyright (c) 2024 IBM Corp.

---

## Citation

If you use this tool in research or production, please cite:

```bibtex
@software{validated_table_extractor,
  title = {Validated Table Extractor: Audit-Ready PDF Table Extraction},
  author = {2dogsandanerd},
  year = {2025},
  url = {https://github.com/2dogsandanerd/validated-table-extractor}
}
```


---

## 🙏 Acknowledgements

- **Docling** - Modern document processing
- **ChromaDB** - Vector storage
- **LlamaIndex** - Retrieval pipelines
- **FastAPI** - API framework
- **Ollama** - Local LLM inference

---

## 📞 Support

- **Community Edition**: GitHub Issues
- **Professional/Enterprise**: [Contact Sales](mailto:your-email@example.com)
- **Documentation**: See `CLAUDE.md` for development guide

---

## 🎯 Roadmap

**Community Edition:**
- [ ] Simple authentication layer
- [ ] Query history tracking
- [ ] Export/import collections
- [ ] Improved error messages

**Professional Features** (Available Now):
- Multi-collection intelligent search
- Advanced reranking with cross-encoders
- ML-powered classification
- Extended format support

---

**Built with ❤️ by developers who needed a solid RAG foundation.**

*If you find this useful, star ⭐ the repo and share with others!*
