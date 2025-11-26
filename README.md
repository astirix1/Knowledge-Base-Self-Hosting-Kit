# Knowledge‑Base Self‑Hosting Kit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


---

## 🎯 What is this?

**Knowledge‑Base Self‑Hosting Kit** is a **complete, production‑ready starter template** that shows how to glue together the **Smart‑Ingest‑Kit** and **Smart‑Router‑Kit** into a fully‑functional, self‑hosted knowledge‑base.

- 📄 **Docling**‑powered ingestion (PDF, DOCX, HTML, images, …) with automatic chunking & metadata extraction.
- 🧭 **Hybrid retrieval** (vector + keyword) + **parent‑document reranker**.
- ⚡️ Docker‑Compose setup that runs **ChromaDB**, **FastAPI**, and an optional **React** UI out of the box.
- 🛠️ Ready for **local LLMs** via Ollama or any OpenAI‑compatible endpoint.

The goal is to give developers a **single repository** that they can clone, run, and extend – no piecing together of disparate tutorials required.

---

## 🚀 Quick Start (5 minutes)

```bash
# 1️⃣ Clone the repo
git clone https://github.com/2dogsandanerd/Knowledge-Base-Self-Hosting-Kit.git
cd Knowledge-Base-Self-Hosting-Kit

# 2️⃣ (Optional) Set your LLM endpoint – see the .env.example file
cp .env.example .env
# Edit .env if you want to use OpenAI, Ollama, etc.

# 3️⃣ Build & run everything with Docker Compose
docker compose up -d --build

# 4️⃣ Open the UI
open http://localhost:3000   # Web UI (or http://localhost:8080/docs for the API)
```

That’s it – the UI lets you **upload documents**, **run queries**, and **inspect the vector store**.

---

## 📦 What's Inside?

```
.
├─ backend/               # FastAPI server
│   ├─ src/
│   │   ├─ api/v1/rag/   # RAG endpoints (ingestion, query, collections, documents)
│   │   ├─ core/          # Docling loader, ChromaDB manager, retrievers, postprocessors
│   │   ├─ services/      # Document loaders, ingestion pipeline, generators
│   │   ├─ config/        # Configuration management
│   │   └─ utils/         # Utility functions
│   ├─ requirements.txt   # Python dependencies
│   └─ Dockerfile
├─ frontend/              # Simple web UI for document ingestion
│   └─ index.html
├─ docker-compose.yml     # Orchestrates backend, worker, frontend, chromadb, redis
├─ .env.example           # Example configuration
└─ README.md               # You are reading it now!
```

---

## 🛠️ Architecture Overview

1. **Ingestion Service** – reads files, uses **Docling** to extract text, creates chunks, and stores embeddings in **ChromaDB**.
2. **Retrieval Pipeline** – hybrid retrieval (vector + BM25) + **parent‑document reranker** for relevance.
3. **API Layer** – FastAPI exposing `/api/v1/rag/*` endpoints for ingestion, querying, and collection management.
4. **Task Queue** – Celery workers for async ingestion with Redis as message broker.
5. **Frontend** – Simple web UI for folder ingestion and progress tracking.
6. **LLM Provider** – configurable via `.env` (Ollama, OpenAI, Anthropic, etc.).

---

## ⚙️ Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `openai`, `ollama`, `anthropic` … | `ollama` |
| `LLM_MODEL` | Model name (e.g. `llama3.2:latest`) | `llama3.2:latest` |
| `CHROMA_HOST` | Host for ChromaDB | `chromadb` |
| `CHROMA_PORT` | Port for ChromaDB | `8000` |
| `INGEST_BATCH_SIZE` | Number of docs per batch | `10` |
| `EMBEDDING_MODEL` | Embedding model for Docling | `nomic-embed-text` |

Edit `.env` (or set environment variables) before starting the stack.

---

## 📚 Documentation & Demo

- **Full docs** live in the `docs/` folder (Markdown + diagrams).
- **Demo video** – see `docs/demo.mp4` (short 2‑minute walkthrough).
- **API reference** – automatically generated Swagger UI at `http://localhost:8000/docs`.

---

## 🤝 Contributing

We welcome contributions! Please read the **CONTRIBUTING.md** for:
- How to open a good issue.
- Coding style (black, isort, mypy).
- Running the test suite (`pytest -q`).
- Submitting a PR – we use **GitHub Actions** to verify CI.

---

## 📜 License

MIT © 2025 2dogsandanerd. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- **Docling** – for brilliant document parsing.
- **LlamaIndex** – for the retrieval pipeline.
- **ChromaDB** – for fast, persistent vector storage.
- The **r/docling** community for early feedback.

---

*If you liked this project, star it ★ and share the link !*

---

## 🏢 Editions

This repository contains the **Community Edition** - a fully functional RAG system for evaluation and learning.

### Community Edition (This Repository)
- ✅ Full RAG pipeline with ChromaDB
- ✅ Docling document processing
- ✅ Hybrid retrieval (vector + keyword)
- ✅ Basic ingestion pipeline
- ✅ Up to 3 collections, 1000 docs per collection
- ✅ Supports: PDF, TXT, Markdown
- ⚠️  Basic heuristic-based features

### Professional Edition
**Advanced features for production deployments:**
- 🚀 10 collections, 5000 docs per collection
- 🚀 Advanced reranking with cross-encoders
- 🚀 Multi-collection intelligent search
- 🚀 Extended format support (DOCX, HTML, PPTX, XLSX)
- 🚀 ML-powered document classification
- 🚀 Intelligent pattern generation
- 🚀 Team collaboration features

### Enterprise Edition
**Full-scale deployment with custom support:**
- 💼 Unlimited collections and documents
- 💼 Custom fine-tuned models
- 💼 SSO and RBAC integration
- 💼 Advanced analytics and monitoring
- 💼 Dedicated support and SLA
- 💼 Custom feature development
- 💼 On-premise deployment assistance

**Interested in Professional or Enterprise?** Contact: [your-contact-email]

---

## 📝 Note on Implementation

This Community Edition demonstrates our RAG architecture and provides functional basic features. Some components include references to advanced features available in paid editions:

- **Generators**: Basic implementations with notes on enterprise ML-powered versions
- **Classification**: Heuristic-based (Enterprise: ML models with confidence calibration)
- **Feature Limits**: Basic tier system (Enterprise: Dynamic licensing with usage tracking)

This approach allows you to:
- ✅ Evaluate the architecture and code quality
- ✅ Deploy a working RAG system immediately
- ✅ Understand what's possible with upgraded editions
- ✅ Make informed decisions about enterprise features

