# nAI Core (FastAPI Backend)

The core backend service for nAI - a local-first document Q&A system.

## Features

- **Document Ingestion** (`/ingest`) — Upload PDF, TXT, MD, HTML files
- **Question Answering** (`/ask`) — Get answers with citations
- **Raw Search** (`/search`) — BM25 and semantic search
- **Document Management** (`/documents`) — List, view, delete
- **Multi-turn Chat** (`/chat`) — Conversation with context
- **Authentication** (`/auth`) — JWT-based user management

## Architecture

```
app/
├── config.py          # Environment configuration
├── main.py            # FastAPI app factory
├── middleware.py      # Request logging, rate limiting
├── routes/            # API endpoints
│   ├── health.py
│   ├── ingest.py
│   ├── ask.py
│   ├── documents.py
│   ├── chat.py
│   └── auth.py
├── services/          # Business logic
│   ├── indexer.py     # Document indexing
│   ├── retriever.py   # BM25 search with caching
│   ├── answerer.py    # Answer generation
│   ├── embeddings.py  # Semantic search (Qdrant)
│   └── auth.py        # Authentication
├── models/            # Pydantic schemas
│   └── schemas.py
└── utils/             # Utilities
    ├── text.py        # Chunking, tokenization
    ├── extractors.py  # Text extraction
    └── logging.py     # Structured logging
```

## Quick Start

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Docker

```bash
docker build -t nai-core .
docker run -p 8000:8000 -v $(pwd)/data:/app/data nai-core
```

### With Full Stack

```bash
cd ../../infra
docker-compose up -d
```

## Configuration

All configuration via environment variables with `NAI_` prefix:

```bash
# Core
NAI_DEBUG=true
NAI_LOG_LEVEL=DEBUG

# LLM
NAI_LLM_ENABLED=true
NAI_LLM_PROVIDER=ollama
NAI_LLM_MODEL=llama3.2
NAI_LLM_BASE_URL=http://localhost:11434

# Embeddings
NAI_EMBEDDINGS_ENABLED=true
NAI_EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Qdrant
NAI_QDRANT_ENABLED=true
NAI_QDRANT_HOST=localhost
NAI_QDRANT_PORT=6333

# Authentication
NAI_AUTH_ENABLED=true
NAI_AUTH_SECRET_KEY=change-me-in-production
```

## API Usage

### Ingest Documents

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@document.pdf"
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "top_k": 5}'
```

### Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "method": "hybrid"}'
```

### List Documents

```bash
curl http://localhost:8000/documents
```

### Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

## Testing

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json
