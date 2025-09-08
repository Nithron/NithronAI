# nAI (Nithron AI) — Local-first, open-core AI stack

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](./LICENSE)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-informational.svg)](apps/nai-core)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](#)
[![Release](https://img.shields.io/badge/nAI-v0.1.0--pre--alpha-blue)](https://github.com/Nithronverse/nAI/releases/tag/v0.1.0-pre-alpha)

**nAI** is a local-first AI starter for NithronOS & Niro:
- **Ingest** PDFs/Markdown/TXT → **index** locally (JSONL).
- **Ask** questions → get **extractive answers with citations** (BM25).
- **No external LLM required** (plug one in later if you like).
- Ships with a **FastAPI** backend and a **tiny static web UI**.

> Privacy by default. Open-core by design. Runs great on a homelab.

## Quickstart

### 1) Run the backend
```bash
cd apps/nai-core
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2) Run the tiny web UI
```bash
cd apps/nai-docs/web
# Serve the static files (choose one):
python -m http.server 5173
# or
npx http-server -p 5173
```

Open: http://localhost:5173 and point it to the API (defaults to http://localhost:8000).

## What this PoC does
- **/ingest**: upload PDFs/TXT/MD; text is chunked and indexed (BM25, local JSON index).
- **/ask**: searches your index and returns an **extractive answer** + **citations** (top passages).
- No external LLM required. You can later wire an LLM by setting env and extending `compose_answer()`.

## Roadmap pointers
- Add optional embeddings + Qdrant (see `infra/docker-compose.yml`).
- Add authentication and per‑workspace indices.
- Replace extractive answer with LLM generation (with structured citations).

## License
Core PoC is **AGPL-3.0-only**. Commercial add-ons TBD in `COMMERCIAL.md`.
