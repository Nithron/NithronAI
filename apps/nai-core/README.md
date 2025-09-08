# nAI Core (FastAPI)
Features:
- `/health` — liveness
- `/ingest` — upload PDF/TXT/MD; builds local JSONL index
- `/ask` — queries index with BM25 and returns passages + extractive answer

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
