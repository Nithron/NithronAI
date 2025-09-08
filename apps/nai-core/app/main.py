import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")
INDEX_DIR = os.path.join(DATA_DIR, "index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.jsonl")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

app = FastAPI(title="nAI Core", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[\t\x0b\x0c]", " ", t)
    return t.strip()

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100):
    text = normalize_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]

def save_doc(file: UploadFile) -> str:
    # Persist original upload to DOCS_DIR with timestamp
    name = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{file.filename}"
    path = os.path.join(DOCS_DIR, name)
    with open(path, "wb") as f:
        f.write(file.file.read())
    return path

def extract_text(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(path)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    elif lower.endswith(".md") or lower.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        return ""

def append_to_index(doc_path: str, chunks: List[str]):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INDEX_PATH, "a", encoding="utf-8") as idx:
        for i, ch in enumerate(chunks):
            rec = {
                "doc_path": os.path.relpath(doc_path, DOCS_DIR),
                "chunk_id": i,
                "text": ch
            }
            idx.write(json.dumps(rec, ensure_ascii=False) + "\n")

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    added = []
    for f in files:
        saved = save_doc(f)
        text = extract_text(saved)
        if not text.strip():
            continue
        chunks = chunk_text(text, max_chars=1200, overlap=150)
        append_to_index(saved, chunks)
        added.append({"file": os.path.basename(saved), "chunks": len(chunks)})
    return {"added": added, "index_path": os.path.relpath(INDEX_PATH, os.path.dirname(__file__))}

def load_index() -> List[Dict[str, Any]]:
    if not os.path.exists(INDEX_PATH):
        return []
    rows = []
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def bm25_search(query: str, rows: List[Dict[str, Any]], top_k: int = 5):
    if not rows:
        return []
    corpus = [r["text"] for r in rows]
    # Simple tokenization
    def tok(s: str):
        return re.findall(r"[\w\-]+", s.lower())
    tokenized_corpus = [tok(c) for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tok(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        r = rows[idx].copy()
        r["score"] = float(score)
        results.append(r)
    return results

def compose_answer(query: str, passages: List[Dict[str, Any]]) -> str:
    if not passages:
        return "I couldn't find anything relevant in your current corpus."
    # Simple extractive "answer": join the best few sentences
    joined = "\n\n".join(p["text"][:500] for p in passages[:3])
    answer = f"""
    ### Draft Answer (extractive)
    Based on your indexed documents, here are the most relevant excerpts:

    {joined}

    (Tip: connect an LLM later for an abstractive answer with citations.)
    """
    return answer.strip()

@app.post("/ask")
async def ask(req: AskRequest):
    rows = load_index()
    hits = bm25_search(req.question, rows, top_k=req.top_k)
    citations = [{
        "doc": h["doc_path"],
        "chunk_id": h["chunk_id"],
        "score": h["score"]
    } for h in hits]
    answer = compose_answer(req.question, hits)
    return {"answer": answer, "citations": citations, "count": len(hits)}
