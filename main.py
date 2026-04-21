import os
import logging
from fastapi import FastAPI
import uvicorn
"""
api.py — FastAPI wrapper around the RAG pipeline.

Endpoints
---------
POST /query          → full answer + sources (JSON)
GET  /query/stream   → streaming answer (SSE / plain text)
GET  /health         → liveness + cache stats
DELETE /cache        → flush the in-memory cache

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Install extras:
    pip install fastapi uvicorn[standard]
"""
# from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from cache import get_cache
from query import RAGResponse, rag_query, rag_query_stream
from retriever import build_vector_store
from observability import get_logger


log = get_logger("api")

# ── Application state ──────────────────────────────────────────────────────────

_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store
    log.info("Connecting to Pinecone...")
    _store = build_vector_store()
    log.info("Vector store ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Gov-Transparency RAG API",
    version="1.0.0",
    lifespan=lifespan,
)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_store():
    if _store is None:
        raise HTTPException(status_code=503, detail="Vector store not ready")
    return _store


# ── Request / response schemas ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    skip_cache: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    cached: bool
    query: str
    error: str | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest):
    """Full RAG pipeline — returns complete answer + source citations."""
    result: RAGResponse = rag_query(
        get_store(),
        body.question,
        skip_cache=body.skip_cache,
    )
    return result.to_dict()


@app.get("/query/stream")
async def query_stream_endpoint(
    q: Annotated[str, Query(min_length=3, max_length=1000)],
    skip_cache: bool = False,
):
    """
    Streaming endpoint — returns answer tokens as plain text/event-stream.
    Use EventSource in the browser or `requests.get(..., stream=True)` in Python.
    """
    return StreamingResponse(
        rag_query_stream(get_store(), q, skip_cache=skip_cache),
        media_type="text/x-ndjson",
    )


@app.get("/health")
async def health():
    cache = get_cache()
    return {
        "status": "ok",
        "store_ready": _store is not None,
        "cache": cache.stats(),
    }


@app.delete("/cache")
async def flush_cache():
    n = get_cache().clear()
    return {"flushed_entries": n}


@app.get("/history")
async def history(limit: int = 20, namespace: str = "epstein-docs"):
    """Return the most recent RAG queries from Supabase query_history."""
    from database import fetch_history
    return fetch_history(limit=limit, namespace=namespace)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gov-transparency-rag")

# app = FastAPI(title="gov-transparency-rag-backend")



if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting app on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")