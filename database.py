"""
database.py — Supabase persistence layer.

Tables
------
processed_pdfs   Track every ingested PDF so the ingestion pipeline
                 can skip already-processed files.

query_history    Append every RAG query + answer for audit, analytics,
                 and future fine-tuning datasets.

All functions are synchronous (Supabase Python SDK is sync).
Errors are caught and logged — DB failures never crash the RAG pipeline.

Schema (run once in Supabase SQL editor)
-----------------------------------------
create table if not exists processed_pdfs (
    id          bigserial primary key,
    filename    text unique not null,
    url         text,
    dataset     text,
    chunk_count int  default 0,
    char_count  int  default 0,
    created_at  timestamptz default now()
);

create table if not exists query_history (
    id          bigserial primary key,
    conversation_id text,
    question    text not null,
    answer      text,
    sources     jsonb,
    namespace   text default 'epstein-docs',
    cached      boolean default false,
    error       text,
    created_at  timestamptz default now()
);

create index if not exists idx_query_history_created
    on query_history (created_at desc);

create index if not exists idx_query_history_conversation
    on query_history (conversation_id, created_at desc);
"""
from __future__ import annotations

import json
from typing import Any

from supabase import create_client, Client

from config import cfg
from observability import get_logger
import os
from dotenv import load_dotenv
load_dotenv()

log = get_logger(__name__)


# ── Client (lazy singleton) ───────────────────────────────────────────────────

_supabase: Client | None = None


def _get_client() -> Client | None:
    global _supabase

    if _supabase is not None:
        return _supabase

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        log.warning("SUPABASE_URL or SUPABASE_KEY missing")
        return None

    try:
        _supabase = create_client(url, key)
        log.info("Supabase client initialised")
        return _supabase
    except Exception as e:
        log.error("Supabase init failed: %s", e)
        return None


# ── processed_pdfs ────────────────────────────────────────────────────────────

def is_already_processed(filename: str) -> bool:
    """Return True if this PDF filename already exists in processed_pdfs."""
    db = _get_client()
    if db is None:
        return False
    try:
        res = (
            db.table("processed_pdfs")
            .select("id")
            .eq("filename", filename)
            .limit(1)
            .execute()
        )
        return len(res.data) > 0
    except Exception as e:
        log.error("is_already_processed error: %s", e)
        return False


def save_pdf_record(
    filename: str,
    url: str,
    dataset: str,
    chunk_count: int,
    char_count: int,
) -> None:
    """Upsert a PDF processing record (idempotent on filename)."""
    db = _get_client()
    if db is None:
        return
    try:
        db.table("processed_pdfs").upsert(
            {
                "filename":    filename,
                "url":         url,
                "dataset":     dataset,
                "chunk_count": chunk_count,
                "char_count":  char_count,
            },
            on_conflict="filename",
        ).execute()
        log.debug("Saved PDF record: %s (%d chunks)", filename, chunk_count)
    except Exception as e:
        log.error("save_pdf_record error for '%s': %s", filename, e)


# ── query_history ─────────────────────────────────────────────────────────────

def save_query(
    question: str,
    answer: str,
    sources: list[dict] | str,
    namespace: str = "epstein-docs",
    cached: bool = False,
    error: str | None = None,
    conversation_id: str | None = None,
) -> None:
    """
    Persist a completed RAG query to query_history.

    Called by the pipeline after every successful (or failed) generation.
    Never raises — DB errors are logged and swallowed so the caller always
    gets its answer back.
    """
    db = _get_client()
    if db is None:
        return
    try:
        sources_json = (
            sources
            if isinstance(sources, str)
            else json.dumps(sources, default=str)
        )
        
        row = {
            "question":  question,
            "answer":    answer,
            "sources":   sources_json,
            "namespace": namespace,
            "cached":    cached,
            "error":     error,
        }
        if conversation_id:
            row["conversation_id"] = conversation_id

        try:
            db.table("query_history").insert(row).execute()
        except Exception as e:
            if not conversation_id:
                raise
            log.warning("query_history conversation_id write failed; retrying without it: %s", e)
            row.pop("conversation_id", None)
            db.table("query_history").insert(row).execute()
        log.debug("Saved query to history: '%.60s'", question)
    except Exception as e:
        log.error("save_query error: %s", e)


def fetch_history(
    limit: int = 10,
    namespace: str | None = None,
    conversation_id: str | None = None,
) -> list[dict]:
    """
    Return the most recent queries from query_history, newest first.

    Parameters
    ----------
    limit       Max rows to return.
    namespace        If set, filter to a specific Pinecone namespace.
    conversation_id  If set, filter to a specific chat session.
    """
    db = _get_client()
    if db is None:
        return []
    try:
        q = db.table("query_history").select("*").order("id", desc=True).limit(limit)
        if namespace:
            q = q.eq("namespace", namespace)
        if conversation_id:
            q = q.eq("conversation_id", conversation_id)
        res = q.execute()
        return [
            {
                "id":         r.get("id"),
                "conversation_id": r.get("conversation_id"),
                "question":   r.get("question"),
                "answer":     r.get("answer"),
                "sources":    _safe_json_loads(r.get("sources") or "[]"),
                "namespace":  r.get("namespace"),
                "cached":     r.get("cached", False),
                "error":      r.get("error"),
                "created_at": r.get("created_at"),
            }
            for r in res.data
            if isinstance(r, dict) 
        ]
        
    except Exception as e:
        log.error("fetch_history error: %s", e)
        return []


# ── stats ─────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """Aggregate counts for the dashboard / health endpoint."""
    db = _get_client()
    if db is None:
        return {"error": "database unavailable"}
    try:
        pdfs    = db.table("processed_pdfs").select("id", count="exact").execute()
        chunks  = db.table("processed_pdfs").select("chunk_count").execute()
        queries = db.table("query_history").select("id", count="exact").execute()
        cached  = (
            db.table("query_history")
            .select("id", count="exact")
            .eq("cached", True)
            .execute()
        )
        return {
            "pdfs_processed":  pdfs.count or 0,
            "total_chunks":    sum(r["chunk_count"] or 0 for r in chunks.data),
            "total_queries":   queries.count or 0,
            "cached_queries":  cached.count or 0,
        }
    except Exception as e:
        log.error("get_stats error: %s", e)
        return {"error": str(e)}


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value          # Supabase jsonb columns come back already parsed
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
