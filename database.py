"""
database.py — Supabase persistence layer.

Tables
------
processed_pdfs   Track every ingested PDF so the ingestion pipeline
                 can skip already-processed files.

query_history    Append every RAG query + answer for audit, analytics,
                 and future fine-tuning datasets.
query_answers    Store one shared answer per normalized application query.

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

create table if not exists query_answers (
    id          bigserial primary key,
    query_hash  text not null,
    question    text not null,
    normalized_question text not null,
    answer      text,
    sources     jsonb,
    namespace   text default 'epstein-docs',
    cached      boolean default false,
    error       text,
    ask_count   int not null default 1,
    created_at  timestamptz default now(),
    updated_at  timestamptz default now(),
    last_asked_at timestamptz default now(),
    unique (namespace, query_hash)
);

create table if not exists query_history (
    id          bigserial primary key,
    user_id     uuid references auth.users(id) on delete cascade,
    query_answer_id bigint references query_answers(id) on delete cascade,
    query_hash  text,
    conversation_id text,
    question    text not null,
    answer      text,
    sources     jsonb,
    namespace   text default 'epstein-docs',
    cached      boolean default false,
    error       text,
    ask_count   int not null default 1,
    last_asked_at timestamptz default now(),
    created_at  timestamptz default now()
);

create index if not exists idx_query_history_created
    on query_history (created_at desc);

create index if not exists idx_query_history_conversation
    on query_history (user_id, conversation_id, created_at desc);

create unique index if not exists idx_query_history_user_conversation_hash
    on query_history (user_id, coalesce(conversation_id, ''), query_hash)
    where user_id is not null and query_hash is not null;
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from supabase import create_client, Client

from config import cfg
from observability import get_logger
import os
from dotenv import load_dotenv
load_dotenv()

log = get_logger(__name__)

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


# ── Client (lazy singleton) ───────────────────────────────────────────────────

_supabase: Client | None = None


def _get_client() -> Client | None:
    global _supabase

    if _supabase is not None:
        return _supabase

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

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
    user_id: str | None = None,
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
        sources_json = _coerce_sources_json(sources)
        normalized_question = normalize_query_text(question)
        query_hash = build_query_hash(question, namespace=namespace)
        answer_row = _upsert_query_answer(
            db,
            query_hash=query_hash,
            question=question,
            normalized_question=normalized_question,
            answer=answer,
            sources=sources_json,
            namespace=namespace,
            cached=cached,
            error=error,
        )

        row: dict[str, Any] = {
            "question":  question,
            "namespace": namespace,
            "query_hash": query_hash,
        }
        if answer_row and answer_row.get("id"):
            row["query_answer_id"] = answer_row["id"]
        if user_id:
            row["user_id"] = user_id
        if conversation_id:
            row["conversation_id"] = conversation_id

        try:
            if user_id:
                history_query = (
                    db.table("query_history")
                    .select("id,ask_count")
                    .eq("user_id", user_id)
                    .eq("query_hash", query_hash)
                )
                if conversation_id:
                    history_query = history_query.eq("conversation_id", conversation_id)
                else:
                    history_query = history_query.is_("conversation_id", "null")
                existing_history = history_query.limit(1).execute()
                if existing_history.data:
                    history_id = existing_history.data[0]["id"]
                    row["ask_count"] = int(existing_history.data[0].get("ask_count") or 0) + 1
                    row["last_asked_at"] = _now_iso()
                    db.table("query_history").update(row).eq("id", history_id).execute()
                else:
                    db.table("query_history").insert(row).execute()
            else:
                db.table("query_history").insert(row).execute()
        except Exception as e:
            if not conversation_id and not user_id:
                raise
            log.warning("query_history scoped write failed; retrying without scoped columns: %s", e)
            row.pop("conversation_id", None)
            row.pop("user_id", None)
            db.table("query_history").insert(row).execute()
        log.debug("Saved query to history: '%.60s'", question)
    except Exception as e:
        log.error("save_query error: %s", e)


def fetch_history(
    limit: int = 10,
    namespace: str | None = None,
    conversation_id: str | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """
    Return the most recent queries from query_history, newest first.

    Parameters
    ----------
    limit       Max rows to return.
    namespace        If set, filter to a specific Pinecone namespace.
    conversation_id  If set, filter to a specific chat session.
    user_id          If set, filter to a specific authenticated user.
    """
    db = _get_client()
    if db is None:
        return []
    try:
        q = (
            db.table("query_history")
            .select("*, query_answers(*)")
            .order("last_asked_at", desc=True)
            .limit(limit)
        )
        if namespace:
            q = q.eq("namespace", namespace)
        if user_id:
            q = q.eq("user_id", user_id)
        if conversation_id:
            q = q.eq("conversation_id", conversation_id)
        res = q.execute()
        return [_history_response_row(r) for r in res.data if isinstance(r, dict)]
        
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


def _coerce_sources_json(sources: list[dict] | str) -> Any:
    if isinstance(sources, str):
        return _safe_json_loads(sources)
    return sources


def normalize_query_text(question: str) -> str:
    """Normalize user question text for application-wide exact deduplication."""
    text = _PUNCTUATION_RE.sub(" ", question.casefold())
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or question.casefold().strip()


def build_query_hash(question: str, *, namespace: str = "epstein-docs") -> str:
    raw = f"{namespace}:{normalize_query_text(question)}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _upsert_query_answer(
    db: Client,
    *,
    query_hash: str,
    question: str,
    normalized_question: str,
    answer: str,
    sources: Any,
    namespace: str,
    cached: bool,
    error: str | None,
) -> dict[str, Any] | None:
    existing = (
        db.table("query_answers")
        .select("*")
        .eq("namespace", namespace)
        .eq("query_hash", query_hash)
        .limit(1)
        .execute()
    )
    row = {
        "query_hash": query_hash,
        "question": question,
        "normalized_question": normalized_question,
        "answer": answer,
        "sources": sources,
        "namespace": namespace,
        "cached": cached,
        "error": error,
        "last_asked_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    if existing.data:
        current = existing.data[0]
        row["ask_count"] = int(current.get("ask_count") or 0) + 1
        updated = db.table("query_answers").update(row).eq("id", current["id"]).execute()
        if updated.data:
            return updated.data[0]
        return {**current, **row}

    row["ask_count"] = 1
    inserted = db.table("query_answers").insert(row).execute()
    if inserted.data:
        return inserted.data[0]
    return None


def _history_response_row(row: dict[str, Any]) -> dict[str, Any]:
    answer_row = row.get("query_answers")
    if not isinstance(answer_row, dict):
        answer_row = {}

    sources = answer_row.get("sources", row.get("sources"))
    return {
        "id": row.get("id"),
        "user_id": row.get("user_id"),
        "query_answer_id": row.get("query_answer_id"),
        "query_hash": row.get("query_hash") or answer_row.get("query_hash"),
        "conversation_id": row.get("conversation_id"),
        "question": answer_row.get("question") or row.get("question"),
        "answer": answer_row.get("answer") or row.get("answer"),
        "sources": _safe_json_loads(sources or "[]"),
        "namespace": answer_row.get("namespace") or row.get("namespace"),
        "cached": answer_row.get("cached", row.get("cached", False)),
        "error": answer_row.get("error", row.get("error")),
        "ask_count": row.get("ask_count", 1),
        "answer_ask_count": answer_row.get("ask_count"),
        "last_asked_at": row.get("last_asked_at"),
        "created_at": row.get("created_at"),
    }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
