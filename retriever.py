"""
retriever.py — Pinecone retrieval + cross-encoder reranking.

Two-stage retrieval (industry standard):
  Stage 1 — ANN search: fast approximate vector search, retrieves top_k=8
  Stage 2 — Reranker:   slower but more accurate cross-encoder, trims to top_k_final=5

Your Pinecone metadata schema (from the sample record):
  has_numbers: bool
  has_dates:   bool
  has_table:   bool
  filename:    str
  source_url:  str
  chunk_index: int
  dataset:     str
  namespace:   str

Query-type detection maps to these metadata filters.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import SecretStr
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from config import cfg
from observability import get_logger

log = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """Typed wrapper around a Langchain Document + retrieval metadata."""
    content: str
    filename: str
    source_url: str
    chunk_index: int
    score: float = 0.0          # filled in by reranker
    metadata: Optional[dict] = None       # full raw metadata

    @classmethod
    def from_document(cls, doc: Document, score: float = 0.0) -> "RetrievedChunk":
        meta = doc.metadata or {}
        return cls(
            content     = doc.page_content,
            filename    = meta.get("filename", meta.get("source", "unknown")),
            source_url  = meta.get("source_url", ""),
            chunk_index = int(meta.get("chunk_index", 0)),
            score       = score,
            metadata    = meta,
        )

    def to_source_dict(self) -> dict:
        return {
            "source_file":  self.filename,
            "source_url":   self.source_url,
            "chunk_index":  self.chunk_index,
            "score":        round(self.score, 4),
            "preview":      self.content[:300] + ("..." if len(self.content) > 300 else ""),
        }


# ── Query analysis ────────────────────────────────────────────────────────────

_NUMERICAL_RE = re.compile(
    r"\b(how much|how many|amount|total|cost|paid|price|number of|count|"
    r"percent|percentage|million|billion)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(when|date|year|month|timeline|period|since|until|between)\b",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(
    r"\b(list|table|all|who were|names of|enumerate|summarise|summarize)\b",
    re.IGNORECASE,
)


def _detect_query_type(question: str) -> Optional[dict]:
    """
    Return a Pinecone metadata filter dict, or None (→ unfiltered search).
    Regex is more precise than word-in-string membership checks.
    """
    if _NUMERICAL_RE.search(question):
        return {"has_numbers": True}
    if _DATE_RE.search(question):
        return {"has_dates": True}
    if _TABLE_RE.search(question):
        return {"has_table": True}
    return None


# ── Vector store factory ───────────────────────────────────────────────────────
def build_vector_store() -> PineconeVectorStore:
    """
    Build and return a PineconeVectorStore connected to the existing index.
    Call once at startup and re-use — the embedding model is stateless.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=cfg.pinecone.embedding_model,
        api_key=SecretStr(cfg.gemini_api_key),
    )

    store = PineconeVectorStore.from_existing_index(
        index_name=cfg.pinecone.index_name,
        embedding=embeddings,
        namespace=cfg.pinecone.namespace,
    )
    log.info(
        "Connected to Pinecone index=%s namespace=%s",
        cfg.pinecone.index_name,
        cfg.pinecone.namespace,
    )
    return store


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_chunks(
    store: PineconeVectorStore,
    question: str,
    *,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Stage-1 retrieval: ANN search with optional metadata filter + fallback.

    Returns raw chunks (before reranking) as typed RetrievedChunk objects.
    """
    top_k = top_k or cfg.pinecone.top_k
    ns    = cfg.pinecone.namespace
    meta_filter = _detect_query_type(question)

    def _search(f: dict | None) -> list[Document]:
        kwargs: dict[str, Any] = dict(query=question, k=top_k, namespace=ns)
        if f:
            kwargs["filter"] = f
        return store.similarity_search(**kwargs)

    docs = _search(meta_filter)

    # Fallback: if the filter returned nothing, retry without it
    if not docs and meta_filter:
        log.warning(
            "Metadata filter %s returned 0 results — falling back to unfiltered search",
            meta_filter,
        )
        docs = _search(None)

    chunks = [RetrievedChunk.from_document(d) for d in docs]
    log.info("Stage-1 retrieved %d chunks for query='%s...'", len(chunks), question[:60])
    return chunks


# ── Reranker ───────────────────────────────────────────────────────────────────

def rerank_chunks(
    chunks: list[RetrievedChunk],
    question: str,
    *,
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """
    Stage-2 reranking: keyword-overlap scoring (zero extra dependencies).

    In production, swap this for:
      • Cohere Rerank API  (best quality, one HTTP call)
      • cross-encoder/ms-marco-MiniLM-L-6-v2  (local, fast)
      • Pinecone's built-in rerank endpoint (if on Enterprise plan)

    The interface is identical regardless of backend:
        input  → list[RetrievedChunk] + question
        output → list[RetrievedChunk] sorted by score, trimmed to top_n
    """
    top_n = top_n or cfg.pinecone.top_k_final

    if not chunks:
        return chunks

    # --- swap in a real reranker here ---
    try:
        return _cohere_rerank(chunks, question, top_n)
    except Exception:
        log.warning("Cohere reranker unavailable, falling back to keyword scorer")
        return _keyword_score_rerank(chunks, question, top_n)


def _keyword_score_rerank(
    chunks: list[RetrievedChunk],
    question: str,
    top_n: int,
) -> list[RetrievedChunk]:
    """
    Fallback reranker: BM25-inspired TF-IDF keyword scorer + deduplication.

    Improvements over the original simple overlap scorer:
      1. Deduplication  — identical (filename, chunk_index) pairs are removed
                          before scoring; keeps the highest-scoring copy only.
      2. IDF weighting  — rare question terms that appear in few chunks get
                          higher weight than common terms like "the" / "was".
      3. Length norm    — longer chunks don't automatically outscore short ones.
      4. Position bonus — a small boost when question terms appear early in
                          the chunk (titles / first sentences tend to be
                          more on-topic than trailing context).

    This is still a lightweight heuristic.  For production quality, install
    Cohere Rerank (see _cohere_rerank below) — it will outperform this on
    all query types.
    """
    import math

    # ── 1. Deduplicate by (filename, chunk_index) ─────────────────────────────
    seen: set[tuple[str, int]] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        key = (chunk.filename, chunk.chunk_index)
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    if not unique:
        return []

    # ── 2. Tokenise question (remove stopwords + punctuation) ─────────────────
    _STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "do", "did", "does", "in", "of", "to", "and", "or", "for",
        "at", "by", "with", "from", "that", "this", "it", "as", "on",
        "how", "what", "who", "when", "where", "which", "according",
    }
    def tokenise(text: str) -> list[str]:
        return [
            t for t in re.sub(r"[^\w\s]", "", text.lower()).split()
            if t not in _STOPWORDS and len(t) > 1
        ]

    q_tokens = tokenise(question)
    if not q_tokens:
        # Degenerate case: no meaningful tokens → return deduplicated chunks as-is
        return unique[:top_n]

    q_token_set = set(q_tokens)
    N = len(unique)

    # ── 3. Compute IDF for each question token ────────────────────────────────
    # df[t] = number of chunks containing token t
    df: dict[str, int] = {}
    chunk_token_lists: list[list[str]] = []
    for chunk in unique:
        tokens = tokenise(chunk.content)
        chunk_token_lists.append(tokens)
        for t in set(tokens) & q_token_set:
            df[t] = df.get(t, 0) + 1

    def idf(t: str) -> float:
        return math.log((N + 1) / (df.get(t, 0) + 1)) + 1.0

    # ── 4. Score each chunk ───────────────────────────────────────────────────
    for chunk, tokens in zip(unique, chunk_token_lists):
        if not tokens:
            chunk.score = 0.0
            continue

        token_counts: dict[str, int] = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        # TF-IDF with length normalisation (k1=1.5, b=0.75 — standard BM25 params)
        avg_len = sum(len(tl) for tl in chunk_token_lists) / max(N, 1)
        k1, b = 1.5, 0.75
        score = 0.0
        for t in q_token_set:
            if t not in token_counts:
                continue
            tf = token_counts[t]
            norm_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tokens) / avg_len))
            score += norm_tf * idf(t)

        # Position bonus: boost if question terms appear in first 20% of chunk
        early_tokens = set(tokens[: max(1, len(tokens) // 5)])
        early_hits = len(q_token_set & early_tokens)
        position_bonus = 0.1 * early_hits

        chunk.score = round(score + position_bonus, 4)

    ranked = sorted(unique, key=lambda c: c.score, reverse=True)
    return ranked[:top_n]


def _cohere_rerank(
    chunks: list[RetrievedChunk],
    question: str,
    top_n: int,
) -> list[RetrievedChunk]:
    """
    Production reranker using the Cohere Rerank API (SDK v5+).
    Requires: pip install cohere  +  COHERE_API_KEY env var.

    Model choice:
      rerank-v3.5          — latest multilingual, best quality
      rerank-english-v3.0  — English-only, slightly faster

    Cohere v5 SDK notes:
      - Use cohere.ClientV2, not cohere.Client (deprecated in v5)
      - response.results is a list of RerankResponseResultsItem
      - each item has .index (int) and .relevance_score (float 0-1)
      - documents can be plain strings; no need to wrap in dicts
    """
    import os
    import cohere

    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY not set")

    # Deduplicate before sending to Cohere (saves tokens + cost)
    seen: set[tuple[str, int]] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        key = (chunk.filename, chunk.chunk_index)
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    co = cohere.ClientV2(api_key=api_key)

    response = co.rerank(
        model     = "rerank-v3.5",
        query     = question,
        documents = [c.content for c in unique],
        top_n     = top_n,
    )

    reranked: list[RetrievedChunk] = []
    for result in response.results:
        chunk = unique[result.index]
        chunk.score = round(result.relevance_score, 4)
        reranked.append(chunk)

    log.info(
        "Cohere rerank: %d → %d chunks, top score=%.4f",
        len(unique),
        len(reranked),
        reranked[0].score if reranked else 0.0,
    )
    return reranked