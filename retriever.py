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

_TOKEN_RE = re.compile(r"\b\w+\b")
_DOMAIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "children": ("child", "minor", "minors"),
    "child": ("children", "minor", "minors"),
    "victimized": ("victim", "victims", "abused", "assaulted", "exploited", "harmed"),
    "victim": ("victimized", "victims", "abused", "assaulted", "exploited"),
    "victims": ("victim", "victimized", "abused", "assaulted", "exploited"),
    "murdered": ("killed", "dead", "death", "homicide"),
    "raped": ("rape", "sexual", "abuse", "assault"),
    "abused": ("abuse", "victimized", "assaulted", "exploited"),
}
_BOOST_TERMS = {
    "child", "children", "minor", "minors", "victim", "victims",
    "victimized", "abuse", "abused", "assault", "assaulted",
    "rape", "raped", "murder", "murdered", "killed", "homicide",
}


def normalise_query(question: str) -> str:
    """Lowercase, remove punctuation, and normalize whitespace."""
    return " ".join(_TOKEN_RE.findall(question.lower()))


def expand_query(question: str) -> str:
    """
    Add a small domain-aware synonym expansion to improve recall on
    paraphrases without replacing the user's original wording.
    """
    tokens = normalise_query(question).split()
    expanded = list(tokens)
    seen = set(tokens)

    for token in tokens:
        for synonym in _DOMAIN_SYNONYMS.get(token, ()):
            if synonym not in seen:
                expanded.append(synonym)
                seen.add(synonym)

    return " ".join(expanded)


def merge_chunks(*chunk_lists: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate retrieved chunks while preserving first-seen order."""
    merged: list[RetrievedChunk] = []
    seen: set[tuple[str, int]] = set()

    for chunk_list in chunk_lists:
        for chunk in chunk_list:
            key = (chunk.filename, chunk.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)

    return merged


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
    Stage-1 retrieval: ANN search over the original and expanded query.

    Returns raw chunks (before reranking) as typed RetrievedChunk objects.
    """
    top_k = top_k or cfg.pinecone.top_k
    ns = cfg.pinecone.namespace

    def _search(search_query: str) -> list[Document]:
        kwargs: dict[str, Any] = dict(query=search_query, k=top_k, namespace=ns)
        return store.similarity_search(**kwargs)

    expanded_query = expand_query(question)
    base_chunks = [RetrievedChunk.from_document(d) for d in _search(question)]
    expanded_chunks: list[RetrievedChunk] = []
    if expanded_query != normalise_query(question):
        expanded_chunks = [RetrievedChunk.from_document(d) for d in _search(expanded_query)]

    chunks = merge_chunks(base_chunks, expanded_chunks)
    log.info(
        "Stage-1 retrieved %d chunks for query='%s...' (expanded=%s)",
        len(chunks),
        question[:60],
        expanded_query != normalise_query(question),
    )
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
    expanded_tokens = tokenise(expand_query(question))
    q_token_set = set(q_tokens)
    expanded_token_set = set(expanded_tokens)
    if not q_tokens:
        # Degenerate case: no meaningful tokens → return deduplicated chunks as-is
        return unique[:top_n]

    N = len(unique)

    # ── 3. Compute IDF for each question token ────────────────────────────────
    # df[t] = number of chunks containing token t
    df: dict[str, int] = {}
    chunk_token_lists: list[list[str]] = []
    for chunk in unique:
        tokens = tokenise(chunk.content)
        chunk_token_lists.append(tokens)
        for t in set(tokens) & expanded_token_set:
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
        for t in expanded_token_set:
            if t not in token_counts:
                continue
            tf = token_counts[t]
            norm_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tokens) / avg_len))
            score += norm_tf * idf(t)

        # Position bonus: boost if question terms appear in first 20% of chunk
        early_tokens = set(tokens[: max(1, len(tokens) // 5)])
        early_hits = len(expanded_token_set & early_tokens)
        position_bonus = 0.1 * early_hits

        boost_hits = len(_BOOST_TERMS & set(tokens) & expanded_token_set)
        exact_hits = len(q_token_set & set(tokens))
        semantic_bonus = 0.15 * boost_hits + 0.05 * exact_hits

        chunk.score = round(score + position_bonus + semantic_bonus, 4)

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
