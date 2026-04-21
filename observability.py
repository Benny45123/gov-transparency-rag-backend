"""
observability.py — Structured logging + per-request tracing.

Every RAG query emits a single JSON trace log at the end, capturing:
  - query text
  - latency breakdown (retrieve / rerank / generate)
  - chunk count and sources
  - cache hit/miss
  - any errors

Usage:
    from observability import Tracer, get_logger
    log = get_logger(__name__)

    with Tracer("rag_query") as t:
        t.set("query", question)
        chunks = retrieve(...)
        t.set("n_chunks", len(chunks))
        ...
    # → logs full trace JSON on __exit__
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from config import cfg


# ── Logging setup ──────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, cfg.log_level, logging.INFO))
    return logger


_root_logger = get_logger("rag")


# ── Per-request tracer ─────────────────────────────────────────────────────────

@dataclass
class Tracer:
    """
    Context-manager that accumulates timing and metadata for one RAG request
    and emits a structured JSON log entry on exit.

    Example
    -------
    with Tracer("rag_query") as t:
        t.set("query", question)
        with t.span("retrieve"):
            chunks = retrieve(...)
        t.set("n_chunks", len(chunks))
    # ─▶ logs {"op":"rag_query","query":"...","n_chunks":5,"latency_ms":{...},...}
    """
    op: str
    _data: dict[str, Any]      = field(default_factory=dict, init=False)
    _spans: dict[str, float]   = field(default_factory=dict, init=False)
    _start: float              = field(default_factory=time.perf_counter, init=False)
    _error: str | None         = field(default=None, init=False)

    # public API
    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    @contextmanager
    def span(self, name: str) -> Generator[None, None, None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._spans[name] = round((time.perf_counter() - t0) * 1000, 1)

    # context-manager protocol
    def __enter__(self) -> "Tracer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        total_ms = round((time.perf_counter() - self._start) * 1000, 1)
        if exc_val:
            self._error = f"{exc_type.__name__}: {exc_val}"

        payload = {
            "op":          self.op,
            "total_ms":    total_ms,
            "latency_ms":  self._spans,
            "error":       self._error,
            **self._data,
        }
        level = logging.ERROR if self._error else logging.INFO
        _root_logger.log(level, json.dumps(payload, default=str))
        return False   # don't suppress exceptions


# ── Simple helpers ─────────────────────────────────────────────────────────────

def log_retrieval(query: str, n_chunks: int, cache_hit: bool, ms: float) -> None:
    _root_logger.info(
        json.dumps({
            "event":    "retrieval",
            "query":    query[:120],
            "n_chunks": n_chunks,
            "cache_hit": cache_hit,
            "ms":       round(ms, 1),
        })
    )


def log_generation(model: str, tokens_out: int, ms: float) -> None:
    _root_logger.info(
        json.dumps({
            "event":      "generation",
            "model":      model,
            "tokens_out": tokens_out,
            "ms":         round(ms, 1),
        })
    )