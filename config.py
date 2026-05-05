"""
config.py — Centralised configuration for the RAG pipeline.
All tunables live here. Never scatter magic numbers across modules.
"""
from __future__ import annotations

import os 
from dataclasses import dataclass,field
from dotenv import load_dotenv
load_dotenv()


@dataclass(frozen=True)
class PineconeConfig:
    index_name: str = "gov-transparency-index"
    namespace: str = "epstein-docs"
    top_k: int = 10  #fetching from pinecone, we get top 10 results, then we rerank them and trim to top_k_final before sending to the LLM
    top_k_final: int = 5
    embedding_model: str = "gemini-embedding-001"

@dataclass(frozen=True)
class LLMConfig:
    model: str         = "llama-3.3-70b-versatile"
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p:float = 0.9
    stream: bool = True


@dataclass(frozen=True)
class CacheConfig:
    enabled: bool    = True
    ttl_seconds: int = 3600   # Redis EXPIRE duration

    # Supported: redis://localhost:6379/0  |  rediss://host:6380/0  (TLS)
    @staticmethod
    def redis_url() -> str:
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int   = 3
    backoff_base: float = 1.5


@dataclass(frozen=True)
class AppConfig:
    pinecone: PineconeConfig = field(default_factory=PineconeConfig)
    llm: LLMConfig           = field(default_factory=LLMConfig)
    cache: CacheConfig       = field(default_factory=CacheConfig)
    retry: RetryConfig       = field(default_factory=RetryConfig)
    log_level: str           = "INFO"

    @property
    def gemini_api_key(self) -> str:
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise EnvironmentError("GEMINI_API_KEY not set")
        return key

    @property
    def groq_api_key(self) -> str:
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            raise EnvironmentError("GROQ_API_KEY not set")
        return key

    @property
    def pinecone_api_key(self) -> str:
        key = os.getenv("PINECONE_API_KEY", "")
        if not key:
            raise EnvironmentError("PINECONE_API_KEY not set")
        return key

    @property
    def supabase_url(self) -> str:
        return os.getenv("SUPABASE_URL", "")

    @property
    def supabase_key(self) -> str:
        return os.getenv("SUPABASE_KEY", "")


cfg = AppConfig(
    pinecone=PineconeConfig(),
    llm=LLMConfig(),
    cache=CacheConfig(),
    retry=RetryConfig(),
)