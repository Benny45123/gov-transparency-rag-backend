from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import redis
from redis.exceptions import RedisError

from config import cfg
from observability import get_logger
import os
from dotenv import load_dotenv
load_dotenv()
log = get_logger(__name__)

class RedisQueryCache:
    def __init__(self) -> None:
        self._enabled = cfg.cache.enabled
        self._ttl = cfg.cache.ttl_seconds
        self._client: redis.Redis | None = None

    def _get_client(self) -> redis.Redis | None:
        """Lazy-init Redis Cloud client. Returns None if connection fails."""
        if self._client is not None:
            return self._client
        
        if not self._enabled:
            return None

        try:
            self._client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),      
                port=int(os.getenv("REDIS_PORT", "6379")),    
                username=os.getenv("REDIS_USERNAME", "default"), 
                password=os.getenv("REDIS_PASSWORD", ""),   
                decode_responses=True,
                socket_connect_timeout=2,
                retry_on_timeout=True,
            )
            self._client.ping() 
            log.info("Connected to Redis Cloud")
        except RedisError as e:
            log.warning("Redis Cloud unavailable: %s", e)
            self._client = None
            
        return self._client

    @staticmethod
    def _normalise(query: str) -> str:
        return " ".join(query.lower().split())

    def _key(self, query: str) -> str:
        digest = hashlib.sha256(self._normalise(query).encode()).hexdigest()
        return f"rag:cache:{digest}"

    def get(self, query: str) -> dict | None:
        client = self._get_client()
        if not client: return None
        
        try:
            raw = client.get(self._key(query))
            if raw:
                log.info("Cache HIT: %.60s", query)
                return json.loads(raw)
        except (RedisError, json.JSONDecodeError) as e:
            log.error("Cache get error: %s", e)
        return None

    def set(self, query: str, payload: dict) -> None:
        client = self._get_client()
        if not client: return

        try:
            client.setex(
                name=self._key(query),
                time=self._ttl,
                value=json.dumps(payload, default=str)
            )
        except RedisError as e:
            log.error("Cache set error: %s", e)

    def invalidate(self, query: str) -> bool:
        client = self._get_client()
        return bool(client.delete(self._key(query))) if client else False

    def clear(self) -> int:
        """Clear our namespace. Note: .keys() is slow on huge DBs; use carefully."""
        client = self._get_client()
        if not client: return 0
        try:
            keys = client.keys("rag:cache:*")
            # If keys is not a concrete iterable (e.g. an awaitable from an async client), avoid unpacking.
            if not isinstance(keys, (list, tuple)) or not keys:
                return 0
            # client.delete may be typed as returning an Awaitable in some redis clients; cast to int for static typing.
            res = client.delete(*keys)
            return cast(int, res)
        except RedisError:
            return 0

    def stats(self) -> dict:
        client = self._get_client()
        if not client: return {"status": "unavailable"}
        try:
            return {
                "status": "connected",
                "count": len(client.keys("rag:cache:*")),
                "memory": client.info("memory").get("used_memory_human"),
            }
        except RedisError:
            return {"status": "error"}

# Singleton pattern
_cache = RedisQueryCache()

def get_cache() -> RedisQueryCache:
    return _cache