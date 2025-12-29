"""Cache service for high-performance data storage and retrieval.

Provides Redis-based caching with fallback to in-memory storage
for development environments.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from config import application_settings
from app.core.exceptions import CacheError

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

logger = logging.getLogger(__name__)


class CacheService:
    """Async cache service with Redis backend and in-memory fallback."""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False
        self.using_redis = False

    async def initialize(self) -> None:
        """Initialize cache service with Redis or fallback to memory."""
        try:
            if REDIS_AVAILABLE and not application_settings.is_development:
                redis_config = application_settings.get_redis_configuration()

                self.redis_client = redis.from_url(
                    redis_config["url"],
                    password=redis_config.get("password"),
                    max_connections=redis_config["max_connections"],
                    decode_responses=True,
                )

                # Test connection
                await self.redis_client.ping()
                self.using_redis = True
                logger.info("Cache service initialized with Redis backend")

            else:
                logger.info("Cache service initialized with in-memory backend")

            self.is_initialized = True

        except Exception as e:
            logger.warning(
                f"Redis connection failed, falling back to memory cache: {e}"
            )
            self.using_redis = False
            self.is_initialized = True

    async def health_check(self) -> bool:
        """Check cache service health."""
        if not self.is_initialized:
            return False

        try:
            if self.using_redis and self.redis_client:
                await self.redis_client.ping()
            return True

        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return False

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        if not self.is_initialized:
            raise CacheError("Cache service not initialized")

        try:
            if self.using_redis and self.redis_client:
                value = await self.redis_client.get(key)
                return json.loads(value) if value else None
            else:
                cache_entry = self.memory_cache.get(key)
                if cache_entry and self._is_valid_entry(cache_entry):
                    return cache_entry["data"]
                elif cache_entry:
                    # Remove expired entry
                    del self.memory_cache[key]
                return None

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None

    async def set(
        self, key: str, value: Dict[str, Any], expire_seconds: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional expiration."""
        if not self.is_initialized:
            raise CacheError("Cache service not initialized")

        try:
            if self.using_redis and self.redis_client:
                await self.redis_client.setex(
                    key,
                    expire_seconds or 3600,  # Default 1 hour
                    json.dumps(value),
                )
            else:
                # In-memory cache with expiration
                expiry_time = datetime.utcnow()
                if expire_seconds:
                    expiry_time += timedelta(seconds=expire_seconds)
                else:
                    expiry_time += timedelta(hours=1)  # Default 1 hour

                self.memory_cache[key] = {"data": value, "expires_at": expiry_time}

            return True

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self.is_initialized:
            raise CacheError("Cache service not initialized")

        try:
            if self.using_redis and self.redis_client:
                result = await self.redis_client.delete(key)
                return bool(result)
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    return True
                return False

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.is_initialized:
            return False

        try:
            if self.using_redis and self.redis_client:
                return bool(await self.redis_client.exists(key))
            else:
                cache_entry = self.memory_cache.get(key)
                return cache_entry is not None and self._is_valid_entry(cache_entry)

        except Exception as e:
            logger.error(f"Cache exists check error for key {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter in cache."""
        if not self.is_initialized:
            raise CacheError("Cache service not initialized")

        try:
            if self.using_redis and self.redis_client:
                return await self.redis_client.incrby(key, amount)
            else:
                current_value = await self.get(key) or {"count": 0}
                new_value = current_value.get("count", 0) + amount
                await self.set(key, {"count": new_value})
                return new_value

        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            raise CacheError(f"Cache increment failed: {e}")

    async def get_keys_pattern(self, pattern: str) -> List[str]:
        """Get all keys matching a pattern."""
        if not self.is_initialized:
            return []

        try:
            if self.using_redis and self.redis_client:
                return await self.redis_client.keys(pattern)
            else:
                # Simple pattern matching for in-memory cache
                import fnmatch

                matching_keys = []
                for key in self.memory_cache.keys():
                    if fnmatch.fnmatch(key, pattern):
                        cache_entry = self.memory_cache[key]
                        if self._is_valid_entry(cache_entry):
                            matching_keys.append(key)
                return matching_keys

        except Exception as e:
            logger.error(f"Cache pattern search error for pattern {pattern}: {e}")
            return []

    async def cleanup_expired(self) -> int:
        """Clean up expired entries (mainly for in-memory cache)."""
        if not self.is_initialized or self.using_redis:
            return 0

        try:
            expired_keys = []
            current_time = datetime.utcnow()

            for key, entry in self.memory_cache.items():
                if current_time > entry.get("expires_at", current_time):
                    expired_keys.append(key)

            for key in expired_keys:
                del self.memory_cache[key]

            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            return 0

    # Document-specific cache methods
    async def update_document_status(
        self, document_id: str, status_data: Dict[str, Any]
    ) -> bool:
        """Update document processing status in cache."""
        cache_key = f"doc_status:{document_id}"
        status_data["last_updated"] = datetime.utcnow().isoformat()
        return await self.set(cache_key, status_data, expire_seconds=7200)  # 2 hours

    async def get_document_status(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document processing status from cache."""
        cache_key = f"doc_status:{document_id}"
        return await self.get(cache_key)

    async def cache_document_result(
        self, document_id: str, result_data: Dict[str, Any]
    ) -> bool:
        """Cache document processing results."""
        cache_key = f"doc_result:{document_id}"
        return await self.set(cache_key, result_data, expire_seconds=86400)  # 24 hours

    async def get_cached_document_result(
        self, document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached document processing results."""
        cache_key = f"doc_result:{document_id}"
        return await self.get(cache_key)

    async def increment_user_upload_count(self, user_id: str) -> int:
        """Increment user's daily upload count."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        cache_key = f"user_uploads:{user_id}:{today}"

        # Set expiration to end of day
        tomorrow = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        expire_seconds = int((tomorrow - datetime.utcnow()).total_seconds())

        count = await self.increment(cache_key)

        if count == 1:  # First increment, set expiration
            await self.set(cache_key, {"count": count}, expire_seconds=expire_seconds)

        return count

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        stats = {
            "backend": "redis" if self.using_redis else "memory",
            "initialized": self.is_initialized,
            "healthy": await self.health_check(),
        }

        try:
            if self.using_redis and self.redis_client:
                info = await self.redis_client.info()
                stats.update(
                    {
                        "connected_clients": info.get("connected_clients", 0),
                        "used_memory_human": info.get("used_memory_human", "0B"),
                        "keyspace_hits": info.get("keyspace_hits", 0),
                        "keyspace_misses": info.get("keyspace_misses", 0),
                    }
                )
            else:
                stats.update(
                    {
                        "memory_entries": len(self.memory_cache),
                        "expired_cleaned": await self.cleanup_expired(),
                    }
                )

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            stats["error"] = str(e)

        return stats

    async def cleanup(self) -> None:
        """Clean up cache service resources."""
        try:
            if self.using_redis and self.redis_client:
                await self.redis_client.close()
            else:
                self.memory_cache.clear()

            logger.info("Cache service cleaned up")

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    def _is_valid_entry(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is still valid (not expired)."""
        expires_at = cache_entry.get("expires_at")
        if not expires_at:
            return True

        return datetime.utcnow() < expires_at
