"""
Enhanced Redis Caching with Intelligent TTL Management.

Provides advanced caching strategies with content-aware TTL,
cache warming, and performance optimization.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheCategory(str, Enum):
    """Categories for cache entries with different TTL strategies."""
    LLM_RESPONSE = "llm_response"         # Short TTL - responses may vary
    EMBEDDING = "embedding"                # Long TTL - embeddings are stable
    DOCUMENT = "document"                  # Medium TTL - document data
    USER_SESSION = "user_session"          # Session-based TTL
    RATE_LIMIT = "rate_limit"              # Very short TTL
    SEARCH_RESULT = "search_result"        # Short TTL - search results change
    STATIC_DATA = "static_data"            # Very long TTL


@dataclass
class TTLStrategy:
    """TTL configuration for a cache category.
    
    Attributes:
        base_ttl_seconds: Base TTL for the category.
        max_ttl_seconds: Maximum TTL allowed.
        min_ttl_seconds: Minimum TTL allowed.
        adaptive: Whether to use adaptive TTL based on access patterns.
        jitter_percent: Random jitter to prevent thundering herd.
    """
    base_ttl_seconds: int
    max_ttl_seconds: int
    min_ttl_seconds: int = 60
    adaptive: bool = True
    jitter_percent: float = 0.1


# Default TTL strategies for each category
DEFAULT_TTL_STRATEGIES: Dict[CacheCategory, TTLStrategy] = {
    CacheCategory.LLM_RESPONSE: TTLStrategy(
        base_ttl_seconds=300,      # 5 minutes
        max_ttl_seconds=3600,      # 1 hour
        min_ttl_seconds=60
    ),
    CacheCategory.EMBEDDING: TTLStrategy(
        base_ttl_seconds=86400,    # 24 hours
        max_ttl_seconds=604800,    # 1 week
        min_ttl_seconds=3600,
        adaptive=False             # Embeddings don't change
    ),
    CacheCategory.DOCUMENT: TTLStrategy(
        base_ttl_seconds=3600,     # 1 hour
        max_ttl_seconds=86400,     # 24 hours
        min_ttl_seconds=600
    ),
    CacheCategory.USER_SESSION: TTLStrategy(
        base_ttl_seconds=1800,     # 30 minutes
        max_ttl_seconds=86400,     # 24 hours
        min_ttl_seconds=300
    ),
    CacheCategory.RATE_LIMIT: TTLStrategy(
        base_ttl_seconds=60,       # 1 minute
        max_ttl_seconds=300,       # 5 minutes
        min_ttl_seconds=10,
        adaptive=False
    ),
    CacheCategory.SEARCH_RESULT: TTLStrategy(
        base_ttl_seconds=600,      # 10 minutes
        max_ttl_seconds=3600,      # 1 hour
        min_ttl_seconds=120
    ),
    CacheCategory.STATIC_DATA: TTLStrategy(
        base_ttl_seconds=86400,    # 24 hours
        max_ttl_seconds=604800,    # 1 week
        min_ttl_seconds=3600,
        adaptive=False
    ),
}


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent TTL.
    
    Attributes:
        key: Cache key.
        value: Cached value.
        category: Cache category for TTL strategy.
        created_at: Creation timestamp.
        expires_at: Expiration timestamp.
        access_count: Number of accesses.
        last_accessed: Last access timestamp.
        content_hash: Hash of content for change detection.
    """
    key: str
    value: Any
    category: CacheCategory
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    content_hash: str = ""


class IntelligentTTLManager:
    """Manages intelligent TTL calculations based on access patterns."""
    
    def __init__(
        self,
        strategies: Optional[Dict[CacheCategory, TTLStrategy]] = None
    ):
        """Initialize TTL manager.
        
        Args:
            strategies: TTL strategies per category.
        """
        self.strategies = strategies or DEFAULT_TTL_STRATEGIES.copy()
        self._access_patterns: Dict[str, List[float]] = {}
    
    def calculate_ttl(
        self,
        key: str,
        category: CacheCategory,
        content_size: Optional[int] = None
    ) -> int:
        """Calculate optimal TTL for a cache entry.
        
        Args:
            key: Cache key.
            category: Cache category.
            content_size: Optional content size in bytes.
            
        Returns:
            TTL in seconds.
        """
        strategy = self.strategies.get(category, DEFAULT_TTL_STRATEGIES[CacheCategory.DOCUMENT])
        
        base_ttl = strategy.base_ttl_seconds
        
        if strategy.adaptive:
            # Adjust based on access frequency
            access_times = self._access_patterns.get(key, [])
            if len(access_times) >= 3:
                # Calculate average access interval
                intervals = [
                    access_times[i+1] - access_times[i]
                    for i in range(len(access_times) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals)
                
                # Frequently accessed items get longer TTL
                if avg_interval < 60:  # Accessed every minute
                    base_ttl = min(base_ttl * 2, strategy.max_ttl_seconds)
                elif avg_interval < 300:  # Every 5 minutes
                    base_ttl = min(int(base_ttl * 1.5), strategy.max_ttl_seconds)
                elif avg_interval > 3600:  # Less than hourly
                    base_ttl = max(base_ttl // 2, strategy.min_ttl_seconds)
        
        # Adjust for content size (larger items get shorter TTL to save memory)
        if content_size and content_size > 1_000_000:  # > 1MB
            base_ttl = max(base_ttl // 2, strategy.min_ttl_seconds)
        
        # Add jitter to prevent thundering herd
        import random
        jitter = random.uniform(
            -strategy.jitter_percent,
            strategy.jitter_percent
        )
        final_ttl = int(base_ttl * (1 + jitter))
        
        return max(strategy.min_ttl_seconds, min(final_ttl, strategy.max_ttl_seconds))
    
    def record_access(self, key: str) -> None:
        """Record a cache access for pattern analysis.
        
        Args:
            key: Cache key.
        """
        if key not in self._access_patterns:
            self._access_patterns[key] = []
        
        self._access_patterns[key].append(time.time())
        
        # Keep only last 10 accesses
        if len(self._access_patterns[key]) > 10:
            self._access_patterns[key] = self._access_patterns[key][-10:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTL manager statistics."""
        return {
            "tracked_keys": len(self._access_patterns),
            "strategies": {
                cat.value: {
                    "base_ttl": strat.base_ttl_seconds,
                    "max_ttl": strat.max_ttl_seconds,
                    "adaptive": strat.adaptive
                }
                for cat, strat in self.strategies.items()
            }
        }


class CacheWarmer:
    """Preloads frequently accessed cache entries."""
    
    def __init__(self):
        """Initialize cache warmer."""
        self._warm_keys: Dict[str, Callable] = {}
        self._last_warm: Dict[str, float] = {}
        self._warm_interval = 300  # 5 minutes
    
    def register(
        self,
        key: str,
        loader: Callable[[], Any],
        interval: Optional[int] = None
    ) -> None:
        """Register a key for cache warming.
        
        Args:
            key: Cache key to warm.
            loader: Function to load the value.
            interval: Optional custom warm interval.
        """
        self._warm_keys[key] = loader
        if interval:
            self._warm_interval = interval
    
    async def warm_all(
        self,
        cache_set: Callable[[str, Any, int], None]
    ) -> Dict[str, bool]:
        """Warm all registered cache entries.
        
        Args:
            cache_set: Function to set cache values.
            
        Returns:
            Dict of key -> success status.
        """
        results = {}
        current_time = time.time()
        
        for key, loader in self._warm_keys.items():
            last = self._last_warm.get(key, 0)
            
            if current_time - last < self._warm_interval:
                results[key] = True  # Skip, recently warmed
                continue
            
            try:
                if asyncio.iscoroutinefunction(loader):
                    value = await loader()
                else:
                    value = loader()
                
                await cache_set(key, value, self._warm_interval * 2)
                self._last_warm[key] = current_time
                results[key] = True
                logger.debug(f"Warmed cache key: {key}")
            except Exception as e:
                logger.warning(f"Failed to warm cache key {key}: {e}")
                results[key] = False
        
        return results
    
    def unregister(self, key: str) -> None:
        """Unregister a key from warming."""
        self._warm_keys.pop(key, None)
        self._last_warm.pop(key, None)


class EnhancedCacheService:
    """Enhanced cache service with intelligent TTL and warming.
    
    Extends basic caching with:
    - Content-aware TTL strategies
    - Access pattern-based TTL adjustment
    - Cache warming for hot keys
    - Multi-tier caching support
    """
    
    def __init__(
        self,
        base_cache_service,
        ttl_manager: Optional[IntelligentTTLManager] = None,
        warmer: Optional[CacheWarmer] = None
    ):
        """Initialize enhanced cache service.
        
        Args:
            base_cache_service: Base CacheService instance.
            ttl_manager: Optional TTL manager.
            warmer: Optional cache warmer.
        """
        self._base = base_cache_service
        self.ttl_manager = ttl_manager or IntelligentTTLManager()
        self.warmer = warmer or CacheWarmer()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._sets = 0
    
    async def get(
        self,
        key: str,
        category: CacheCategory = CacheCategory.DOCUMENT
    ) -> Optional[Any]:
        """Get value with access tracking.
        
        Args:
            key: Cache key.
            category: Cache category.
            
        Returns:
            Cached value or None.
        """
        value = await self._base.get(key)
        
        if value is not None:
            self._hits += 1
            self.ttl_manager.record_access(key)
        else:
            self._misses += 1
        
        return value
    
    async def set(
        self,
        key: str,
        value: Any,
        category: CacheCategory = CacheCategory.DOCUMENT,
        custom_ttl: Optional[int] = None
    ) -> None:
        """Set value with intelligent TTL.
        
        Args:
            key: Cache key.
            value: Value to cache.
            category: Cache category for TTL strategy.
            custom_ttl: Optional override TTL.
        """
        import json
        content_size = len(json.dumps(value)) if value else 0
        
        ttl = custom_ttl or self.ttl_manager.calculate_ttl(
            key, category, content_size
        )
        
        await self._base.set(key, value, expire_seconds=ttl)
        self._sets += 1
    
    async def get_or_set(
        self,
        key: str,
        loader: Callable[[], Any],
        category: CacheCategory = CacheCategory.DOCUMENT,
        custom_ttl: Optional[int] = None
    ) -> Any:
        """Get from cache or compute and cache.
        
        Args:
            key: Cache key.
            loader: Function to compute value if not cached.
            category: Cache category.
            custom_ttl: Optional override TTL.
            
        Returns:
            Cached or computed value.
        """
        value = await self.get(key, category)
        
        if value is None:
            if asyncio.iscoroutinefunction(loader):
                value = await loader()
            else:
                value = loader()
            
            await self.set(key, value, category, custom_ttl)
        
        return value
    
    async def warm_cache(self) -> Dict[str, bool]:
        """Run cache warming for registered keys."""
        return await self.warmer.warm_all(
            lambda k, v, t: self._base.set(k, v, expire_seconds=t)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "hit_rate_percent": round(hit_rate, 2),
            "ttl_manager": self.ttl_manager.get_stats(),
            "warmed_keys": len(self.warmer._warm_keys)
        }


# Global instances
_ttl_manager: Optional[IntelligentTTLManager] = None
_cache_warmer: Optional[CacheWarmer] = None


def get_ttl_manager() -> IntelligentTTLManager:
    """Get global TTL manager."""
    global _ttl_manager
    if _ttl_manager is None:
        _ttl_manager = IntelligentTTLManager()
    return _ttl_manager


def get_cache_warmer() -> CacheWarmer:
    """Get global cache warmer."""
    global _cache_warmer
    if _cache_warmer is None:
        _cache_warmer = CacheWarmer()
    return _cache_warmer
