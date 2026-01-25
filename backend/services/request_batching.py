"""
Request Batching Service for LLM Inference.

Provides intelligent request batching to optimize LLM throughput
by grouping similar requests and reducing API call overhead.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single request in a batch.
    
    Attributes:
        request_id: Unique request identifier.
        prompt: The prompt text.
        system_prompt: Optional system prompt.
        kwargs: Additional generation parameters.
        future: Future to resolve with result.
        created_at: Request creation timestamp.
    """
    request_id: str
    prompt: str
    system_prompt: Optional[str]
    kwargs: Dict[str, Any]
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)
    
    @property
    def age_ms(self) -> float:
        """Get request age in milliseconds."""
        return (time.time() - self.created_at) * 1000


@dataclass
class BatchResult:
    """Result from batch processing.
    
    Attributes:
        request_id: Original request ID.
        content: Generated content.
        success: Whether generation succeeded.
        error: Error message if failed.
        latency_ms: Processing latency.
    """
    request_id: str
    content: str
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0.0


class RequestBatcher:
    """Intelligent request batching for LLM inference.
    
    Groups similar requests together and processes them in batches
    to optimize throughput and reduce latency.
    
    Features:
    - Time-based batch triggering
    - Size-based batch triggering
    - Prompt similarity grouping
    - Request deduplication
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: float = 100.0,
        enable_deduplication: bool = True,
        similarity_threshold: float = 0.9
    ):
        """Initialize request batcher.
        
        Args:
            max_batch_size: Maximum requests per batch.
            max_wait_ms: Maximum wait time before processing.
            enable_deduplication: Deduplicate identical requests.
            similarity_threshold: Threshold for grouping similar requests.
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.enable_deduplication = enable_deduplication
        self.similarity_threshold = similarity_threshold
        
        self._pending_requests: Dict[str, BatchRequest] = {}
        self._request_hashes: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._process_task: Optional[asyncio.Task] = None
        self._batch_processor: Optional[Callable] = None
        
        # Statistics
        self._batches_processed = 0
        self._requests_processed = 0
        self._dedup_hits = 0
    
    def set_batch_processor(
        self,
        processor: Callable[[List[BatchRequest]], List[BatchResult]]
    ) -> None:
        """Set the batch processing function.
        
        Args:
            processor: Async function that processes a batch of requests.
        """
        self._batch_processor = processor
    
    async def submit(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Submit a request for batched processing.
        
        Args:
            prompt: The prompt text.
            system_prompt: Optional system prompt.
            **kwargs: Additional generation parameters.
            
        Returns:
            Generated content.
        """
        request_id = self._generate_request_id()
        future = asyncio.get_event_loop().create_future()
        
        request = BatchRequest(
            request_id=request_id,
            prompt=prompt,
            system_prompt=system_prompt,
            kwargs=kwargs,
            future=future
        )
        
        async with self._lock:
            # Check for duplicate request
            if self.enable_deduplication:
                prompt_hash = self._hash_prompt(prompt, system_prompt)
                
                if prompt_hash in self._request_hashes:
                    # Reuse existing request's future
                    existing_ids = self._request_hashes[prompt_hash]
                    if existing_ids:
                        existing_request = self._pending_requests.get(existing_ids[0])
                        if existing_request:
                            self._dedup_hits += 1
                            logger.debug(f"Deduplicating request {request_id}")
                            return await existing_request.future
                
                self._request_hashes[prompt_hash].append(request_id)
            
            self._pending_requests[request_id] = request
            
            # Start processing if batch is ready
            if len(self._pending_requests) >= self.max_batch_size:
                asyncio.create_task(self._process_batch())
            elif self._process_task is None or self._process_task.done():
                self._process_task = asyncio.create_task(self._wait_and_process())
        
        return await future
    
    async def _wait_and_process(self) -> None:
        """Wait for max_wait_ms then process pending requests."""
        await asyncio.sleep(self.max_wait_ms / 1000.0)
        await self._process_batch()
    
    async def _process_batch(self) -> None:
        """Process the current batch of requests."""
        async with self._lock:
            if not self._pending_requests:
                return
            
            # Get batch of requests
            requests = list(self._pending_requests.values())[:self.max_batch_size]
            
            # Remove from pending
            for req in requests:
                self._pending_requests.pop(req.request_id, None)
            
            # Clear hash entries
            if self.enable_deduplication:
                for req in requests:
                    prompt_hash = self._hash_prompt(req.prompt, req.system_prompt)
                    if prompt_hash in self._request_hashes:
                        self._request_hashes[prompt_hash] = [
                            rid for rid in self._request_hashes[prompt_hash]
                            if rid != req.request_id
                        ]
                        if not self._request_hashes[prompt_hash]:
                            del self._request_hashes[prompt_hash]
        
        if not requests:
            return
        
        logger.info(f"Processing batch of {len(requests)} requests")
        self._batches_processed += 1
        
        try:
            if self._batch_processor:
                results = await self._batch_processor(requests)
                
                # Resolve futures
                result_map = {r.request_id: r for r in results}
                for req in requests:
                    result = result_map.get(req.request_id)
                    if result and result.success:
                        req.future.set_result(result.content)
                    elif result:
                        req.future.set_exception(Exception(result.error))
                    else:
                        req.future.set_exception(Exception("No result returned"))
            else:
                # No processor set, fail all requests
                for req in requests:
                    req.future.set_exception(
                        Exception("Batch processor not configured")
                    )
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            for req in requests:
                if not req.future.done():
                    req.future.set_exception(e)
        
        self._requests_processed += len(requests)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _hash_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Create hash for deduplication."""
        content = f"{system_prompt or ''}|{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics.
        
        Returns:
            Dictionary with statistics.
        """
        return {
            "batches_processed": self._batches_processed,
            "requests_processed": self._requests_processed,
            "pending_requests": len(self._pending_requests),
            "dedup_hits": self._dedup_hits,
            "avg_batch_size": (
                self._requests_processed / self._batches_processed
                if self._batches_processed > 0 else 0
            )
        }
    
    async def flush(self) -> None:
        """Process all pending requests immediately."""
        while self._pending_requests:
            await self._process_batch()


class MemoizationCache:
    """Cache for memoizing LLM responses.
    
    Caches responses based on prompt content with configurable TTL.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: float = 3600.0
    ):
        """Initialize memoization cache.
        
        Args:
            max_size: Maximum cache entries.
            default_ttl_seconds: Default TTL for entries.
        """
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, Tuple[str, float]] = {}  # hash -> (content, expiry)
        self._access_order: List[str] = []
    
    def get(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Get cached response.
        
        Args:
            prompt: Prompt text.
            system_prompt: Optional system prompt.
            
        Returns:
            Cached content or None.
        """
        key = self._make_key(prompt, system_prompt)
        
        if key in self._cache:
            content, expiry = self._cache[key]
            if time.time() < expiry:
                # Update access order
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return content
            else:
                # Expired
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
        
        return None
    
    def set(
        self,
        prompt: str,
        content: str,
        system_prompt: Optional[str] = None,
        ttl: Optional[float] = None
    ) -> None:
        """Cache a response.
        
        Args:
            prompt: Prompt text.
            content: Response content.
            system_prompt: Optional system prompt.
            ttl: Optional custom TTL.
        """
        key = self._make_key(prompt, system_prompt)
        expiry = time.time() + (ttl or self.default_ttl)
        
        # Evict if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
        
        self._cache[key] = (content, expiry)
        self._access_order.append(key)
    
    def _make_key(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Create cache key."""
        content = f"{system_prompt or ''}|{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
    
    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# Global instances
_batcher: Optional[RequestBatcher] = None
_memo_cache: Optional[MemoizationCache] = None


def get_request_batcher() -> RequestBatcher:
    """Get the global request batcher.
    
    Returns:
        RequestBatcher singleton.
    """
    global _batcher
    if _batcher is None:
        _batcher = RequestBatcher()
    return _batcher


def get_memoization_cache() -> MemoizationCache:
    """Get the global memoization cache.
    
    Returns:
        MemoizationCache singleton.
    """
    global _memo_cache
    if _memo_cache is None:
        _memo_cache = MemoizationCache()
    return _memo_cache
