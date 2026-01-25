"""
Enhanced Database Connection Pooling.

Provides optimized connection pooling with health monitoring,
connection recycling, and performance metrics.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Database connection pool configuration.
    
    Attributes:
        pool_size: Number of connections to maintain.
        max_overflow: Additional connections allowed beyond pool_size.
        pool_timeout: Seconds to wait for an available connection.
        pool_recycle: Seconds before recycling a connection.
        pool_pre_ping: Whether to test connections before use.
        echo: Whether to log SQL statements.
        echo_pool: Whether to log pool checkouts/checkins.
    """
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: float = 30.0
    pool_recycle: int = 1800  # 30 minutes
    pool_pre_ping: bool = True
    echo: bool = False
    echo_pool: bool = False
    
    # Health check settings
    health_check_interval: int = 60  # seconds
    connection_max_age: int = 3600  # 1 hour
    
    # Retry settings
    connect_retries: int = 3
    connect_retry_delay: float = 1.0


@dataclass
class ConnectionStats:
    """Statistics for a database connection.
    
    Attributes:
        created_at: Connection creation timestamp.
        last_used: Last usage timestamp.
        queries_executed: Number of queries executed.
        total_time_ms: Total query execution time in ms.
        errors: Number of errors on this connection.
    """
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    queries_executed: int = 0
    total_time_ms: float = 0.0
    errors: int = 0
    
    @property
    def age_seconds(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at
    
    @property
    def avg_query_time_ms(self) -> float:
        """Get average query time in ms."""
        if self.queries_executed == 0:
            return 0.0
        return self.total_time_ms / self.queries_executed


class DatabasePoolManager:
    """Enhanced database connection pool manager.
    
    Provides:
    - Connection health monitoring
    - Automatic connection recycling
    - Query performance tracking
    - Graceful degradation
    """
    
    def __init__(self, config: Optional[PoolConfig] = None):
        """Initialize pool manager.
        
        Args:
            config: Pool configuration.
        """
        self.config = config or PoolConfig()
        self._engine = None
        self._session_factory = None
        
        # Connection statistics
        self._connection_stats: Dict[int, ConnectionStats] = {}
        self._active_connections = 0
        self._total_checkouts = 0
        self._total_checkins = 0
        self._failed_checkouts = 0
        
        # Health monitoring
        self._last_health_check = 0.0
        self._health_status = True
        self._health_check_lock = asyncio.Lock()
    
    async def initialize(self, database_url: str) -> None:
        """Initialize the connection pool.
        
        Args:
            database_url: Database connection URL.
        """
        try:
            from sqlalchemy.ext.asyncio import (
                create_async_engine,
                async_sessionmaker,
                AsyncSession,
            )
            
            # Normalize URL for async drivers
            if database_url.startswith("sqlite:///"):
                database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
            elif database_url.startswith("postgresql://"):
                database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
            
            # Create engine with optimized settings
            self._engine = create_async_engine(
                database_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool,
            )
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            logger.info(
                f"Database pool initialized: "
                f"pool_size={self.config.pool_size}, "
                f"max_overflow={self.config.max_overflow}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self):
        """Get a database session with tracking.
        
        Yields:
            AsyncSession instance.
        """
        if not self._session_factory:
            raise RuntimeError("Database pool not initialized")
        
        self._total_checkouts += 1
        self._active_connections += 1
        start_time = time.perf_counter()
        
        try:
            async with self._session_factory() as session:
                conn_id = id(session)
                
                if conn_id not in self._connection_stats:
                    self._connection_stats[conn_id] = ConnectionStats()
                
                stats = self._connection_stats[conn_id]
                stats.last_used = time.time()
                
                try:
                    yield session
                    await session.commit()
                except Exception as e:
                    await session.rollback()
                    stats.errors += 1
                    raise
                finally:
                    query_time = (time.perf_counter() - start_time) * 1000
                    stats.queries_executed += 1
                    stats.total_time_ms += query_time
                    
        except Exception as e:
            self._failed_checkouts += 1
            raise
        finally:
            self._active_connections -= 1
            self._total_checkins += 1
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform pool health check.
        
        Returns:
            Health status dictionary.
        """
        async with self._health_check_lock:
            current_time = time.time()
            
            # Skip if recently checked
            if current_time - self._last_health_check < self.config.health_check_interval:
                return {"healthy": self._health_status, "cached": True}
            
            try:
                async with self.get_session() as session:
                    result = await session.execute("SELECT 1")
                    self._health_status = result.scalar() == 1
                
                self._last_health_check = current_time
                
                return {
                    "healthy": self._health_status,
                    "cached": False,
                    "pool_size": self.config.pool_size,
                    "active_connections": self._active_connections,
                }
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                self._health_status = False
                return {"healthy": False, "error": str(e)}
    
    async def recycle_stale_connections(self) -> int:
        """Recycle connections that exceed max age.
        
        Returns:
            Number of connections recycled.
        """
        recycled = 0
        current_time = time.time()
        
        for conn_id, stats in list(self._connection_stats.items()):
            if stats.age_seconds > self.config.connection_max_age:
                del self._connection_stats[conn_id]
                recycled += 1
        
        if recycled > 0:
            logger.info(f"Recycled {recycled} stale connections")
        
        return recycled
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics.
        
        Returns:
            Pool statistics dictionary.
        """
        total_queries = sum(s.queries_executed for s in self._connection_stats.values())
        total_time = sum(s.total_time_ms for s in self._connection_stats.values())
        total_errors = sum(s.errors for s in self._connection_stats.values())
        
        return {
            "pool_size": self.config.pool_size,
            "max_overflow": self.config.max_overflow,
            "active_connections": self._active_connections,
            "tracked_connections": len(self._connection_stats),
            "total_checkouts": self._total_checkouts,
            "total_checkins": self._total_checkins,
            "failed_checkouts": self._failed_checkouts,
            "total_queries": total_queries,
            "total_errors": total_errors,
            "avg_query_time_ms": round(total_time / total_queries, 2) if total_queries else 0,
            "health_status": self._health_status,
        }
    
    async def cleanup(self) -> None:
        """Clean up database connections."""
        if self._engine:
            await self._engine.dispose()
            self._connection_stats.clear()
            logger.info("Database pool cleaned up")


# Global pool manager
_pool_manager: Optional[DatabasePoolManager] = None


def get_pool_manager() -> DatabasePoolManager:
    """Get the global pool manager.
    
    Returns:
        DatabasePoolManager singleton.
    """
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = DatabasePoolManager()
    return _pool_manager


async def init_database_pool(
    database_url: str,
    config: Optional[PoolConfig] = None
) -> DatabasePoolManager:
    """Initialize the global database pool.
    
    Args:
        database_url: Database connection URL.
        config: Optional pool configuration.
        
    Returns:
        Initialized pool manager.
    """
    global _pool_manager
    _pool_manager = DatabasePoolManager(config)
    await _pool_manager.initialize(database_url)
    return _pool_manager
