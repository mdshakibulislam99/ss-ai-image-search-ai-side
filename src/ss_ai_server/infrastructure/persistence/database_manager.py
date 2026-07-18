"""
Database Manager - Database connection and session management
"""

from typing import Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from ...config.settings import settings


class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self) -> None:
        """Initialize database manager"""
        self._engine = None
        self._session_factory = None
    
    async def initialize(self) -> None:
        """Initialize database connection"""
        self._engine = create_async_engine(
            settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            echo=settings.debug
        )
        
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self._session_factory:
            await self.initialize()
        
        return self._session_factory()
    
    async def close(self) -> None:
        """Close database connection"""
        if self._engine:
            await self._engine.dispose()
    
    async def create_tables(self) -> None:
        """Create all tables"""
        from ...domain.entities.product import Product  # type: ignore
        from ...domain.entities.embedding import Embedding  # type: ignore
        from ...domain.entities.api_key import ApiKey  # type: ignore
        from ...domain.entities.queue_job import QueueJob  # type: ignore
        from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, Text, JSON
        from sqlalchemy.ext.declarative import declarative_base
        
        Base = declarative_base()
        
        # Define tables
        class ProductTable(Base):
            __tablename__ = "products"
            
            product_id = Column(String(255), primary_key=True)
            site_id = Column(String(255), nullable=False, index=True)
            title = Column(Text)
            description = Column(Text)
            image_urls = Column(JSON)
            metadata = Column(JSON)
            embedding_id = Column(String(255))
            indexed_at = Column(DateTime)
            created_at = Column(DateTime)
            updated_at = Column(DateTime)
        
        class ApiKeyTable(Base):
            __tablename__ = "api_keys"
            
            key_id = Column(String(255), primary_key=True)
            key_hash = Column(String(255), nullable=False, index=True)
            site_id = Column(String(255), nullable=False, index=True)
            name = Column(String(255))
            permissions = Column(JSON)
            rate_limit = Column(Integer, default=60)
            ip_whitelist = Column(JSON)
            is_active = Column(Boolean, default=True)
            created_at = Column(DateTime)
            expires_at = Column(DateTime)
            last_used_at = Column(DateTime)
        
        class QueueJobTable(Base):
            __tablename__ = "queue_jobs"
            
            job_id = Column(String(255), primary_key=True)
            job_type = Column(String(100), nullable=False)
            status = Column(String(50), nullable=False, index=True)
            priority = Column(Integer, default=0)
            retry_count = Column(Integer, default=0)
            max_retries = Column(Integer, default=3)
            payload = Column(JSON)
            result = Column(JSON)
            error = Column(Text)
            created_at = Column(DateTime, index=True)
            started_at = Column(DateTime)
            completed_at = Column(DateTime)
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)