"""
SS AI Server - Main Application Entry Point

Production-grade FastAPI application for image similarity search
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from .config.settings import Settings, settings
from .container import configure_services, container
from .domain.interfaces.logger import Logger
from .domain.interfaces.cache import Cache
from .infrastructure.logging.logger import LoggerImpl
from .exceptions.base_exceptions import SSAIServerException
from .presentation.api.v1.health import router as health_router
from .presentation.api.v1.search import router as search_router
from .presentation.api.v1.index import router as index_router
from .presentation.api.v1.admin import router as admin_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.log_format == "json" else structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    
    Handles startup and shutdown events
    """
    # Startup
    logger.info("Starting SS AI Server", version=settings.app_version, environment=settings.environment)
    
    try:
        # Configure dependency injection container
        configure_services(settings)
        logger.info("Dependency injection container configured")
        
        # Initialize vector store
        from .domain.interfaces.vector_store import VectorStore
        vector_store = container.resolve(VectorStore)
        vector_store.initialize(settings.vector_store_dimensions, settings.vector_store_metric)
        logger.info("Vector store initialized", dimensions=settings.vector_store_dimensions)
        
        # Initialize AI provider
        from .domain.interfaces.ai_provider import AIProvider
        from .infrastructure.ai.provider_factory import AIProviderFactory
        ai_provider = AIProviderFactory.create_provider(settings.ai_default_provider)
        logger.info("AI provider initialized", provider=settings.ai_default_provider)
        
        # Initialize indexing service
        from .domain.services.indexing_service import IndexingService
        from .domain.interfaces.repository import Repository
        from .domain.entities.product import Product
        
        product_repository = container.resolve(Repository)
        indexing_service = IndexingService(
            ai_provider=ai_provider,
            vector_store=vector_store,
            product_repository=product_repository,
            logger=container.resolve(Logger),
        )
        logger.info("Indexing service initialized")
        
        # Initialize search service
        from .domain.services.search_service import SearchService
        from .domain.interfaces.cache import Cache
        
        cache = container.resolve(Cache)
        search_service = SearchService(
            ai_provider=ai_provider,
            vector_store=vector_store,
            product_repository=product_repository,
            cache=cache,
            logger=container.resolve(Logger),
        )
        logger.info("Search service initialized")
        
        # Store initialized services in app state
        app.state.vector_store = vector_store
        app.state.ai_provider = ai_provider
        app.state.indexing_service = indexing_service
        app.state.product_repository = product_repository
        app.state.search_service = search_service
        app.state.cache = cache
        
        logger.info("SS AI Server started successfully")
        
        yield
        
        # Shutdown
        logger.info("Shutting down SS AI Server")
        
        # Save vector store if persistent
        try:
            if hasattr(vector_store, 'save'):
                vector_store.save(settings.vector_store_path)
                logger.info("Vector store saved")
        except Exception as e:
            logger.warning("Failed to save vector store", error=str(e))
        
        logger.info("SS AI Server shutdown complete")
        
    except Exception as e:
        logger.error("Failed to start SS AI Server", error=str(e))
        raise


def create_application() -> FastAPI:
    """
    Create and configure FastAPI application
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade AI backend for WordPress image similarity search",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        lifespan=lifespan,
    )
    
    # Configure CORS
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("CORS middleware configured", origins=settings.cors_origins)
    
    # Add rate limiting middleware
    if settings.rate_limit_enabled:
        from .presentation.middleware.rate_limit_middleware import RateLimitMiddleware
        cache = container.resolve(Cache)
        app.add_middleware(
            RateLimitMiddleware,
            settings=settings,
            cache=cache,
        )
        logger.info("Rate limiting middleware configured")
    
    # Add authentication middleware
    from .presentation.middleware.auth_middleware import APIKeyAuthMiddleware
    cache = container.resolve(Cache)
    app.add_middleware(
        APIKeyAuthMiddleware,
        settings=settings,
        cache=cache,
    )
    logger.info("Authentication middleware configured")
    
    # Register routers
    app.include_router(health_router, prefix="/api/v1/health", tags=["health"])
    app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
    app.include_router(index_router, prefix="/api/v1/index", tags=["index"])
    app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])
    
    # Global exception handler
    @app.exception_handler(SSAIServerException)
    async def ssai_exception_handler(request: Any, exc: SSAIServerException):
        """Handle custom SS AI Server exceptions"""
        logger.error("SSAI exception", error=exc.message, details=exc.details)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.message,
                "details": exc.details,
            },
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Any, exc: Exception):
        """Handle unhandled exceptions"""
        logger.error("Unhandled exception", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error",
                "message": str(exc) if settings.is_development else "An unexpected error occurred",
            },
        )
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint"""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "status": "running",
            "docs": "/docs" if settings.is_development else "disabled",
        }
    
    return app


# Create application instance
app = create_application()


def run():
    """Run the application with uvicorn"""
    import uvicorn
    
    uvicorn.run(
        "ss_ai_server.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        reload=settings.is_development,
        access_log=True,
    )


if __name__ == "__main__":
    run()