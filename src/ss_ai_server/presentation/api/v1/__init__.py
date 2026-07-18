"""
API v1 - Version 1 of the REST API
"""

from fastapi import APIRouter

from . import search, index, health, admin

# Create main API v1 router
api_v1_router = APIRouter()

# Include all endpoint routers
api_v1_router.include_router(search.router, prefix="/search", tags=["search"])
api_v1_router.include_router(index.router, prefix="/index", tags=["index"])
api_v1_router.include_router(health.router, prefix="/health", tags=["health"])
api_v1_router.include_router(admin.router, prefix="/admin", tags=["admin"])

__all__ = ["api_v1_router"]