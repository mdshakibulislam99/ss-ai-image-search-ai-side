"""
Admin API endpoints
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from ...domain.interfaces.vector_store import VectorStore
from ...domain.interfaces.repository import Repository
from ...domain.interfaces.cache import Cache
from ...domain.entities.product import Product
from ...container import container

router = APIRouter()


def get_vector_store() -> VectorStore:
    """Dependency injection for vector store"""
    return container.resolve(VectorStore)


def get_product_repository() -> Repository[Product]:
    """Dependency injection for product repository"""
    return container.resolve(Repository)


def get_cache() -> Cache:
    """Dependency injection for cache"""
    return container.resolve(Cache)


@router.get("/stats", summary="Get system statistics")
async def get_stats(
    product_repo: Repository[Product] = Depends(get_product_repository),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """Get system statistics"""
    try:
        total_products = await product_repo.count()
        vector_stats = vector_store.get_stats()
        
        return {
            "success": True,
            "stats": {
                "total_products": total_products,
                "total_embeddings": vector_stats.total_vectors,
                "vector_store_size_bytes": vector_stats.index_size,
                "dimensions": vector_stats.dimensions,
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}",
        )


@router.get("/queue", summary="Get queue status")
async def get_queue_status():
    """Get queue status and statistics"""
    try:
        from ....infrastructure.queue.memory_queue import MemoryQueue
        queue = MemoryQueue()
        stats = queue.get_queue_stats()
        
        return {
            "success": True,
            "queue": {
                "pending": stats.get("pending", 0),
                "processing": stats.get("processing", 0),
                "completed": stats.get("completed", 0),
                "failed": stats.get("failed", 0),
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue status: {str(e)}",
        )


@router.post("/queue/clear", summary="Clear queue")
async def clear_queue():
    """Clear all jobs from the queue"""
    try:
        from ....infrastructure.queue.memory_queue import MemoryQueue
        queue = MemoryQueue()
        cleared = queue.clear_queue()
        
        return {
            "success": True,
            "message": f"Cleared {cleared} jobs from queue",
            "cleared_count": cleared,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear queue: {str(e)}",
        )


@router.post("/vector-store/backup", summary="Backup vector store")
async def backup_vector_store(vector_store: VectorStore = Depends(get_vector_store)):
    """Backup vector store to disk"""
    try:
        from ....config.settings import settings
        import time
        
        backup_path = f"{settings.vector_store_path}.backup.{int(time.time())}"
        vector_store.save(backup_path)
        
        return {
            "success": True,
            "message": "Vector store backed up successfully",
            "backup_path": backup_path,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to backup vector store: {str(e)}",
        )


@router.post("/vector-store/restore", summary="Restore vector store")
async def restore_vector_store(vector_store: VectorStore = Depends(get_vector_store)):
    """Restore vector store from disk"""
    try:
        from ....config.settings import settings
        import glob
        
        # Find latest backup
        backups = sorted(glob.glob(f"{settings.vector_store_path}.backup.*"))
        if not backups:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No backup found",
            )
        
        latest_backup = backups[-1]
        vector_store.load(latest_backup)
        
        return {
            "success": True,
            "message": "Vector store restored successfully",
            "restored_from": latest_backup,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restore vector store: {str(e)}",
        )


@router.get("/models", summary="Get available AI models")
async def get_models():
    """Get list of available AI models"""
    from ...infrastructure.ai.provider_factory import AIProviderFactory
    
    try:
        models = AIProviderFactory.get_available_models()
        return {
            "success": True,
            "models": models,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models: {str(e)}",
        )


@router.post("/cache/clear", summary="Clear cache")
async def clear_cache(cache: Cache = Depends(get_cache)):
    """Clear all caches"""
    try:
        if hasattr(cache, 'clear'):
            cache.clear()
            return {"success": True, "message": "Cache cleared successfully"}
        else:
            return {"success": False, "message": "Cache clearing not supported"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}",
        )


@router.get("/vector-store/stats", summary="Get vector store statistics")
async def get_vector_store_stats(vector_store: VectorStore = Depends(get_vector_store)):
    """Get vector store statistics"""
    try:
        stats = vector_store.get_stats()
        health = vector_store.health_check()
        
        return {
            "success": True,
            "stats": {
                "total_vectors": stats.total_vectors,
                "dimensions": stats.dimensions,
                "index_size": stats.index_size,
                "healthy": health.get("healthy", False),
                "index_type": health.get("index_type", "unknown"),
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vector store statistics: {str(e)}",
        )
