"""
Admin API endpoints
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/stats", summary="Get system statistics")
async def get_stats():
    """Get system statistics"""
    return {
        "message": "System statistics endpoint - Implementation pending",
        "stats": {
            "total_products": 0,
            "total_embeddings": 0,
            "vector_store_size": 0,
        }
    }


@router.get("/models", summary="Get available AI models")
async def get_models():
    """Get list of available AI models"""
    return {
        "message": "Available models endpoint - Implementation pending",
        "models": []
    }


@router.post("/cache/clear", summary="Clear cache")
async def clear_cache():
    """Clear all caches"""
    return {"message": "Cache cleared successfully"}


@router.get("/vector-store/stats", summary="Get vector store statistics")
async def get_vector_store_stats():
    """Get vector store statistics"""
    return {
        "message": "Vector store statistics endpoint - Implementation pending",
        "stats": {
            "total_vectors": 0,
            "dimensions": 0,
            "index_size": 0,
        }
    }