"""
Health check API endpoints
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime

router = APIRouter()


@router.get("/", summary="Overall health check")
async def health_check():
    """
    Overall health check endpoint
    
    Returns the health status of the entire system
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "service": "ss-ai-server"
    }


@router.get("/ready", summary="Readiness probe")
async def readiness_check():
    """
    Readiness probe for Kubernetes
    
    Checks if the service is ready to accept requests
    """
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/live", summary="Liveness probe")
async def liveness_check():
    """
    Liveness probe for Kubernetes
    
    Checks if the service is alive
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }