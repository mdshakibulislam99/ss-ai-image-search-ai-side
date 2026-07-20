"""
Search API endpoints for similar image search
"""

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field, validator

from ...application.use_cases.search_image import SearchImageUseCase
from ...application.dto.requests.search_request import SearchRequest
from ...application.dto.responses.search_response import SearchResponse, SearchResultItem

router = APIRouter()


# API request/response models
class SearchImageRequest(BaseModel):
    """API request model for image search"""
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    
    @validator("threshold")
    def validate_threshold(cls, v: float) -> float:
        """Validate threshold"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        return v


class SearchImageResponse(BaseModel):
    """API response model for image search"""
    success: bool
    results: List[SearchResultItem]
    total_count: int = 0
    processing_time_ms: float = 0.0
    message: Optional[str] = None


def get_search_use_case() -> SearchImageUseCase:
    """Dependency injection for search use case."""
    from ....container import container
    return container.resolve(SearchImageUseCase)


@router.post(
    "/image",
    response_model=SearchImageResponse,
    summary="Search for similar products by image",
    description="Upload an image to find visually similar WooCommerce products. "
                "Supports JPG, PNG, and WEBP formats.",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Search completed successfully"},
        400: {"description": "Invalid request or image"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def search_by_image(
    file: UploadFile = File(..., description="Image file to search (JPG, PNG, WEBP)"),
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results"),
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    use_case: SearchImageUseCase = Depends(get_search_use_case),
) -> SearchImageResponse:
    """
    Search for similar products by uploading an image.
    
    The endpoint:
    1. Validates the uploaded image file
    2. Generates an embedding using the configured AI provider
    3. Searches the vector store for nearest neighbors
    4. Enriches results with product metadata
    5. Returns sorted results with similarity scores
    
    Supported formats: JPG, PNG, WEBP
    Maximum file size: 10MB (configurable)
    """
    start_time = time.time()
    
    # Validate file presence
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided",
        )
    
    # Validate file extension
    if file.filename:
        ext = file.filename.lower().split(".")[-1] if "." in file.filename else ""
        if ext not in ("jpg", "jpeg", "png", "webp"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: .{ext}. Supported formats: jpg, jpeg, png, webp",
            )
    
    # Read file content
    try:
        image_data = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {e}",
        )
    
    # Validate file size
    if len(image_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    
    if len(image_data) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size ({len(image_data)} bytes) exceeds maximum allowed size (10MB)",
        )
    
    # Validate content type
    if file.content_type and file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type: {file.content_type}. Supported: image/jpeg, image/png, image/webp",
        )
    
    # Create search request DTO
    search_request = SearchRequest(
        image_data=image_data,
        original_filename=file.filename,
        content_type=file.content_type,
        limit=limit,
        threshold=threshold,
    )
    
    # Execute use case
    try:
        result = await use_case.execute(search_request)
        
        # Convert to API response
        return SearchImageResponse(
            success=result.success,
            results=result.results,
            total_count=result.total_count,
            processing_time_ms=result.processing_time_ms,
            message=result.message,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal search error: {str(e)}",
        )


@router.get(
    "/health",
    summary="Search service health",
    description="Check if the search service is healthy and operational.",
    status_code=status.HTTP_200_OK,
)
async def search_health():
    """Check if search service is healthy"""
    return {
        "status": "healthy",
        "service": "similar-image-search",
        "version": "1.0.0",
    }