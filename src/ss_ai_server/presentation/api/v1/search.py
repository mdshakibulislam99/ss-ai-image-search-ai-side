"""
Search API endpoints
"""

from fastapi import APIRouter, HTTPException, status

from ...dtos.requests.search_request import SearchRequest
from ...dtos.responses.search_response import SearchResponse
from ...use_cases.search_image import SearchImageUseCase

router = APIRouter()


@router.post("/image", response_model=SearchResponse, summary="Search by image")
async def search_by_image(request: SearchRequest):
    """
    Search for similar images by uploading an image
    
    - **image_data**: Binary image data
    - **limit**: Maximum number of results (1-100)
    - **threshold**: Minimum similarity threshold (0.0-1.0)
    """
    try:
        use_case = SearchImageUseCase()
        result = await use_case.execute(request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health", summary="Search service health")
async def search_health():
    """Check if search service is healthy"""
    return {"status": "healthy", "service": "search"}