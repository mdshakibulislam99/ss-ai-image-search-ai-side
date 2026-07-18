"""
SearchImageUseCase - Use case for searching similar images
"""

from typing import Optional

from ..interfaces.use_case import UseCase  # type: ignore
from ..dto.requests.search_request import SearchRequest
from ..dto.responses.search_response import SearchResponse
from ...domain.services.embedding_service import EmbeddingService
from ...domain.services.search_service import SearchService
from ...domain.value_objects.embedding_vector import EmbeddingVector  # type: ignore


class SearchImageUseCase(UseCase):
    """Use case for searching similar images"""
    
    def __init__(self, embedding_service: EmbeddingService, search_service: SearchService) -> None:
        """
        Initialize search image use case
        
        Args:
            embedding_service: Service for generating embeddings
            search_service: Service for searching
        """
        self.embedding_service = embedding_service
        self.search_service = search_service
    
    async def execute(self, request: SearchRequest) -> SearchResponse:
        """
        Execute search use case
        
        Args:
            request: Search request
            
        Returns:
            Search response
        """
        # Generate embedding for query image
        query_embedding = await self.embedding_service.generate_embedding(
            image_data=request.image_data,
            cache_key=request.cache_key
        )
        
        # Search for similar images
        results = await self.search_service.search_similar(
            query_embedding=query_embedding,
            limit=request.limit,
            threshold=request.threshold,
            cache_key=request.cache_key
        )
        
        # Create response
        return SearchResponse(
            success=True,
            results=results,
            query_embedding=query_embedding,
            total_count=len(results)
        )