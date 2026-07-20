"""
Index API endpoints for product indexing
"""

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator

from ...application.use_cases.index_product import IndexProductUseCase, BatchIndexUseCase
from ...application.dto.requests.index_request import IndexRequest, BatchIndexRequest
from ...application.dto.responses.index_response import IndexResponse, BatchIndexResponse
from ...domain.interfaces.vector_store import VectorStore

router = APIRouter()


# Pydantic models for API request/response validation
class ImageMetadata(BaseModel):
    """Image metadata model"""
    url: str = Field(..., description="Image URL")
    alt: Optional[str] = None


class IndexProductRequest(BaseModel):
    """API request model for indexing a product"""
    product_id: str = Field(..., min_length=1, description="Unique product identifier")
    site_id: str = Field(..., min_length=1, description="Site/tenant identifier")
    title: Optional[str] = Field(None, description="Product title")
    sku: Optional[str] = Field(None, description="Product SKU")
    description: Optional[str] = Field(None, description="Product description")
    categories: List[str] = Field(default_factory=list, description="Product categories")
    attributes: dict = Field(default_factory=dict, description="Product attributes")
    featured_image_url: Optional[str] = Field(None, description="Featured image URL")
    gallery_image_urls: List[str] = Field(default_factory=list, description="Gallery image URLs")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    reindex: bool = Field(default=False, description="Re-index existing product")
    
    @validator("featured_image_url")
    def validate_featured_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate featured image URL"""
        if v is not None:
            if not v.startswith(("http://", "https://")):
                raise ValueError("featured_image_url must start with http:// or https://")
        return v
    
    @validator("gallery_image_urls", each_item=True)
    def validate_gallery_urls(cls, v: str) -> str:
        """Validate gallery image URLs"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Gallery image URLs must start with http:// or https://")
        return v


class IndexProductResponse(BaseModel):
    """API response model for indexing"""
    success: bool
    product_id: str
    images_indexed: int = 0
    processing_time_ms: float = 0.0
    message: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class BatchIndexRequest(BaseModel):
    """API request model for batch indexing"""
    products: List[IndexProductRequest] = Field(..., min_length=1, max_length=100)


class BatchIndexResponse(BaseModel):
    """API response model for batch indexing"""
    success: bool
    total_products: int = 0
    indexed_count: int = 0
    failed_count: int = 0
    results: List[IndexProductResponse] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    message: Optional[str] = None


def get_index_use_case() -> IndexProductUseCase:
    """Dependency injection for index product use case."""
    from ....container import container
    return container.resolve(IndexProductUseCase)


def get_batch_index_use_case() -> BatchIndexUseCase:
    """Dependency injection for batch index use case."""
    from ....container import container
    return container.resolve(BatchIndexUseCase)


@router.post(
    "/product",
    response_model=IndexProductResponse,
    summary="Index a single product",
    description="Index a WooCommerce product for visual similarity search. "
                "Downloads product images, generates embeddings, and stores them.",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Product indexed successfully"},
        400: {"description": "Invalid request data"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def index_product(
    request: IndexProductRequest,
    use_case: IndexProductUseCase = Depends(get_index_use_case),
) -> IndexProductResponse:
    """
    Index a single product for visual similarity search.
    
    The endpoint:
    1. Validates the request data
    2. Downloads product images
    3. Generates image embeddings using the configured AI provider
    4. Stores embeddings in the vector store
    5. Persists product metadata
    6. Returns indexing results
    """
    try:
        # Convert API request to use case DTO
        index_request = IndexRequest(
            product_id=request.product_id,
            site_id=request.site_id,
            title=request.title,
            sku=request.sku,
            description=request.description,
            categories=request.categories,
            attributes=dict(request.attributes),
            featured_image_url=request.featured_image_url,
            gallery_image_urls=request.gallery_image_urls,
            metadata=dict(request.metadata),
            reindex=request.reindex,
        )
        
        # Execute use case
        result = await use_case.execute(index_request)
        
        # Convert to API response
        return IndexProductResponse(
            success=result.success,
            product_id=result.product_id,
            images_indexed=result.images_indexed,
            processing_time_ms=result.processing_time_ms,
            message=result.message,
            errors=result.errors,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=BatchIndexResponse,
    summary="Batch index products",
    description="Index multiple products in a single request.",
    status_code=status.HTTP_200_OK,
)
async def batch_index(
    request: BatchIndexRequest,
    use_case: BatchIndexUseCase = Depends(get_batch_index_use_case),
) -> BatchIndexResponse:
    """
    Batch index multiple products.
    
    Processes up to 100 products per request.
    Each product is indexed independently; failures in one don't affect others.
    """
    try:
        # Convert API request to use case DTO
        products = []
        for p in request.products:
            products.append(IndexRequest(
                product_id=p.product_id,
                site_id=p.site_id,
                title=p.title,
                sku=p.sku,
                description=p.description,
                categories=p.categories,
                attributes=dict(p.attributes),
                featured_image_url=p.featured_image_url,
                gallery_image_urls=p.gallery_image_urls,
                metadata=dict(p.metadata),
                reindex=p.reindex,
            ))
        
        batch_request = BatchIndexRequest(products=products)
        
        # Execute use case
        result = await use_case.execute(batch_request)
        
        # Convert to API response
        return BatchIndexResponse(
            success=result.success,
            total_products=result.total_products,
            indexed_count=result.indexed_count,
            failed_count=result.failed_count,
            results=[
                IndexProductResponse(
                    success=r.success,
                    product_id=r.product_id,
                    images_indexed=r.images_indexed,
                    processing_time_ms=r.processing_time_ms,
                    message=r.message,
                    errors=r.errors,
                )
                for r in result.results
            ],
            processing_time_ms=result.processing_time_ms,
            message=result.message,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch indexing failed: {str(e)}",
        )


@router.delete(
    "/product/{product_id}",
    summary="Delete product embeddings",
    description="Delete a product's embeddings from the vector store.",
    status_code=status.HTTP_200_OK,
)
async def delete_product_embedding(product_id: str):
    """
    Delete a product's embeddings from the index.
    """
    from ....container import container
    try:
        vector_store = container.resolve(VectorStore)
        deleted = vector_store.delete_vectors([product_id])
        return {
            "success": True,
            "product_id": product_id,
            "deleted": deleted,
            "message": f"Deleted {deleted} embeddings for product {product_id}",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete product embeddings: {str(e)}",
        )
