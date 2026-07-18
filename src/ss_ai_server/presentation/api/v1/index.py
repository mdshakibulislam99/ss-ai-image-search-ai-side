"""
Index API endpoints
"""

from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.post("/product", summary="Index a single product")
async def index_product():
    """Index a single product for search"""
    return {"message": "Product indexing endpoint - Implementation pending"}


@router.post("/batch", summary="Batch index products")
async def batch_index():
    """Batch index multiple products"""
    return {"message": "Batch indexing endpoint - Implementation pending"}


@router.delete("/product/{product_id}", summary="Delete product embedding")
async def delete_product(product_id: str):
    """Delete a product's embedding from the index"""
    return {"message": f"Product {product_id} deleted from index"}


@router.post("/refresh", summary="Refresh index")
async def refresh_index():
    """Refresh the entire index"""
    return {"message": "Index refresh initiated"}