"""
Product Repository Implementation

Concrete implementation of the Product repository
"""

from typing import List, Optional

from ..domain.interfaces.repository import Repository
from ..domain.entities.product import Product


class ProductRepository(Repository[Product]):
    """
    Concrete implementation of Product repository
    
    In-memory implementation for demonstration.
    In production, this would use SQLAlchemy or similar ORM.
    """
    
    def __init__(self):
        """Initialize repository"""
        self._products = {}
    
    async def get_by_id(self, id: str) -> Optional[Product]:
        """Get product by ID"""
        return self._products.get(id)
    
    async def get_all(self) -> List[Product]:
        """Get all products"""
        return list(self._products.values())
    
    async def add(self, entity: Product) -> Product:
        """Add new product"""
        self._products[entity.product_id] = entity
        return entity
    
    async def update(self, entity: Product) -> Product:
        """Update existing product"""
        if entity.product_id in self._products:
            self._products[entity.product_id] = entity
        return entity
    
    async def delete(self, id: str) -> bool:
        """Delete product by ID"""
        if id in self._products:
            del self._products[id]
            return True
        return False
    
    async def exists(self, id: str) -> bool:
        """Check if product exists"""
        return id in self._products
    
    async def count(self) -> int:
        """Count products"""
        return len(self._products)
    
    async def get_by_site_id(self, site_id: str) -> List[Product]:
        """
        Get all products for a site
        
        Args:
            site_id: Site ID
            
        Returns:
            List of products for the site
        """
        return [p for p in self._products.values() if p.site_id == site_id]
    
    async def get_indexed_products(self) -> List[Product]:
        """
        Get all indexed products
        
        Returns:
            List of indexed products
        """
        return [p for p in self._products.values() if p.is_indexed()]