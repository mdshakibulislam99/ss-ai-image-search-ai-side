"""
SearchResponse - Response DTO for image search
"""

from dataclasses import dataclass
from typing import Any,  Dict,  List, Optional

from ...domain.entities.search_result import SearchResult  # type: ignore
from ...domain.value_objects.embedding_vector import EmbeddingVector  # type: ignore


@dataclass
class SearchResponse:
    """Search response DTO"""
    
    success: bool
    results: List[SearchResult]
    query_embedding: Optional[EmbeddingVector] = None
    total_count: int = 0
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "results": [r.to_dict() for r in self.results],
            "total_count": self.total_count,
            "message": self.message,
            "metadata": self.metadata,
        }