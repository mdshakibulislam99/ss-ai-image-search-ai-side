"""
FAISS Vector Store Implementation

Provides efficient similarity search using Facebook AI Similarity Search (FAISS)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base_vector_store import BaseVectorStore
from ...domain.interfaces.vector_store import VectorStoreStats
from ...domain.value_objects.embedding_vector import EmbeddingVector
from ...domain.entities.search_result import SearchResult
from ...domain.interfaces.logger import Logger
from ...config.settings import Settings
from ...exceptions.infrastructure_exceptions import VectorStoreError


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store implementation
    
    Features:
    - Efficient similarity search
    - Multiple index types (IndexFlat, IndexIVFFlat, etc.)
    - Persistence to disk
    - Metadata support
    - Health check capabilities
    """
    
    # Supported FAISS index types
    INDEX_TYPES = {
        "flat": "IndexFlat",
        "ivf_flat": "IndexIVFFlat",
        "hnsw": "IndexHNSW",
        "ivf_hnsw": "IndexIVFHNSW",
    }
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        logger: Optional[Logger] = None,
        index_type: str = "flat",
        index_path: Optional[str] = None,
        nlist: int = 100,
        nprobe: int = 10
    ) -> None:
        """
        Initialize FAISS vector store
        
        Args:
            settings: Application settings instance
            logger: Logger instance
            index_type: FAISS index type (flat, ivf_flat, hnsw, ivf_hnsw)
            index_path: Path for index persistence
            nlist: Number of clusters for IVF indexes
            nprobe: Number of clusters to search for IVF indexes
        """
        super().__init__()
        
        self._settings = settings or Settings()
        self._logger = logger
        self._index_type = index_type
        self._index_path = Path(index_path or self._settings.vector_store_path)
        self._nlist = nlist
        self._nprobe = nprobe
        
        # FAISS index
        self._index = None
        
        # Vector ID mapping (product_id -> faiss_id)
        self._id_map: Dict[str, int] = {}
        
        # Reverse mapping (faiss_id -> product_id)
        self._reverse_id_map: Dict[int, str] = {}
        
        # Metadata storage
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
        # Current FAISS ID counter
        self._current_id = 0
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate vector store configuration"""
        if self._index_type not in self.INDEX_TYPES:
            raise VectorStoreError(
                "faiss",
                "initialize",
                f"Invalid index type '{self._index_type}'. "
                f"Supported types: {list(self.INDEX_TYPES.keys())}"
            )
    
    def _log_debug(self, message: str) -> None:
        """Log debug message"""
        if self._logger:
            self._logger.debug(message)
    
    def _log_info(self, message: str) -> None:
        """Log info message"""
        if self._logger:
            self._logger.info(message)
    
    def _log_warning(self, message: str) -> None:
        """Log warning message"""
        if self._logger:
            self._logger.warning(message)
    
    def _log_error(self, message: str) -> None:
        """Log error message"""
        if self._logger:
            self._logger.error(message)
    
    def initialize(self, dimensions: int, metric: str = "cosine") -> None:
        """
        Initialize FAISS index
        
        Args:
            dimensions: Vector dimensions
            metric: Distance metric (cosine, l2, inner_product)
        """
        try:
            import faiss
            
            self._dimensions = dimensions
            self._metric = metric
            
            # Map metric to FAISS metric
            faiss_metric = self._get_faiss_metric(metric)
            
            # Create FAISS index based on type
            if self._index_type == "flat":
                self._index = faiss.IndexFlat(dimensions, faiss_metric)
            elif self._index_type == "ivf_flat":
                quantizer = faiss.IndexFlat(dimensions, faiss_metric)
                self._index = faiss.IndexIVFFlat(quantizer, dimensions, self._nlist, faiss_metric)
            elif self._index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(dimensions, 32, faiss_metric)
            elif self._index_type == "ivf_hnsw":
                quantizer = faiss.IndexHNSWFlat(dimensions, 32, faiss_metric)
                self._index = faiss.IndexIVFFlat(quantizer, dimensions, self._nlist, faiss_metric)
            
            # Set nprobe for IVF indexes
            if self._index_type in ["ivf_flat", "ivf_hnsw"]:
                self._index.nprobe = self._nprobe
            
            self._initialized = True
            self._log_info(f"FAISS index initialized: type={self._index_type}, dimensions={dimensions}, metric={metric}")
            
        except ImportError as e:
            raise VectorStoreError(
                "faiss",
                "initialize",
                f"FAISS library not installed: {e}. Install with: pip install faiss-cpu"
            )
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "initialize",
                f"Failed to initialize FAISS index: {str(e)}"
            )
    
    def _get_faiss_metric(self, metric: str) -> int:
        """
        Convert metric name to FAISS metric constant
        
        Args:
            metric: Metric name (cosine, l2, inner_product)
            
        Returns:
            FAISS metric constant
        """
        import faiss
        
        metric_map = {
            "l2": faiss.METRIC_L2,
            "cosine": faiss.METRIC_INNER_PRODUCT,
            "inner_product": faiss.METRIC_INNER_PRODUCT,
        }
        
        return metric_map.get(metric, faiss.METRIC_INNER_PRODUCT)
    
    def add_vectors(self, vectors: List[Tuple[str, EmbeddingVector, Dict[str, Any]]]) -> None:
        """
        Add vectors to FAISS index
        
        Args:
            vectors: List of (product_id, embedding, metadata) tuples
        """
        if not self._initialized:
            raise VectorStoreError(
                "faiss",
                "add_vectors",
                "Vector store not initialized. Call initialize() first."
            )
        
        if not vectors:
            return
        
        try:
            # Prepare vectors for FAISS
            vector_ids = []
            vector_data = []
            
            for product_id, embedding, metadata in vectors:
                # Convert embedding to numpy array
                vector = np.array(embedding.vector, dtype=np.float32)
                
                # Normalize for cosine similarity
                if self._metric == "cosine":
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
                
                vector_data.append(vector)
                vector_ids.append(self._current_id)
                
                # Update mappings
                self._id_map[product_id] = self._current_id
                self._reverse_id_map[self._current_id] = product_id
                self._metadata[product_id] = metadata or {}
                
                self._current_id += 1
            
            # Stack vectors
            vectors_array = np.stack(vector_data)
            
            # Add to FAISS index
            self._index.add(vectors_array)
            
            self._log_info(f"Added {len(vectors)} vectors to FAISS index")
            
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "add_vectors",
                f"Failed to add vectors: {str(e)}"
            )
    
    def search(
        self,
        query_vector: EmbeddingVector,
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding
            limit: Maximum results
            filter_dict: Metadata filters
            
        Returns:
            List of search results sorted by similarity
        """
        if not self._initialized:
            raise VectorStoreError(
                "faiss",
                "search",
                "Vector store not initialized. Call initialize() first."
            )
        
        if self._index.ntotal == 0:
            return []
        
        try:
            # Convert query to numpy array
            query = np.array(query_vector.vector, dtype=np.float32)
            
            # Normalize for cosine similarity
            if self._metric == "cosine":
                norm = np.linalg.norm(query)
                if norm > 0:
                    query = query / norm
            
            # Reshape for FAISS
            query = query.reshape(1, -1)
            
            # Search
            distances, indices = self._index.search(query, min(limit, self._index.ntotal))
            
            # Convert to search results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue
                
                product_id = self._reverse_id_map.get(idx)
                if product_id is None:
                    continue
                
                # Apply filter if provided
                if filter_dict:
                    metadata = self._metadata.get(product_id, {})
                    if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                        continue
                
                # Convert distance to similarity score
                if self._metric == "cosine":
                    similarity = float(dist)
                else:
                    similarity = 1.0 / (1.0 + float(dist))
                
                result = SearchResult(
                    product_id=product_id,
                    similarity=similarity,
                    metadata=self._metadata.get(product_id, {})
                )
                results.append(result)
            
            self._log_debug(f"Search returned {len(results)} results")
            
            return results
            
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "search",
                f"Failed to search vectors: {str(e)}"
            )
    
    def delete_vectors(self, product_ids: List[str]) -> int:
        """
        Delete vectors by product IDs
        
        Args:
            product_ids: List of product IDs
            
        Returns:
            Number of vectors deleted
        """
        if not self._initialized:
            raise VectorStoreError(
                "faiss",
                "delete_vectors",
                "Vector store not initialized. Call initialize() first."
            )
        
        if not product_ids:
            return 0
        
        try:
            deleted_count = 0
            
            for product_id in product_ids:
                if product_id in self._id_map:
                    faiss_id = self._id_map.pop(product_id)
                    self._reverse_id_map.pop(faiss_id, None)
                    self._metadata.pop(product_id, None)
                    deleted_count += 1
            
            self._log_info(f"Deleted {deleted_count} vectors from FAISS index")
            
            return deleted_count
            
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "delete_vectors",
                f"Failed to delete vectors: {str(e)}"
            )
    
    def get_vector(self, product_id: str) -> Optional[Tuple[EmbeddingVector, Dict[str, Any]]]:
        """
        Get single vector by product ID
        
        Args:
            product_id: Product ID
            
        Returns:
            Tuple of (embedding, metadata) or None
        """
        if not self._initialized:
            raise VectorStoreError(
                "faiss",
                "get_vector",
                "Vector store not initialized. Call initialize() first."
            )
        
        if product_id not in self._id_map:
            return None
        
        try:
            faiss_id = self._id_map[product_id]
            
            if hasattr(self._index, 'reconstruct'):
                vector = self._index.reconstruct(faiss_id)
            else:
                self._log_warning("Vector reconstruction not supported for this index type")
                return None
            
            embedding = EmbeddingVector(
                vector=vector.tolist(),
                model="faiss",
                provider="faiss",
                dimensions=len(vector)
            )
            
            return embedding, self._metadata.get(product_id, {})
            
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "get_vector",
                f"Failed to get vector: {str(e)}"
            )
    
    def get_stats(self) -> VectorStoreStats:
        """
        Get vector store statistics
        
        Returns:
            Vector store statistics
        """
        if not self._initialized:
            raise VectorStoreError(
                "faiss",
                "get_stats",
                "Vector store not initialized. Call initialize() first."
            )
        
        try:
            return VectorStoreStats(
                total_vectors=self._index.ntotal,
                dimensions=self._dimensions,
                index_size=self._index.ntotal * self._dimensions * 4
            )
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "get_stats",
                f"Failed to get stats: {str(e)}"
            )
    
    def save(self, path: str) -> None:
        """
        Persist FAISS index to disk
        
        Args:
            path: Path to save index
        """
        if not self._initialized:
            raise VectorStoreError(
                "faiss",
                "save",
                "Vector store not initialized. Call initialize() first."
            )
        
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            import faiss
            faiss.write_index(self._index, str(save_path))
            
            import json
            metadata_path = save_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    "id_map": self._id_map,
                    "metadata": self._metadata,
                    "current_id": self._current_id,
                    "dimensions": self._dimensions,
                    "metric": self._metric,
                    "index_type": self._index_type
                }, f)
            
            self._log_info(f"FAISS index saved to {path}")
            
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "save",
                f"Failed to save index: {str(e)}"
            )
    
    def load(self, path: str) -> None:
        """
        Load FAISS index from disk
        
        Args:
            path: Path to load index from
        """
        try:
            import faiss
            import json
            
            load_path = Path(path)
            
            if not load_path.exists():
                raise VectorStoreError(
                    "faiss",
                    "load",
                    f"Index file not found: {path}"
                )
            
            self._index = faiss.read_index(str(load_path))
            
            metadata_path = load_path.with_suffix('.metadata.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self._id_map = data.get("id_map", {})
                    self._metadata = data.get("metadata", {})
                    self._current_id = data.get("current_id", 0)
                    self._dimensions = data.get("dimensions")
                    self._metric = data.get("metric")
                    self._index_type = data.get("index_type", "flat")
            
            self._reverse_id_map = {v: k for k, v in self._id_map.items()}
            self._initialized = True
            
            self._log_info(f"FAISS index loaded from {path}")
            
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "load",
                f"Failed to load index: {str(e)}"
            )
    
    def clear(self) -> None:
        """Clear all vectors from the index"""
        if not self._initialized:
            raise VectorStoreError(
                "faiss",
                "clear",
                "Vector store not initialized. Call initialize() first."
            )
        
        try:
            import faiss
            
            faiss_metric = self._get_faiss_metric(self._metric)
            
            if self._index_type == "flat":
                self._index = faiss.IndexFlat(self._dimensions, faiss_metric)
            elif self._index_type == "ivf_flat":
                quantizer = faiss.IndexFlat(self._dimensions, faiss_metric)
                self._index = faiss.IndexIVFFlat(quantizer, self._dimensions, self._nlist, faiss_metric)
            elif self._index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(self._dimensions, 32, faiss_metric)
            elif self._index_type == "ivf_hnsw":
                quantizer = faiss.IndexHNSWFlat(self._dimensions, 32, faiss_metric)
                self._index = faiss.IndexIVFFlat(quantizer, self._dimensions, self._nlist, faiss_metric)
            
            if self._index_type in ["ivf_flat", "ivf_hnsw"]:
                self._index.nprobe = self._nprobe
            
            self._id_map.clear()
            self._reverse_id_map.clear()
            self._metadata.clear()
            self._current_id = 0
            
            self._log_info("FAISS index cleared")
            
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "clear",
                f"Failed to clear index: {str(e)}"
            )
    
    def get_all_vectors(self) -> List[Tuple[str, EmbeddingVector]]:
        """
        Get all vectors (for backup/migration)
        
        Returns:
            List of (product_id, embedding) tuples
        """
        if not self._initialized:
            raise VectorStoreError(
                "faiss",
                "get_all_vectors",
                "Vector store not initialized. Call initialize() first."
            )
        
        try:
            results = []
            
            for product_id, faiss_id in self._id_map.items():
                if hasattr(self._index, 'reconstruct'):
                    vector = self._index.reconstruct(faiss_id)
                    embedding = EmbeddingVector(
                        vector=vector.tolist(),
                        model="faiss",
                        provider="faiss",
                        dimensions=len(vector)
                    )
                    results.append((product_id, embedding))
            
            return results
            
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "get_all_vectors",
                f"Failed to get all vectors: {str(e)}"
            )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector store
        
        Returns:
            Health check result dictionary
        """
        health = {
            "healthy": True,
            "index_type": self._index_type,
            "initialized": self._initialized,
            "total_vectors": 0,
            "dimensions": self._dimensions,
            "errors": []
        }
        
        if not self._initialized:
            health["healthy"] = False
            health["errors"].append("Vector store not initialized")
            return health
        
        try:
            health["total_vectors"] = self._index.ntotal
            
            if self._index.ntotal < 0:
                health["healthy"] = False
                health["errors"].append("Invalid index state")
            
        except Exception as e:
            health["healthy"] = False
            health["errors"].append(str(e))
        
        return health
    
    def update_vector(
        self,
        product_id: str,
        embedding: EmbeddingVector,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing vector
        
        Note: FAISS doesn't support direct updates, so we delete and re-add
        
        Args:
            product_id: Product ID
            embedding: New embedding vector
            metadata: New metadata
            
        Returns:
            True if updated successfully
        """
        if not self._initialized:
            raise VectorStoreError(
                "faiss",
                "update_vector",
                "Vector store not initialized. Call initialize() first."
            )
        
        if product_id not in self._id_map:
            return False
        
        try:
            self.delete_vectors([product_id])
            self.add_vectors([(product_id, embedding, metadata or {})])
            
            self._log_info(f"Updated vector for product {product_id}")
            
            return True
            
        except Exception as e:
            raise VectorStoreError(
                "faiss",
                "update_vector",
                f"Failed to update vector: {str(e)}"
            )
