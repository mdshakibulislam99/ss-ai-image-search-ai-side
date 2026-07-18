"""
OpenCLIP Vision AI Provider

Implements image embedding generation using OpenCLIP library
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from .base_provider import BaseAIProvider
from ...domain.interfaces.ai_provider import ModelInfo
from ...domain.value_objects.embedding_vector import EmbeddingVector
from ...domain.interfaces.logger import Logger
from ...config.settings import Settings
from ...exceptions.infrastructure_exceptions import (
    ModelLoadException,
    InferenceException,
    ConfigurationException
)


class OpenCLIPProvider(BaseAIProvider):
    """
    OpenCLIP-based vision AI provider for image embeddings
    
    Features:
    - Automatic model download on first run
    - Model caching to disk
    - CPU/GPU auto-detection
    - Normalized embedding vectors
    - Batch processing support
    """
    
    # Supported OpenCLIP models with their dimensions
    SUPPORTED_MODELS = [
        ModelInfo(
            name="ViT-B-32",
            provider="openclip",
            dimensions=512,
            version="openai"
        ),
        ModelInfo(
            name="ViT-B-16",
            provider="openclip",
            dimensions=512,
            version="openai"
        ),
        ModelInfo(
            name="ViT-L-14",
            provider="openclip",
            dimensions=768,
            version="openai"
        ),
        ModelInfo(
            name="ViT-L-14-336",
            provider="openclip",
            dimensions=768,
            version="openai"
        ),
        ModelInfo(
            name="ViT-H-14",
            provider="openclip",
            dimensions=1024,
            version="openai"
        ),
        ModelInfo(
            name="ViT-g-14",
            provider="openclip",
            dimensions=1024,
            version="openai"
        ),
    ]
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        logger: Optional[Logger] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        precision: Optional[str] = None
    ) -> None:
        """
        Initialize OpenCLIP provider
        
        Args:
            settings: Application settings instance
            logger: Logger instance
            model_name: Model name to use (default from settings)
            device: Device to use ('cpu', 'cuda', 'auto')
            cache_dir: Model cache directory
            precision: Model precision ('fp32', 'fp16', 'int8')
        """
        super().__init__()
        
        self._settings = settings or Settings()
        self._logger = logger
        self._model_name = model_name or self._settings.ai_default_model
        self._device = device or self._settings.ai_device
        self._cache_dir = Path(cache_dir or self._settings.ai_cache_dir)
        self._precision = precision or self._settings.ai_precision
        
        # Model components
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._actual_device = None
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate provider configuration"""
        # Check if model is supported
        supported_models = [m.name for m in self.SUPPORTED_MODELS]
        if self._model_name not in supported_models:
            raise ConfigurationException(
                f"Model '{self._model_name}' is not supported. "
                f"Supported models: {supported_models}"
            )
        
        # Validate device
        if self._device not in ['auto', 'cpu', 'cuda', 'mps']:
            raise ConfigurationException(
                f"Invalid device '{self._device}'. "
                f"Must be 'auto', 'cpu', 'cuda', or 'mps'"
            )
        
        # Validate precision
        if self._precision not in ['fp32', 'fp16', 'int8']:
            raise ConfigurationException(
                f"Invalid precision '{self._precision}'. "
                f"Must be 'fp32', 'fp16', or 'int8'"
            )
    
    def _detect_device(self) -> str:
        """
        Detect the best available device
        
        Returns:
            Device string ('cpu', 'cuda', 'mps')
        """
        if self._device != 'auto':
            return self._device
        
        # Try to detect GPU
        try:
            import torch
            if torch.cuda.is_available():
                self._log_info("CUDA device detected and will be used")
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._log_info("Apple MPS device detected and will be used")
                return 'mps'
        except ImportError:
            self._log_warning("PyTorch not available, falling back to CPU")
        
        self._log_info("Using CPU for inference")
        return 'cpu'
    
    def get_provider_name(self) -> str:
        """Return provider name"""
        return "openclip"
    
    def get_supported_models(self) -> List[ModelInfo]:
        """Return list of supported models"""
        return self.SUPPORTED_MODELS.copy()
    
    def load_model(self, model_name: str) -> None:
        """
        Load OpenCLIP model into memory
        
        Args:
            model_name: Name of the model to load
            
        Raises:
            ModelLoadException: If model loading fails
        """
        try:
            # Import OpenCLIP
            import open_clip
            import torch
            
            self._log_info(f"Loading OpenCLIP model: {model_name}")
            
            # Detect device
            self._actual_device = self._detect_device()
            
            # Create cache directory if it doesn't exist
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load model with caching
            model, preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained='openai',
                cache_dir=str(self._cache_dir),
                device=self._actual_device
            )
            
            # Load tokenizer
            tokenizer = open_clip.get_tokenizer(model_name)
            
            # Set model to evaluation mode
            model.eval()
            
            # Apply precision settings
            if self._precision == 'fp16' and self._actual_device != 'cpu':
                model = model.half()
            elif self._precision == 'int8':
                try:
                    import torch.quantization
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                except Exception as e:
                    self._log_warning(f"INT8 quantization not supported: {e}")
            
            # Store model components
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = tokenizer
            self._model_name = model_name
            self._loaded = True
            
            self._log_info(
                f"Model loaded successfully on {self._actual_device} "
                f"with precision {self._precision}"
            )
            
        except ImportError as e:
            raise ModelLoadException(
                f"OpenCLIP library not installed: {e}. "
                f"Install with: pip install open-clip-torch",
                "openclip"
            )
        except Exception as e:
            raise ModelLoadException(
                f"Failed to load model '{model_name}': {str(e)}",
                model_name,
                {"error": str(e)}
            )
    
    def unload_model(self) -> None:
        """Unload model from memory"""
        if self._model is not None:
            try:
                import torch
                if self._actual_device == 'cuda':
                    torch.cuda.empty_cache()
                del self._model
                del self._preprocess
                del self._tokenizer
            except Exception as e:
                self._log_warning(f"Error during model unload: {e}")
            
            self._model = None
            self._preprocess = None
            self._tokenizer = None
            self._model_name = None
            self._loaded = False
            self._actual_device = None
            
            self._log_info("Model unloaded successfully")
    
    def generate_embedding(self, image_data: bytes) -> EmbeddingVector:
        """
        Generate embedding for single image
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Normalized embedding vector
            
        Raises:
            InferenceException: If embedding generation fails
        """
        if not self.is_model_loaded():
            raise InferenceException(
                "Model not loaded. Call load_model() first.",
                "openclip"
            )
        
        try:
            import torch
            from PIL import Image
            import io
            
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_tensor = self._preprocess(image).unsqueeze(0)
            
            # Move to device
            image_tensor = image_tensor.to(self._actual_device)
            
            # Generate embedding
            with torch.no_grad():
                # Get image features
                image_features = self._model.encode_image(image_tensor)
                
                # Normalize embedding
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy array
                embedding = image_features.cpu().numpy().flatten()
            
            # Create embedding vector
            embedding_vector = EmbeddingVector(
                vector=embedding.tolist(),
                model=self._model_name,
                provider=self.get_provider_name(),
                dimensions=len(embedding)
            )
            
            self._log_debug(f"Generated embedding with {len(embedding)} dimensions")
            
            return embedding_vector
            
        except Exception as e:
            raise InferenceException(
                f"Failed to generate embedding: {str(e)}",
                "openclip",
                {"error": str(e)}
            )
    
    def generate_embeddings_batch(self, images_data: List[bytes]) -> List[EmbeddingVector]:
        """
        Generate embeddings for multiple images
        
        Args:
            images_data: List of raw image bytes
            
        Returns:
            List of normalized embedding vectors
            
        Raises:
            InferenceException: If embedding generation fails
        """
        if not self.is_model_loaded():
            raise InferenceException(
                "Model not loaded. Call load_model() first.",
                "openclip"
            )
        
        if not images_data:
            return []
        
        try:
            import torch
            from PIL import Image
            import io
            
            self._log_info(f"Generating embeddings for {len(images_data)} images")
            
            # Preprocess all images
            image_tensors = []
            for image_data in images_data:
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                image_tensor = self._preprocess(image)
                image_tensors.append(image_tensor)
            
            # Stack into batch
            batch = torch.stack(image_tensors).to(self._actual_device)
            
            # Generate embeddings
            with torch.no_grad():
                # Get image features
                image_features = self._model.encode_image(batch)
                
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy arrays
                embeddings = image_features.cpu().numpy()
            
            # Create embedding vectors
            embedding_vectors = []
            for embedding in embeddings:
                embedding_vector = EmbeddingVector(
                    vector=embedding.tolist(),
                    model=self._model_name,
                    provider=self.get_provider_name(),
                    dimensions=len(embedding)
                )
                embedding_vectors.append(embedding_vector)
            
            self._log_info(f"Generated {len(embedding_vectors)} embeddings successfully")
            
            return embedding_vectors
            
        except Exception as e:
            raise InferenceException(
                f"Failed to generate batch embeddings: {str(e)}",
                "openclip",
                {"error": str(e), "batch_size": len(images_data)}
            )
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model information"""
        if not self._model_name:
            return None
        
        for model in self.SUPPORTED_MODELS:
            if model.name == self._model_name:
                return model
        return None
    
    def warmup(self) -> None:
        """Warm up model for faster inference"""
        if not self.is_model_loaded():
            self._log_warning("Cannot warmup: model not loaded")
            return
        
        try:
            import torch
            from PIL import Image
            import io
            
            self._log_info("Warming up model with dummy image")
            
            # Create a dummy image
            dummy_image = Image.new('RGB', (224, 224), color='red')
            image_tensor = self._preprocess(dummy_image).unsqueeze(0).to(self._actual_device)
            
            # Run inference
            with torch.no_grad():
                _ = self._model.encode_image(image_tensor)
            
            self._log_info("Model warmup completed")
            
        except Exception as e:
            self._log_warning(f"Model warmup failed: {e}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded and self._model is not None
    
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
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information
        
        Returns:
            Dictionary with device information
        """
        info = {
            "device": self._actual_device or "not_loaded",
            "configured_device": self._device,
            "precision": self._precision,
            "model_loaded": self.is_model_loaded()
        }
        
        if self._actual_device == 'cuda':
            try:
                import torch
                info["cuda_version"] = torch.version.cuda
                info["cuda_device_count"] = torch.cuda.device_count()
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
                info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0)
            except Exception:
                pass
        
        return info
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get model cache information
        
        Returns:
            Dictionary with cache information
        """
        cache_info = {
            "cache_dir": str(self._cache_dir),
            "cache_exists": self._cache_dir.exists(),
            "cached_models": []
        }
        
        if self._cache_dir.exists():
            for item in self._cache_dir.iterdir():
                if item.is_file():
                    cache_info["cached_models"].append({
                        "name": item.name,
                        "size_mb": item.stat().st_size / (1024 * 1024)
                    })
        
        return cache_info