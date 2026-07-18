"""
Dependency Injection Container

Service container for managing dependencies and their lifetimes
"""

from typing import Dict, Type, TypeVar, Generic, Optional, Any
from enum import Enum

from .config.settings import Settings
from .domain.interfaces.ai_provider import AIProvider
from .domain.interfaces.vector_store import VectorStore
from .domain.interfaces.cache import Cache
from .domain.interfaces.queue import Queue
from .domain.interfaces.storage import Storage
from .domain.interfaces.logger import Logger
from .domain.interfaces.repository import Repository
from .infrastructure.cache.memory_cache import MemoryCache
from .infrastructure.logging.logger import LoggerImpl
from .infrastructure.storage.local_storage import LocalStorage

T = TypeVar("T")


class ServiceLifetime(Enum):
    """Service lifetime enumeration"""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


class ServiceDescriptor(Generic[T]):
    """Service descriptor for container registration"""
    
    def __init__(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        instance: Optional[T] = None,
        factory: Optional[callable] = None,
    ):
        """
        Initialize service descriptor
        
        Args:
            service_type: Service interface/type
            implementation_type: Implementation type
            lifetime: Service lifetime
            instance: Pre-created instance (for singletons)
            factory: Factory function for creating instances
        """
        self.service_type = service_type
        self.implementation_type = implementation_type
        self.lifetime = lifetime
        self.instance = instance
        self.factory = factory


class Container:
    """
    Dependency Injection Container
    
    Manages service registration, resolution, and lifetime
    """
    
    def __init__(self):
        """Initialize container"""
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
    
    def register(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        instance: Optional[T] = None,
        factory: Optional[callable] = None,
    ) -> None:
        """
        Register a service
        
        Args:
            service_type: Service interface/type
            implementation_type: Implementation type (if not using factory)
            lifetime: Service lifetime
            instance: Pre-created instance (for singletons)
            factory: Factory function for creating instances
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=lifetime,
            instance=instance,
            factory=factory,
        )
        self._services[service_type] = descriptor
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service
        
        Args:
            service_type: Service type to resolve
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service is not registered
        """
        if service_type not in self._services:
            raise KeyError(f"Service {service_type} is not registered")
        
        descriptor = self._services[service_type]
        
        # Handle singleton lifetime
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if descriptor.instance is not None:
                return descriptor.instance
            
            if service_type in self._singletons:
                return self._singletons[service_type]
            
            instance = self._create_instance(descriptor)
            self._singletons[service_type] = instance
            return instance
        
        # Handle scoped lifetime
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]
            
            instance = self._create_instance(descriptor)
            self._scoped_instances[service_type] = instance
            return instance
        
        # Handle transient lifetime
        else:
            return self._create_instance(descriptor)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create an instance from descriptor"""
        # Use factory if provided
        if descriptor.factory is not None:
            return descriptor.factory(self)
        
        # Use pre-created instance if provided
        if descriptor.instance is not None:
            return descriptor.instance
        
        # Use implementation type
        if descriptor.implementation_type is not None:
            # Try to resolve dependencies from constructor
            import inspect
            sig = inspect.signature(descriptor.implementation_type.__init__)
            params = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                param_type = param.annotation
                if param_type != inspect.Parameter.empty:
                    try:
                        params[param_name] = self.resolve(param_type)
                    except KeyError:
                        # Dependency not registered, use default if available
                        if param.default != inspect.Parameter.empty:
                            params[param_name] = param.default
            
            return descriptor.implementation_type(**params)
        
        raise ValueError(f"Cannot create instance for {descriptor.service_type}")
    
    def clear_scoped(self) -> None:
        """Clear scoped instances"""
        self._scoped_instances.clear()
    
    def clear_all(self) -> None:
        """Clear all cached instances"""
        self._singletons.clear()
        self._scoped_instances.clear()


# Global container instance
container = Container()


def configure_services(settings: Settings) -> None:
    """
    Configure all services in the container
    
    Args:
        settings: Application settings
    """
    # Register settings as singleton
    container.register(Settings, instance=settings)
    
    # Register logger as singleton
    container.register(
        Logger,
        implementation_type=LoggerImpl,
        lifetime=ServiceLifetime.SINGLETON
    )
    
    # Register cache as singleton
    container.register(
        Cache,
        implementation_type=MemoryCache,
        lifetime=ServiceLifetime.SINGLETON
    )
    
    # Register storage as singleton
    container.register(
        Storage,
        implementation_type=LocalStorage,
        lifetime=ServiceLifetime.SINGLETON
    )
    
    # Register AI providers (transient - created per request)
    # These will be created via factory pattern
    container.register(
        AIProvider,
        lifetime=ServiceLifetime.TRANSIENT
    )
    
    # Register vector stores (transient - created per request)
    container.register(
        VectorStore,
        lifetime=ServiceLifetime.TRANSIENT
    )
    
    # Register repositories (transient)
    container.register(
        Repository,
        lifetime=ServiceLifetime.TRANSIENT
    )
    
    # Register queues (transient)
    container.register(
        Queue,
        lifetime=ServiceLifetime.TRANSIENT
    )