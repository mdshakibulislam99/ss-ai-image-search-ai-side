"""
Queue interface - Abstract base class for job queues
"""

from abc import ABC, abstractmethod
from typing import Any,  Dict,  Generic, Optional, TypeVar

from ..entities.queue_job import QueueJob  # type: ignore

T = TypeVar("T", bound=QueueJob)


class QueueStats:
    """Queue statistics"""
    
    def __init__(self, pending: int, processing: int, completed: int, failed: int) -> None:
        self.pending = pending
        self.processing = processing
        self.completed = completed
        self.failed = failed
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary"""
        return {
            "pending": self.pending,
            "processing": self.processing,
            "completed": self.completed,
            "failed": self.failed,
        }


class Queue(ABC, Generic[T]):
    """Abstract base class for job queues"""
    
    @abstractmethod
    async def enqueue(self, job: T, priority: int = 0) -> str:
        """
        Add job to queue
        
        Args:
            job: Job to enqueue
            priority: Job priority (higher = more important)
            
        Returns:
            Job ID
        """
        pass
    
    @abstractmethod
    async def dequeue(self) -> Optional[T]:
        """Get next job from queue"""
        pass
    
    @abstractmethod
    async def get_job(self, job_id: str) -> Optional[T]:
        """Get job by ID"""
        pass
    
    @abstractmethod
    async def update_job_status(self, job_id: str, status: str, 
                                result: Optional[Dict[str, Any]] = None, 
                                error: Optional[str] = None) -> None:
        """Update job status"""
        pass
    
    @abstractmethod
    async def retry_job(self, job_id: str) -> bool:
        """Retry failed job"""
        pass
    
    @abstractmethod
    async def get_queue_stats(self) -> QueueStats:
        """Get queue statistics"""
        pass
    
    @abstractmethod
    async def clear_queue(self, job_type: Optional[str] = None) -> int:
        """Clear queue, optionally by job type"""
        pass