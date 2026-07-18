"""
Memory Queue - In-memory queue implementation
"""

from typing import Optional

from ...domain.interfaces.queue import Queue, QueueStats  # type: ignore
from ...domain.entities.queue_job import QueueJob, JobStatus  # type: ignore


class MemoryQueue(Queue):
    """In-memory queue implementation"""
    
    def __init__(self) -> None:
        """Initialize memory queue"""
        self._queue = []
        self._jobs = {}
    
    async def enqueue(self, job: QueueJob, priority: int = 0) -> str:
        """
        Add job to queue
        
        Args:
            job: Job to enqueue
            priority: Job priority (higher = more important)
            
        Returns:
            Job ID
        """
        job.status = JobStatus.QUEUED
        self._jobs[job.job_id] = job
        
        # Insert based on priority (higher priority first)
        inserted = False
        for i, (existing_priority, _) in enumerate(self._queue):
            if priority > existing_priority:
                self._queue.insert(i, (priority, job.job_id))
                inserted = True
                break
        
        if not inserted:
            self._queue.append((priority, job.job_id))
        
        return job.job_id
    
    async def dequeue(self) -> Optional[QueueJob]:
        """Get next job from queue"""
        if not self._queue:
            return None
        
        # Get highest priority job
        priority, job_id = self._queue.pop(0)
        job = self._jobs.get(job_id)
        
        if job:
            job.start()
        
        return job
    
    async def get_job(self, job_id: str) -> Optional[QueueJob]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    async def update_job_status(self, job_id: str, status: str, 
                                result: Optional[dict] = None, 
                                error: Optional[str] = None) -> None:
        """Update job status"""
        job = self._jobs.get(job_id)
        if job:
            job.status = JobStatus(status)
            if result:
                job.result = result
            if error:
                job.error = error
    
    async def retry_job(self, job_id: str) -> bool:
        """Retry failed job"""
        job = self._jobs.get(job_id)
        if job and job.can_retry():
            job.retry()
            await self.enqueue(job, job.priority)
            return True
        return False
    
    async def get_queue_stats(self) -> QueueStats:
        """Get queue statistics"""
        pending = sum(1 for job in self._jobs.values() if job.status == JobStatus.QUEUED)
        processing = sum(1 for job in self._jobs.values() if job.status == JobStatus.PROCESSING)
        completed = sum(1 for job in self._jobs.values() if job.status == JobStatus.COMPLETED)
        failed = sum(1 for job in self._jobs.values() if job.status == JobStatus.FAILED)
        
        return QueueStats(
            pending=pending,
            processing=processing,
            completed=completed,
            failed=failed
        )
    
    async def clear_queue(self, job_type: Optional[str] = None) -> int:
        """Clear queue, optionally by job type"""
        if job_type:
            # Remove jobs of specific type
            removed = 0
            self._queue = [(p, jid) for p, jid in self._queue 
                          if self._jobs.get(jid) and self._jobs[jid].job_type.value != job_type]
            for job_id in list(self._jobs.keys()):
                if self._jobs[job_id].job_type.value == job_type:
                    del self._jobs[job_id]
                    removed += 1
            return removed
        else:
            # Clear all
            count = len(self._queue)
            self._queue.clear()
            self._jobs.clear()
            return count