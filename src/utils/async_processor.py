import asyncio
import uuid
import time
import os
import logging
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProcessingTask:
    task_id: str
    filename: str
    file_path: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: float = 0.0


class BatchProcessor:
    def __init__(self, max_concurrent_tasks: int = 4, progress_callback: Optional[Callable] = None):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.progress_callback = progress_callback
        self.tasks: Dict[str, ProcessingTask] = {}
        self.task_queue: List[str] = []
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
    
    async def add_task(self, filename: str, file_path: str, priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Add a new processing task to the queue"""
        task_id = str(uuid.uuid4())
        task = ProcessingTask(
            task_id=task_id,
            filename=filename,
            file_path=file_path,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.tasks[task_id] = task
        self.task_queue.append(task_id)
        
        # Sort queue by priority
        self.task_queue.sort(key=lambda tid: self.tasks[tid].priority.value, reverse=True)
        
        # Launch processing loop as a background task (non-blocking)
        if not self._running:
            self._processing_task = asyncio.create_task(self._start_processing())
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Get the status of a specific task"""
        return self.tasks.get(task_id)
    
    async def get_all_tasks(self) -> List[ProcessingTask]:
        """Get all tasks in the processor"""
        return list(self.tasks.values())
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                
                # Cancel the async task if it's running
                if task_id in self.active_tasks:
                    self.active_tasks[task_id].cancel()
                    del self.active_tasks[task_id]
                
                # Remove from queue
                if task_id in self.task_queue:
                    self.task_queue.remove(task_id)
                
                await self._update_progress()
                return True
        return False
    
    async def clear_completed_tasks(self) -> int:
        """Clear completed tasks and return count"""
        completed_count = 0
        tasks_to_remove = []
        
        for task_id, task in self.tasks.items():
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                tasks_to_remove.append(task_id)
                completed_count += 1
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        return completed_count
    
    async def _start_processing(self):
        """Start the processing loop (runs as a background task)"""
        self._running = True
        try:
            while self._running:
                if not self.task_queue:
                    # No more tasks — stop the loop
                    break
                await self._process_queue()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
        except asyncio.CancelledError:
            logger.info("Processing loop cancelled")
        finally:
            self._running = False
    
    async def _process_queue(self):
        """Process tasks in the queue"""
        # Get available slots
        available_slots = self.max_concurrent_tasks - len(self.active_tasks)
        if available_slots <= 0:
            return
        
        # Get tasks from queue
        tasks_to_process = []
        for _ in range(min(available_slots, len(self.task_queue))):
            if self.task_queue:
                task_id = self.task_queue.pop(0)
                task = self.tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    tasks_to_process.append(task_id)
        
        # Create tasks for processing
        for task_id in tasks_to_process:
            task = self.tasks[task_id]
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            
            # Create async task
            async_task = asyncio.create_task(self._process_task(task_id))
            self.active_tasks[task_id] = async_task
            
            # Use ensure_future to properly schedule the async completion handler
            async_task.add_done_callback(
                lambda t, tid=task_id: asyncio.ensure_future(self._task_completed(tid))
            )
    
    async def _process_task(self, task_id: str):
        """Process a single task using the actual segmentation model"""
        task = self.tasks[task_id]
        
        try:
            # Import here to avoid circular imports
            from models import segment_image
            
            # Update progress: starting
            task.progress = 10
            await self._update_progress()
            
            # Run the actual segmentation in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            task.progress = 30
            await self._update_progress()
            
            result = await loop.run_in_executor(
                None,
                lambda: segment_image(task.file_path, filename=task.filename, return_overlay=True)
            )
            
            task.progress = 90
            await self._update_progress()
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 100
            task.completed_at = datetime.now()
            
            await self._update_progress()
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            raise
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
        finally:
            # Clean up the temporary file after processing
            if task.file_path and os.path.exists(task.file_path):
                try:
                    os.remove(task.file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {task.file_path}: {e}")
    
    async def _task_completed(self, task_id: str):
        """Handle task completion"""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        await self._update_progress()
    
    async def _update_progress(self):
        """Update progress callback with current status"""
        if self.progress_callback:
            status = {
                "total_tasks": len(self.tasks),
                "pending_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                "processing_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.PROCESSING]),
                "completed_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
                "failed_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED]),
                "cancelled_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.CANCELLED]),
                "active_tasks": len(self.active_tasks),
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "filename": task.filename,
                        "status": task.status.value,
                        "progress": task.progress,
                        "created_at": task.created_at.isoformat(),
                        "started_at": task.started_at.isoformat() if task.started_at else None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                        "error": task.error
                    }
                    for task in self.tasks.values()
                ]
            }
            try:
                await self.progress_callback(status)
            except Exception as e:
                logger.warning(f"Failed to send progress update: {e}")
    
    async def stop(self):
        """Stop the processor"""
        self._running = False
        # Cancel the processing loop
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
        # Cancel all active tasks
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)