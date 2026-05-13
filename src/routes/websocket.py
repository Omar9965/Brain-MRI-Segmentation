from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Any
import logging
from utils.websocket_manager import manager
from utils.async_processor import BatchProcessor, TaskPriority
import asyncio
import json


# Global batch processor instance
batch_processor = None


async def get_batch_processor():
    """Get or create the global batch processor"""
    global batch_processor
    if batch_processor is None:
        batch_processor = BatchProcessor(
            max_concurrent_tasks=4,
            progress_callback=manager.send_progress_update
        )
    return batch_processor


router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/progress/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        await manager.send_personal_message({
            "type": "connection_established",
            "session_id": session_id,
            "message": "Connected to progress updates"
        }, websocket)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for incoming message with timeout using asyncio.wait_for
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                
                if message.get("type") == "get_status":
                    # Send current status
                    processor = await get_batch_processor()
                    tasks = await processor.get_all_tasks()
                    await manager.send_personal_message({
                        "type": "current_status",
                        "session_id": session_id,
                        "tasks": [
                            {
                                "task_id": task.task_id,
                                "filename": task.filename,
                                "status": task.status.value,
                                "progress": task.progress,
                                "created_at": task.created_at.isoformat(),
                                "started_at": task.started_at.isoformat() if task.started_at else None,
                                "completed_at": task.completed_at.isoformat() if task.completed_at else None
                            }
                            for task in tasks
                        ]
                    }, websocket)
                
                elif message.get("type") == "cancel_task":
                    task_id = message.get("task_id")
                    if task_id:
                        processor = await get_batch_processor()
                        success = await processor.cancel_task(task_id)
                        await manager.send_personal_message({
                            "type": "cancel_response",
                            "task_id": task_id,
                            "success": success
                        }, websocket)
                
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await manager.send_personal_message({
                    "type": "ping"
                }, websocket)
                
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)


# Helper function to broadcast progress updates
async def broadcast_progress_update(status: Dict[str, Any]):
    """Broadcast progress update to all connected clients"""
    await manager.send_progress_update(status)


# Helper function to broadcast task updates
async def broadcast_task_update(task_id: str, task_data: Dict[str, Any]):
    """Broadcast task update to all connected clients"""
    await manager.send_task_update(task_id, task_data)


# Helper function to broadcast completion
async def broadcast_completion(task_id: str, result: Dict[str, Any]):
    """Broadcast completion notification"""
    await manager.send_completion(task_id, result)


# Helper function to broadcast errors
async def broadcast_error(error_message: str, task_id: str = None):
    """Broadcast error message"""
    await manager.send_error(error_message, task_id)