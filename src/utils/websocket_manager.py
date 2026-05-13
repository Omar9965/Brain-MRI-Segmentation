import asyncio
import json
import logging
from typing import Dict, Set, Optional
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Disconnect a WebSocket connection"""
        self.active_connections.discard(websocket)
        self.logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            self.logger.error(f"Error sending message to WebSocket: {e}")
            await self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected WebSockets"""
        if not self.active_connections:
            return
        
        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Error broadcasting message to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            await self.disconnect(websocket)
    
    async def send_progress_update(self, status: dict):
        """Send progress update to all connected clients"""
        message = {
            "type": "progress_update",
            "timestamp": datetime.now().isoformat(),
            "data": status
        }
        await self.broadcast(message)
    
    async def send_task_update(self, task_id: str, task_data: dict):
        """Send task update to all connected clients"""
        message = {
            "type": "task_update",
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "data": task_data
        }
        await self.broadcast(message)
    
    async def send_error(self, error_message: str, task_id: str = None):
        """Send error message to all connected clients"""
        message = {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "message": error_message,
            "task_id": task_id
        }
        await self.broadcast(message)
    
    async def send_completion(self, task_id: str, result: dict):
        """Send completion notification to all connected clients"""
        message = {
            "type": "task_completed",
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "data": result
        }
        await self.broadcast(message)


# Global connection manager instance
manager = ConnectionManager()