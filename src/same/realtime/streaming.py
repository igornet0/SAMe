"""
Real-time streaming processing and WebSocket integration for SAMe system
"""

import logging
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from datetime import datetime
import weakref

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of real-time events"""
    SEARCH_REQUEST = "search_request"
    SEARCH_RESULT = "search_result"
    PROCESSING_UPDATE = "processing_update"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class StreamingMode(Enum):
    """Streaming processing modes"""
    REAL_TIME = "real_time"      # Immediate processing
    BATCH = "batch"              # Batch processing with intervals
    HYBRID = "hybrid"            # Adaptive based on load


@dataclass
class StreamEvent:
    """Real-time stream event"""
    event_id: str
    event_type: EventType
    timestamp: float
    data: Dict[str, Any]
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: int = 1  # 1=low, 5=high
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        event_dict = asdict(self)
        event_dict['event_type'] = self.event_type.value
        event_dict['timestamp'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return json.dumps(event_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StreamEvent':
        """Create from JSON string"""
        data = json.loads(json_str)
        data['event_type'] = EventType(data['event_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp']).timestamp()
        return cls(**data)


class WebSocketConnection:
    """WebSocket connection wrapper"""
    
    def __init__(self, websocket, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.session_id = str(uuid.uuid4())
        self.connected_at = time.time()
        self.last_heartbeat = time.time()
        self.subscriptions: Set[EventType] = set()
        self.is_active = True
        
    async def send_event(self, event: StreamEvent):
        """Send event to client"""
        try:
            if self.is_active and not self.websocket.closed:
                await self.websocket.send_text(event.to_json())
                return True
        except Exception as e:
            logger.error(f"Error sending event to client {self.client_id}: {e}")
            self.is_active = False
        return False
    
    async def send_heartbeat(self):
        """Send heartbeat to client"""
        heartbeat_event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.HEARTBEAT,
            timestamp=time.time(),
            data={"status": "alive", "server_time": time.time()},
            client_id=self.client_id,
            session_id=self.session_id
        )
        return await self.send_event(heartbeat_event)
    
    def update_heartbeat(self):
        """Update last heartbeat timestamp"""
        self.last_heartbeat = time.time()
    
    def is_stale(self, timeout: int = 60) -> bool:
        """Check if connection is stale"""
        return time.time() - self.last_heartbeat > timeout


class RealTimeProcessor:
    """Real-time streaming processor"""
    
    def __init__(self, mode: StreamingMode = StreamingMode.HYBRID):
        self.mode = mode
        self.connections: Dict[str, WebSocketConnection] = {}
        self.event_queue = asyncio.Queue()
        self.processing_tasks: Set[asyncio.Task] = set()
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.metrics = {
            'events_processed': 0,
            'events_failed': 0,
            'active_connections': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        self._running = False
        self._heartbeat_task = None
        
    async def start(self):
        """Start the real-time processor"""
        if self._running:
            return
            
        self._running = True
        logger.info(f"Starting real-time processor in {self.mode.value} mode")
        
        # Start event processing task
        processing_task = asyncio.create_task(self._process_events())
        self.processing_tasks.add(processing_task)
        
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start connection cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_stale_connections())
        self.processing_tasks.add(cleanup_task)
        
        logger.info("Real-time processor started successfully")
    
    async def stop(self):
        """Stop the real-time processor"""
        if not self._running:
            return
            
        self._running = False
        logger.info("Stopping real-time processor")
        
        # Cancel all tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            
        for task in self.processing_tasks:
            task.cancel()
        
        # Close all connections
        for connection in list(self.connections.values()):
            await self._disconnect_client(connection.client_id)
        
        logger.info("Real-time processor stopped")
    
    async def connect_client(self, websocket, client_id: str = None) -> str:
        """Connect a new WebSocket client"""
        if not client_id:
            client_id = str(uuid.uuid4())
        
        connection = WebSocketConnection(websocket, client_id)
        self.connections[client_id] = connection
        self.metrics['active_connections'] = len(self.connections)
        
        logger.info(f"Client {client_id} connected (session: {connection.session_id})")
        
        # Send welcome event
        welcome_event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.SYSTEM_STATUS,
            timestamp=time.time(),
            data={
                "status": "connected",
                "client_id": client_id,
                "session_id": connection.session_id,
                "server_capabilities": ["search", "processing", "monitoring"]
            },
            client_id=client_id,
            session_id=connection.session_id
        )
        
        await connection.send_event(welcome_event)
        return client_id
    
    async def disconnect_client(self, client_id: str):
        """Disconnect a WebSocket client"""
        await self._disconnect_client(client_id)
    
    async def _disconnect_client(self, client_id: str):
        """Internal disconnect method"""
        if client_id in self.connections:
            connection = self.connections[client_id]
            connection.is_active = False
            
            try:
                if not connection.websocket.closed:
                    await connection.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket for {client_id}: {e}")
            
            del self.connections[client_id]
            self.metrics['active_connections'] = len(self.connections)
            logger.info(f"Client {client_id} disconnected")
    
    async def subscribe_client(self, client_id: str, event_types: List[EventType]):
        """Subscribe client to specific event types"""
        if client_id in self.connections:
            connection = self.connections[client_id]
            connection.subscriptions.update(event_types)
            logger.debug(f"Client {client_id} subscribed to {[et.value for et in event_types]}")
    
    async def emit_event(self, event: StreamEvent):
        """Emit event to the processing queue"""
        await self.event_queue.put(event)
    
    async def broadcast_event(self, event: StreamEvent, target_clients: Optional[List[str]] = None):
        """Broadcast event to connected clients"""
        if target_clients:
            # Send to specific clients
            for client_id in target_clients:
                if client_id in self.connections:
                    connection = self.connections[client_id]
                    if event.event_type in connection.subscriptions or not connection.subscriptions:
                        await connection.send_event(event)
        else:
            # Broadcast to all subscribed clients
            for connection in self.connections.values():
                if event.event_type in connection.subscriptions or not connection.subscriptions:
                    await connection.send_event(event)
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """Register event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _process_events(self):
        """Main event processing loop"""
        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                start_time = time.time()
                
                # Process event with registered handlers
                if event.event_type in self.event_handlers:
                    for handler in self.event_handlers[event.event_type]:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(event)
                            else:
                                handler(event)
                        except Exception as e:
                            logger.error(f"Error in event handler: {e}")
                
                # Broadcast event to clients
                await self.broadcast_event(event)
                
                # Update metrics
                processing_time = time.time() - start_time
                self.metrics['events_processed'] += 1
                self.metrics['total_processing_time'] += processing_time
                self.metrics['average_processing_time'] = (
                    self.metrics['total_processing_time'] / self.metrics['events_processed']
                )
                
            except asyncio.TimeoutError:
                continue  # No events to process
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                self.metrics['events_failed'] += 1
    
    async def _heartbeat_loop(self):
        """Heartbeat loop to keep connections alive"""
        while self._running:
            try:
                for connection in list(self.connections.values()):
                    if connection.is_active:
                        success = await connection.send_heartbeat()
                        if not success:
                            await self._disconnect_client(connection.client_id)
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_stale_connections(self):
        """Clean up stale connections"""
        while self._running:
            try:
                stale_clients = []
                for client_id, connection in self.connections.items():
                    if connection.is_stale():
                        stale_clients.append(client_id)
                
                for client_id in stale_clients:
                    await self._disconnect_client(client_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(10)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get real-time processing metrics"""
        return {
            **self.metrics,
            'mode': self.mode.value,
            'running': self._running,
            'queue_size': self.event_queue.qsize(),
            'registered_handlers': {
                event_type.value: len(handlers) 
                for event_type, handlers in self.event_handlers.items()
            }
        }


# Global real-time processor instance
realtime_processor = RealTimeProcessor()
