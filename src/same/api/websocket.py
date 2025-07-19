"""
WebSocket API endpoints for real-time communication
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.routing import APIRouter
import uuid

from ..realtime.streaming import realtime_processor, StreamEvent, EventType
from ..search_engine.semantic_search import SemanticSearchEngine
from ..text_processing.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

# WebSocket router
websocket_router = APIRouter()


class WebSocketManager:
    """WebSocket connection manager with real-time search capabilities"""
    
    def __init__(self):
        self.search_engine: Optional[SemanticSearchEngine] = None
        self.preprocessor: Optional[TextPreprocessor] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize search components"""
        if self._initialized:
            return
        
        try:
            # Initialize search engine
            self.search_engine = SemanticSearchEngine()
            
            # Initialize text preprocessor
            self.preprocessor = TextPreprocessor()
            
            # Start real-time processor
            await realtime_processor.start()
            
            # Register event handlers
            realtime_processor.register_event_handler(
                EventType.SEARCH_REQUEST, 
                self._handle_search_request
            )
            
            self._initialized = True
            logger.info("WebSocket manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}")
            raise
    
    async def _handle_search_request(self, event: StreamEvent):
        """Handle real-time search requests"""
        try:
            search_data = event.data
            query = search_data.get('query', '')
            max_results = search_data.get('max_results', 10)
            
            if not query:
                return
            
            # Emit processing update
            processing_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.PROCESSING_UPDATE,
                timestamp=event.timestamp,
                data={
                    'status': 'processing',
                    'query': query,
                    'stage': 'search_execution'
                },
                client_id=event.client_id,
                session_id=event.session_id
            )
            await realtime_processor.broadcast_event(processing_event, [event.client_id])
            
            # Perform search
            if self.search_engine and hasattr(self.search_engine, 'search'):
                results = self.search_engine.search(query, top_k=max_results)
            else:
                results = []
            
            # Emit search results
            result_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.SEARCH_RESULT,
                timestamp=event.timestamp,
                data={
                    'query': query,
                    'results': results,
                    'result_count': len(results),
                    'processing_time': 0.1  # Placeholder
                },
                client_id=event.client_id,
                session_id=event.session_id
            )
            await realtime_processor.broadcast_event(result_event, [event.client_id])
            
        except Exception as e:
            logger.error(f"Error handling search request: {e}")
            
            # Emit error event
            error_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.ERROR,
                timestamp=event.timestamp,
                data={
                    'error': str(e),
                    'error_type': 'search_error',
                    'query': search_data.get('query', '')
                },
                client_id=event.client_id,
                session_id=event.session_id
            )
            await realtime_processor.broadcast_event(error_event, [event.client_id])


# Global WebSocket manager
ws_manager = WebSocketManager()


@websocket_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication"""
    await websocket.accept()
    
    # Initialize manager if needed
    await ws_manager.initialize()
    
    # Connect client
    client_id = await realtime_processor.connect_client(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await _handle_websocket_message(message, client_id)
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON from client {client_id}: {data}")
                await _send_error(websocket, "Invalid JSON format")
            except Exception as e:
                logger.error(f"Error handling message from {client_id}: {e}")
                await _send_error(websocket, str(e))
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        await realtime_processor.disconnect_client(client_id)


@websocket_router.websocket("/ws/search")
async def search_websocket_endpoint(websocket: WebSocket):
    """Dedicated WebSocket endpoint for search operations"""
    await websocket.accept()
    
    # Initialize manager if needed
    await ws_manager.initialize()
    
    # Connect client with search subscription
    client_id = await realtime_processor.connect_client(websocket)
    await realtime_processor.subscribe_client(
        client_id, 
        [EventType.SEARCH_RESULT, EventType.PROCESSING_UPDATE, EventType.ERROR]
    )
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                if message.get('type') == 'search':
                    # Create search event
                    search_event = StreamEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=EventType.SEARCH_REQUEST,
                        timestamp=asyncio.get_event_loop().time(),
                        data={
                            'query': message.get('query', ''),
                            'max_results': message.get('max_results', 10),
                            'filters': message.get('filters', {})
                        },
                        client_id=client_id
                    )
                    
                    # Emit search event
                    await realtime_processor.emit_event(search_event)
                
                elif message.get('type') == 'ping':
                    # Update heartbeat
                    if client_id in realtime_processor.connections:
                        realtime_processor.connections[client_id].update_heartbeat()
                
            except json.JSONDecodeError:
                await _send_error(websocket, "Invalid JSON format")
            except Exception as e:
                logger.error(f"Error in search WebSocket: {e}")
                await _send_error(websocket, str(e))
                
    except WebSocketDisconnect:
        logger.info(f"Search client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Search WebSocket error: {e}")
    finally:
        await realtime_processor.disconnect_client(client_id)


@websocket_router.websocket("/ws/monitor")
async def monitor_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for system monitoring"""
    await websocket.accept()
    
    # Initialize manager if needed
    await ws_manager.initialize()
    
    # Connect client with monitoring subscription
    client_id = await realtime_processor.connect_client(websocket)
    await realtime_processor.subscribe_client(
        client_id, 
        [EventType.SYSTEM_STATUS, EventType.PROCESSING_UPDATE]
    )
    
    try:
        # Send initial system status
        status_event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.SYSTEM_STATUS,
            timestamp=asyncio.get_event_loop().time(),
            data={
                'system_status': 'operational',
                'metrics': realtime_processor.get_metrics(),
                'active_connections': len(realtime_processor.connections)
            },
            client_id=client_id
        )
        await realtime_processor.broadcast_event(status_event, [client_id])
        
        # Keep connection alive and send periodic updates
        while True:
            await asyncio.sleep(10)  # Send updates every 10 seconds
            
            # Send system metrics update
            metrics_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.SYSTEM_STATUS,
                timestamp=asyncio.get_event_loop().time(),
                data={
                    'metrics': realtime_processor.get_metrics(),
                    'timestamp': asyncio.get_event_loop().time()
                },
                client_id=client_id
            )
            await realtime_processor.broadcast_event(metrics_event, [client_id])
            
    except WebSocketDisconnect:
        logger.info(f"Monitor client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Monitor WebSocket error: {e}")
    finally:
        await realtime_processor.disconnect_client(client_id)


async def _handle_websocket_message(message: Dict[str, Any], client_id: str):
    """Handle incoming WebSocket message"""
    message_type = message.get('type')
    
    if message_type == 'subscribe':
        # Subscribe to event types
        event_types = [EventType(et) for et in message.get('event_types', [])]
        await realtime_processor.subscribe_client(client_id, event_types)
        
    elif message_type == 'search':
        # Handle search request
        search_event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.SEARCH_REQUEST,
            timestamp=asyncio.get_event_loop().time(),
            data=message.get('data', {}),
            client_id=client_id
        )
        await realtime_processor.emit_event(search_event)
        
    elif message_type == 'ping':
        # Update heartbeat
        if client_id in realtime_processor.connections:
            realtime_processor.connections[client_id].update_heartbeat()
    
    else:
        logger.warning(f"Unknown message type from {client_id}: {message_type}")


async def _send_error(websocket: WebSocket, error_message: str):
    """Send error message to WebSocket client"""
    error_response = {
        'type': 'error',
        'error': error_message,
        'timestamp': asyncio.get_event_loop().time()
    }
    
    try:
        await websocket.send_text(json.dumps(error_response))
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")


# Health check for WebSocket system
async def get_websocket_health() -> Dict[str, Any]:
    """Get WebSocket system health status"""
    return {
        'status': 'healthy' if realtime_processor._running else 'stopped',
        'active_connections': len(realtime_processor.connections),
        'metrics': realtime_processor.get_metrics(),
        'initialized': ws_manager._initialized
    }
