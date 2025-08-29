"""
FastAPI REST and WebSocket API for people counting system.
Provides live counts, historical data, and real-time updates.
"""
import logging
import asyncio
import json
from typing import Dict, List, Optional
from datetime import datetime, date
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from db import CounterDB

logger = logging.getLogger(__name__)

# Pydantic models for API responses
class LiveCountsResponse(BaseModel):
    """Live counts response model."""
    in_count: int = 0
    out_count: int = 0
    occupancy: int = 0
    fps: float = 0.0
    timestamp: str
    status: str = "active"

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    uptime_seconds: float
    database_connected: bool
    active_connections: int

class CrossingEvent(BaseModel):
    """Crossing event model."""
    track_id: int
    direction: str
    confidence: float
    x: float
    y: float
    timestamp: str
    frame_number: Optional[int] = None

class CountsHistoryResponse(BaseModel):
    """Historical counts response model."""
    date: str
    in_count: int
    out_count: int
    occupancy: int

class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSockets."""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

class PeopleCounterAPI:
    """Main API class for people counting system."""
    
    def __init__(self, db_path: str = "counts.db"):
        """Initialize API with database connection."""
        self.app = FastAPI(
            title="People Counter API",
            description="Real-time people counting system with RTSP/MP4 support",
            version="1.0.0"
        )
        
        # Database
        self.db = CounterDB(db_path)
        
        # WebSocket manager
        self.ws_manager = WebSocketManager()
        
        # State
        self.start_time = datetime.now()
        self.current_counts = {'in': 0, 'out': 0, 'occupancy': 0}
        self.current_fps = 0.0
        self.system_status = "starting"
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self._register_routes()
        
        logger.info("People Counter API initialized")
    
    def _register_routes(self):
        """Register all API routes."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize database on startup."""
            await self.db.init_db()
            self.system_status = "active"
            logger.info("API startup complete")
        
        @self.app.get("/", response_model=Dict)
        async def root():
            """Root endpoint with basic info."""
            return {
                "name": "People Counter API",
                "version": "1.0.0",
                "status": self.system_status,
                "endpoints": {
                    "live_counts": "/api/live",
                    "today_counts": "/api/counts/today",
                    "health": "/api/health",
                    "websocket": "/ws/live"
                }
            }
        
        @self.app.get("/api/live", response_model=LiveCountsResponse)
        async def get_live_counts():
            """Get current live counts."""
            try:
                # Get today's counts from database
                db_counts = await self.db.get_today_counts()
                
                return LiveCountsResponse(
                    in_count=db_counts['in'],
                    out_count=db_counts['out'],
                    occupancy=db_counts['occupancy'],
                    fps=self.current_fps,
                    timestamp=datetime.now().isoformat(),
                    status=self.system_status
                )
            except Exception as e:
                logger.error(f"Failed to get live counts: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve live counts")
        
        @self.app.get("/api/counts/today", response_model=CountsHistoryResponse)
        async def get_today_counts():
            """Get today's total counts."""
            try:
                counts = await self.db.get_today_counts()
                today = date.today().isoformat()
                
                return CountsHistoryResponse(
                    date=today,
                    in_count=counts['in'],
                    out_count=counts['out'],
                    occupancy=counts['occupancy']
                )
            except Exception as e:
                logger.error(f"Failed to get today's counts: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve today's counts")
        
        @self.app.get("/api/counts/{target_date}", response_model=CountsHistoryResponse)
        async def get_counts_by_date(target_date: str):
            """Get counts for specific date (YYYY-MM-DD format)."""
            try:
                # Validate date format
                datetime.strptime(target_date, '%Y-%m-%d')
                
                counts = await self.db.get_counts_by_date(target_date)
                
                return CountsHistoryResponse(
                    date=target_date,
                    in_count=counts['in'],
                    out_count=counts['out'],
                    occupancy=counts['occupancy']
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
            except Exception as e:
                logger.error(f"Failed to get counts for {target_date}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve counts for {target_date}")
        
        @self.app.get("/api/crossings/recent")
        async def get_recent_crossings(limit: int = 100):
            """Get recent crossing events."""
            try:
                if limit > 1000:  # Prevent excessive queries
                    limit = 1000
                
                crossings = await self.db.get_recent_crossings(limit)
                return {"crossings": crossings, "count": len(crossings)}
            except Exception as e:
                logger.error(f"Failed to get recent crossings: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve recent crossings")
        
        @self.app.get("/api/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            try:
                # Test database connection
                await self.db.get_today_counts()
                db_connected = True
            except Exception:
                db_connected = False
            
            uptime = (datetime.now() - self.start_time).total_seconds()
            
            return HealthResponse(
                status=self.system_status,
                timestamp=datetime.now().isoformat(),
                uptime_seconds=uptime,
                database_connected=db_connected,
                active_connections=len(self.ws_manager.active_connections)
            )
        
        @self.app.post("/api/reset")
        async def reset_counts(background_tasks: BackgroundTasks):
            """Reset daily counts (admin endpoint)."""
            try:
                # This would typically require authentication in production
                self.current_counts = {'in': 0, 'out': 0, 'occupancy': 0}
                
                # Broadcast reset to WebSocket clients
                reset_message = json.dumps({
                    "type": "reset",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Counts have been reset"
                })
                background_tasks.add_task(self.ws_manager.broadcast, reset_message)
                
                return {"status": "success", "message": "Counts reset successfully"}
            except Exception as e:
                logger.error(f"Failed to reset counts: {e}")
                raise HTTPException(status_code=500, detail="Failed to reset counts")
        
        @self.app.websocket("/ws/live")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.ws_manager.connect(websocket)
            
            try:
                # Send initial data
                initial_data = {
                    "type": "initial",
                    "data": {
                        "in": self.current_counts['in'],
                        "out": self.current_counts['out'],
                        "occupancy": self.current_counts['occupancy'],
                        "fps": self.current_fps,
                        "status": self.system_status
                    },
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(initial_data))
                
                # Keep connection alive and handle client messages
                while True:
                    try:
                        # Wait for client message (ping/pong or commands)
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                        
                        # Handle client messages
                        try:
                            message = json.loads(data)
                            if message.get("type") == "ping":
                                pong_response = {
                                    "type": "pong",
                                    "timestamp": datetime.now().isoformat()
                                }
                                await websocket.send_text(json.dumps(pong_response))
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from WebSocket client: {data}")
                    
                    except asyncio.TimeoutError:
                        # Send keepalive
                        keepalive = {
                            "type": "keepalive",
                            "timestamp": datetime.now().isoformat()
                        }
                        await websocket.send_text(json.dumps(keepalive))
                    
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.ws_manager.disconnect(websocket)
    
    async def update_counts(self, counts: Dict[str, int], fps: float = 0.0):
        """Update current counts and broadcast to WebSocket clients.
        
        Args:
            counts: Dictionary with 'in', 'out', 'occupancy' keys
            fps: Current processing FPS
        """
        self.current_counts = counts
        self.current_fps = fps
        
        # Broadcast to WebSocket clients
        update_message = {
            "type": "update",
            "data": {
                "in": counts['in'],
                "out": counts['out'],
                "occupancy": counts['occupancy'],
                "fps": fps
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await self.ws_manager.broadcast(json.dumps(update_message))
    
    async def broadcast_crossing(self, crossing_event: Dict):
        """Broadcast crossing event to WebSocket clients.
        
        Args:
            crossing_event: Crossing event dictionary
        """
        message = {
            "type": "crossing",
            "data": crossing_event,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.ws_manager.broadcast(json.dumps(message))
    
    def set_status(self, status: str):
        """Update system status."""
        self.system_status = status
        logger.info(f"System status updated to: {status}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the API server.
        
        Args:
            host: Host address to bind to
            port: Port number to bind to
            **kwargs: Additional uvicorn parameters
        """
        logger.info(f"Starting API server on {host}:{port}")
        
        # Default uvicorn config optimized for production
        config = {
            "host": host,
            "port": port,
            "log_level": "info",
            "access_log": True,
            "loop": "asyncio",
            **kwargs
        }
        
        uvicorn.run(self.app, **config)
