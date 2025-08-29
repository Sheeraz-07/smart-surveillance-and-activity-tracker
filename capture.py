"""
OpenCV+FFmpeg wrapper for RTSP and MP4 capture with robust reconnect/backoff logic.
Handles connection failures, latency drops, and automatic recovery.
"""
import logging
import time
import threading
from typing import Optional, Tuple, Callable
import cv2
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoCapture:
    """Robust video capture with reconnection and error handling."""
    
    def __init__(self,
                 source: str,
                 buffer_size: int = 1,
                 reconnect_delay: float = 5.0,
                 max_reconnect_attempts: int = -1,  # -1 for infinite
                 timeout_seconds: float = 10.0,
                 drop_threshold_ms: float = 1000.0,  # Drop frames if latency > 1s
                 fps_target: Optional[float] = None):
        """Initialize video capture.
        
        Args:
            source: Video source (RTSP URL, file path, or camera index)
            buffer_size: OpenCV buffer size (1 for real-time, larger for stability)
            reconnect_delay: Seconds to wait before reconnection attempt
            max_reconnect_attempts: Maximum reconnection attempts (-1 for infinite)
            timeout_seconds: Timeout for connection attempts
            drop_threshold_ms: Drop frames if older than this (milliseconds)
            fps_target: Target FPS for frame rate limiting
        """
        self.source = source
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.timeout_seconds = timeout_seconds
        self.drop_threshold_ms = drop_threshold_ms
        self.fps_target = fps_target
        
        # State variables
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.reconnect_count = 0
        self.last_frame_time = 0.0
        self.frame_count = 0
        
        # Threading
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_timestamp = 0.0
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'reconnections': 0,
            'last_fps': 0.0,
            'connection_time': 0.0
        }
        
        logger.info(f"Initialized capture for source: {source}")
    
    def _create_capture(self) -> bool:
        """Create OpenCV VideoCapture object with optimized settings.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Release existing capture
            if self.cap is not None:
                self.cap.release()
            
            # Create new capture
            self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Configure capture settings for performance
            # Reduce buffer size for real-time processing
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            # Set timeout for network streams
            if isinstance(self.source, str) and self.source.startswith(('rtsp://', 'http://')):
                # RTSP/HTTP specific settings
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.timeout_seconds * 1000))
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.timeout_seconds * 1000))
                
                # Use FFmpeg backend for better RTSP support
                if cv2.CAP_FFMPEG in [cv2.CAP_FFMPEG]:  # Check if available
                    logger.debug("Using FFmpeg backend for network stream")
            
            # Test frame read
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to read test frame")
                return False
            
            # Get stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Connected to {self.source}: {width}x{height} @ {fps:.1f} FPS")
            
            self.is_connected = True
            self.stats['connection_time'] = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Exception creating capture: {e}")
            return False
    
    def connect(self) -> bool:
        """Connect to video source with retry logic.
        
        Returns:
            True if connected successfully
        """
        self.reconnect_count = 0
        
        while (self.max_reconnect_attempts == -1 or 
               self.reconnect_count < self.max_reconnect_attempts):
            
            if self._stop_event.is_set():
                return False
            
            logger.info(f"Attempting connection to {self.source} (attempt {self.reconnect_count + 1})")
            
            if self._create_capture():
                self.reconnect_count = 0
                return True
            
            self.reconnect_count += 1
            self.stats['reconnections'] += 1
            
            if (self.max_reconnect_attempts != -1 and 
                self.reconnect_count >= self.max_reconnect_attempts):
                logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                break
            
            logger.warning(f"Connection failed, retrying in {self.reconnect_delay}s...")
            time.sleep(self.reconnect_delay)
        
        return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame with error handling.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_connected or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.warning("Failed to read frame, connection may be lost")
                self.is_connected = False
                return False, None
            
            # Check for frame staleness (important for RTSP streams)
            current_time = time.time()
            if self.drop_threshold_ms > 0:
                frame_age_ms = (current_time - self.last_frame_time) * 1000
                if frame_age_ms > self.drop_threshold_ms and self.frame_count > 0:
                    self.stats['frames_dropped'] += 1
                    logger.debug(f"Dropped stale frame (age: {frame_age_ms:.1f}ms)")
            
            self.last_frame_time = current_time
            self.frame_count += 1
            self.stats['frames_captured'] += 1
            
            # Calculate FPS
            if self.frame_count % 30 == 0:  # Update every 30 frames
                if hasattr(self, '_fps_start_time'):
                    elapsed = current_time - self._fps_start_time
                    self.stats['last_fps'] = 30.0 / elapsed if elapsed > 0 else 0.0
                self._fps_start_time = current_time
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Exception reading frame: {e}")
            self.is_connected = False
            return False, None
    
    def start_threaded_capture(self, frame_callback: Optional[Callable] = None):
        """Start threaded frame capture for continuous operation.
        
        Args:
            frame_callback: Optional callback function for each frame
        """
        def capture_loop():
            """Main capture loop running in separate thread."""
            logger.info("Started threaded capture loop")
            
            while not self._stop_event.is_set():
                if not self.is_connected:
                    if not self.connect():
                        logger.error("Failed to establish connection, stopping capture")
                        break
                
                ret, frame = self.read_frame()
                
                if ret and frame is not None:
                    # Store latest frame
                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                        self._frame_timestamp = time.time()
                    
                    # Call callback if provided
                    if frame_callback:
                        try:
                            frame_callback(frame)
                        except Exception as e:
                            logger.error(f"Frame callback error: {e}")
                
                elif not self.is_connected:
                    # Connection lost, attempt reconnection
                    logger.warning("Connection lost, attempting reconnection...")
                    time.sleep(self.reconnect_delay)
                
                # FPS limiting
                if self.fps_target:
                    time.sleep(1.0 / self.fps_target)
            
            logger.info("Capture loop stopped")
        
        # Start capture thread
        self._capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self._capture_thread.start()
    
    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Get the most recent frame from threaded capture.
        
        Returns:
            Tuple of (has_frame, frame, timestamp)
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy(), self._frame_timestamp
            return False, None, 0.0
    
    def stop(self):
        """Stop capture and cleanup resources."""
        logger.info("Stopping video capture...")
        
        # Signal stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if hasattr(self, '_capture_thread') and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)
        
        # Release capture
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.is_connected = False
        logger.info("Video capture stopped")
    
    def get_stats(self) -> dict:
        """Get capture statistics."""
        stats = self.stats.copy()
        stats.update({
            'is_connected': self.is_connected,
            'reconnect_count': self.reconnect_count,
            'frame_count': self.frame_count,
            'source': self.source
        })
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

class MultiSourceCapture:
    """Capture manager for multiple video sources."""
    
    def __init__(self):
        """Initialize multi-source capture manager."""
        self.captures: dict = {}
        self.active_source: Optional[str] = None
        
    def add_source(self, name: str, source: str, **kwargs) -> bool:
        """Add a video source.
        
        Args:
            name: Unique name for the source
            source: Video source path/URL
            **kwargs: Additional VideoCapture parameters
            
        Returns:
            True if added successfully
        """
        try:
            capture = VideoCapture(source, **kwargs)
            self.captures[name] = capture
            logger.info(f"Added video source '{name}': {source}")
            return True
        except Exception as e:
            logger.error(f"Failed to add source '{name}': {e}")
            return False
    
    def set_active_source(self, name: str) -> bool:
        """Set the active video source.
        
        Args:
            name: Name of source to activate
            
        Returns:
            True if activated successfully
        """
        if name not in self.captures:
            logger.error(f"Source '{name}' not found")
            return False
        
        # Stop current active source
        if self.active_source and self.active_source in self.captures:
            self.captures[self.active_source].stop()
        
        # Start new source
        capture = self.captures[name]
        if capture.connect():
            self.active_source = name
            logger.info(f"Activated source '{name}'")
            return True
        else:
            logger.error(f"Failed to connect to source '{name}'")
            return False
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get frame from active source.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.active_source or self.active_source not in self.captures:
            return False, None
        
        return self.captures[self.active_source].read_frame()
    
    def stop_all(self):
        """Stop all video sources."""
        for capture in self.captures.values():
            capture.stop()
        self.active_source = None
        logger.info("Stopped all video sources")
