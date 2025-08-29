"""
Main orchestration module for people counting system.
Handles configuration loading, processing loop, and API server startup.
"""
import logging
import asyncio
import threading
import time
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import cv2
import numpy as np

from capture import VideoCapture
from detector import PersonDetector
from tracker import TrackerManager
from counter import PeopleCounter, CrossingLine
from db import CounterDB
from api import PeopleCounterAPI

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("opencv").setLevel(logging.WARNING)

class PeopleCounterSystem:
    """Main people counting system orchestrator."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the people counting system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Setup logging
        setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file=self.config.get('logging', {}).get('file', 'logs/people_counter.log')
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing People Counter System")
        
        # Initialize components
        self.capture: Optional[VideoCapture] = None
        self.detector: Optional[PersonDetector] = None
        self.tracker: Optional[TrackerManager] = None
        self.counter: Optional[PeopleCounter] = None
        self.db: Optional[CounterDB] = None
        self.api: Optional[PeopleCounterAPI] = None
        
        # State
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.api_thread: Optional[threading.Thread] = None
        self.show_window = True  # Enable camera window display
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detections_count': 0,
            'crossings_count': 0,
            'start_time': time.time(),
            'fps': 0.0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        default_config = {
            'video': {
                'source': 0,  # Default to webcam
                'buffer_size': 1,
                'reconnect_delay': 5.0,
                'timeout_seconds': 10.0
            },
            'detector': {
                'model_path': 'models/yolo11n.onnx',
                'imgsz': 512,
                'conf_threshold': 0.35,
                'nms_threshold': 0.45,
                'use_ultralytics': True,
                'num_threads': 4
            },
            'tracker': {
                'type': 'norfair',
                'distance_threshold': 0.7,
                'hit_counter_max': 15,
                'initialization_delay': 3
            },
            'counter': {
                'line_start': [320, 200],
                'line_end': [320, 400],
                'debounce_frames': 15,
                'debounce_time': 2.0,
                'min_track_length': 3
            },
            'database': {
                'path': 'counts.db'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'enable': True
            },
            'display': {
                'show_window': True,
                'window_name': 'People Counter',
                'window_width': 800,
                'window_height': 600
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/people_counter.log'
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                
                # Merge with defaults
                def merge_dict(default, user):
                    for key, value in user.items():
                        if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                            merge_dict(default[key], value)
                        else:
                            default[key] = value
                
                merge_dict(default_config, user_config)
                return default_config
                
            except Exception as e:
                print(f"Error loading config from {self.config_path}: {e}")
                print("Using default configuration")
        else:
            print(f"Config file {self.config_path} not found, using defaults")
        
        return default_config
    
    async def initialize(self):
        """Initialize all system components."""
        self.logger.info("Initializing system components...")
        
        try:
            # Initialize database
            self.db = CounterDB(self.config['database']['path'])
            await self.db.init_db()
            
            # Initialize detector
            detector_config = self.config['detector']
            self.detector = PersonDetector(
                model_path=detector_config['model_path'],
                imgsz=detector_config['imgsz'],
                conf_threshold=detector_config['conf_threshold'],
                nms_threshold=detector_config['nms_threshold'],
                use_ultralytics=detector_config['use_ultralytics'],
                num_threads=detector_config['num_threads']
            )
            
            # Initialize tracker
            tracker_config = self.config['tracker']
            self.tracker = TrackerManager(
                tracker_type=tracker_config['type'],
                distance_threshold=tracker_config.get('distance_threshold', 0.7),
                hit_counter_max=tracker_config.get('hit_counter_max', 15),
                initialization_delay=tracker_config.get('initialization_delay', 3)
            )
            
            # Initialize counter
            counter_config = self.config['counter']
            counting_line = CrossingLine(
                start_point=tuple(counter_config['line_start']),
                end_point=tuple(counter_config['line_end']),
                name="main_line"
            )
            
            self.counter = PeopleCounter(
                counting_line=counting_line,
                debounce_frames=counter_config['debounce_frames'],
                debounce_time=counter_config['debounce_time'],
                min_track_length=counter_config['min_track_length']
            )
            
            # Initialize video capture
            video_config = self.config['video']
            self.capture = VideoCapture(
                source=video_config['source'],
                buffer_size=video_config['buffer_size'],
                reconnect_delay=video_config['reconnect_delay'],
                timeout_seconds=video_config['timeout_seconds']
            )
            
            # Initialize API if enabled
            if self.config['api']['enable']:
                self.api = PeopleCounterAPI(self.config['database']['path'])
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def _draw_visualizations(self, frame: np.ndarray, detections, tracked_objects) -> np.ndarray:
        """Draw detection boxes, tracking info, and counting line on frame."""
        display_frame = frame.copy()
        
        # Draw counting line
        line_info = self.counter.get_line_info()
        start_pt = tuple(map(int, line_info['start_point']))
        end_pt = tuple(map(int, line_info['end_point']))
        cv2.line(display_frame, start_pt, end_pt, (0, 255, 0), 3)
        
        # Draw detection boxes
        for det in detections:
            x1, y1, x2, y2 = det.xyxy.astype(int)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f'{det.confidence:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw tracked objects
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.xyxy.astype(int)
            center = tuple(map(int, obj.center))
            
            # Different color for tracked objects
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(display_frame, center, 5, (0, 255, 255), -1)
            cv2.putText(display_frame, f'ID:{obj.track_id}', 
                       (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw counts
        counts = self.counter.get_counts()
        info_text = [
            f"In: {counts['in']}",
            f"Out: {counts['out']}", 
            f"Occupancy: {counts['occupancy']}",
            f"FPS: {self.stats['fps']:.1f}",
            f"Detections: {len(detections)}",
            f"Tracks: {len(tracked_objects)}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(display_frame, text, (10, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame

    async def _processing_loop(self):
        """Main processing loop for video frames."""
        self.logger.info("Starting processing loop")
        
        # Connect to video source
        if not self.capture.connect():
            self.logger.error("Failed to connect to video source")
            return
        
        # Initialize display window if enabled
        display_config = self.config.get('display', {})
        show_window = display_config.get('show_window', True)
        window_name = display_config.get('window_name', 'People Counter')
        
        if show_window:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            self.logger.info(f"Camera window '{window_name}' initialized")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while self.running:
            try:
                # Read frame
                ret, frame = self.capture.read_frame()
                
                if not ret or frame is None:
                    if not self.capture.is_connected:
                        self.logger.warning("Video connection lost, attempting reconnection...")
                        if not self.capture.connect():
                            self.logger.error("Failed to reconnect, stopping processing")
                            break
                    continue
                
                # Detect persons
                try:
                    detections = self.detector.detect(frame)
                    self.stats['detections_count'] += len(detections)
                except Exception as e:
                    self.logger.warning(f"Detection failed: {e}")
                    detections = []
                
                # Update tracker
                try:
                    tracked_objects = self.tracker.update(detections)
                except Exception as e:
                    self.logger.warning(f"Tracker update failed: {e}")
                    tracked_objects = []
                
                # Update counter
                try:
                    crossings = self.counter.update(tracked_objects)
                except Exception as e:
                    self.logger.warning(f"Counter update failed: {e}")
                    crossings = []
                
                # Log crossings to database
                for crossing in crossings:
                    await self.db.log_crossing(
                        track_id=crossing['track_id'],
                        direction=crossing['direction'],
                        confidence=crossing['confidence'],
                        x=crossing['position'][0],
                        y=crossing['position'][1],
                        frame_number=crossing.get('frame_number')
                    )
                    
                    # Broadcast to API clients
                    if self.api:
                        await self.api.broadcast_crossing(crossing)
                
                self.stats['crossings_count'] += len(crossings)
                
                # Update API with current counts
                if self.api:
                    counts = self.counter.get_counts()
                    await self.api.update_counts(counts, self.stats['fps'])
                
                # Display frame with visualizations
                if show_window:
                    display_frame = self._draw_visualizations(frame, detections, tracked_objects)
                    
                    # Resize for display if needed
                    window_width = display_config.get('window_width', 800)
                    window_height = display_config.get('window_height', 600)
                    h, w = display_frame.shape[:2]
                    if w > window_width or h > window_height:
                        scale = min(window_width/w, window_height/h)
                        new_w, new_h = int(w*scale), int(h*scale)
                        display_frame = cv2.resize(display_frame, (new_w, new_h))
                    
                    cv2.imshow(window_name, display_frame)
                    
                    # Handle window events
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC key
                        self.logger.info("Window close requested")
                        self.stop()
                        break
                
                # Update statistics
                frame_count += 1
                self.stats['frames_processed'] = frame_count
                
                # Calculate FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start_time
                    self.stats['fps'] = 30.0 / elapsed if elapsed > 0 else 0.0
                    fps_start_time = time.time()
                    
                    self.logger.debug(f"Processing: {self.stats['fps']:.1f} FPS, "
                                    f"Detections: {len(detections)}, "
                                    f"Tracks: {len(tracked_objects)}, "
                                    f"Crossings: {len(crossings)}")
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.001)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)  # Wait before retrying
        
        # Clean up display window
        if show_window:
            cv2.destroyAllWindows()
        
        self.logger.info("Processing loop stopped")
    
    def _run_api_server(self):
        """Run API server in separate thread."""
        if not self.api:
            return
        
        try:
            api_config = self.config['api']
            self.api.run(
                host=api_config['host'],
                port=api_config['port'],
                log_level="info"
            )
        except Exception as e:
            self.logger.error(f"API server error: {e}")
    
    async def start(self):
        """Start the people counting system."""
        self.logger.info("Starting People Counter System")
        
        # Initialize components
        await self.initialize()
        
        self.running = True
        
        # Start API server in separate thread
        if self.api:
            self.api_thread = threading.Thread(target=self._run_api_server, daemon=True)
            self.api_thread.start()
            self.logger.info(f"API server started on {self.config['api']['host']}:{self.config['api']['port']}")
        
        # Start processing loop
        await self._processing_loop()
    
    def stop(self):
        """Stop the people counting system."""
        self.logger.info("Stopping People Counter System")
        
        self.running = False
        
        # Stop video capture
        if self.capture:
            self.capture.stop()
        
        # Print final statistics
        elapsed = time.time() - self.stats['start_time']
        self.logger.info(f"Final Statistics:")
        self.logger.info(f"  Runtime: {elapsed:.1f} seconds")
        self.logger.info(f"  Frames processed: {self.stats['frames_processed']}")
        self.logger.info(f"  Average FPS: {self.stats['frames_processed'] / elapsed:.1f}")
        self.logger.info(f"  Total detections: {self.stats['detections_count']}")
        self.logger.info(f"  Total crossings: {self.stats['crossings_count']}")
        
        if self.counter:
            counts = self.counter.get_counts()
            self.logger.info(f"  Final counts: In={counts['in']}, Out={counts['out']}, Occupancy={counts['occupancy']}")

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="People Counter System")
    parser.add_argument("--config", "-c", default="config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Create system instance
    system = PeopleCounterSystem(config_path=args.config)
    
    try:
        # Start the system
        await system.start()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
        logging.exception("System error")
    finally:
        system.stop()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
