#!/usr/bin/env python3
"""
Enhanced People Counter System with Multi-Object Detection and Interval-Based Activity Classification

Features:
- Multi-object detection (person, laptop, cell phone, book, bottle, cup)
- Blue dots visualization for objects
- 5-second interval-based activity classification
- Person-object interaction detection
- Real-time tracking and counting
- REST API and WebSocket support
"""

import cv2
import numpy as np
import logging
import time
import yaml
import argparse
import signal
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import sys

from activity_classifier import ActivityClassifier
from counter import PeopleCounter
from tracker import TrackerManager
from detector import PersonDetector
from db import CounterDB
from api import PeopleCounterAPI
from interval_classifier import IntervalActivityClassifier
from capture import VideoCapture

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Setup logging
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = logs_dir / log_file
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

class PeopleCounterSystem:
    """Main system class that orchestrates all components."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the people counter system."""
        self.config = self._load_config(config_path)
        self.running = False
        self.stats = {
            'start_time': None,
            'frames_processed': 0,
            'total_detections': 0,
            'total_crossings': 0
        }
        
        # Initialize components
        self.detector = None
        self.tracker = None
        self.counter = None
        self.activity_classifier = None
        self.interval_classifier = None
        self.db = None
        self.api_server = None
        self.capture = None
        
        # Setup logging
        setup_logging(
            self.config.get('logging', {}).get('level', 'INFO'),
            self.config.get('logging', {}).get('file')
        )
        
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config file: {e}")
            raise
    
    def initialize(self):
        """Initialize all system components."""
        self.logger.info("Initializing People Counter System...")
        
        try:
            # Initialize detector with object detection enabled
            detector_config = self.config.get('detector', {})
            detector_config['detect_objects'] = True  # Enable multi-object detection
            self.detector = PersonDetector(detector_config)
            
            # Initialize tracker
            tracker_config = self.config.get('tracker', {})
            self.tracker = TrackerManager(tracker_config)
            
            # Initialize counter
            counter_config = self.config.get('counter', {})
            self.counter = PeopleCounter(counter_config)
            
            # Initialize activity classifier
            activity_config = self.config.get('activity_classifier', {})
            self.activity_classifier = ActivityClassifier(activity_config)
            
            # Initialize interval-based activity classifier
            interval_config = self.config.get('interval_classifier', {})
            self.interval_classifier = IntervalActivityClassifier(interval_config)
            
            # Initialize database
            db_config = self.config.get('database', {})
            if db_config.get('enabled', False):
                self.db = CounterDB(db_config)
            
            # Initialize API server
            api_config = self.config.get('api', {})
            if api_config.get('enabled', True):
                self.api_server = PeopleCounterAPI(
                    host=api_config.get('host', '0.0.0.0'),
                    port=api_config.get('port', 8000)
                )
                # Start API server in background
                api_thread = threading.Thread(target=self.api_server.start, daemon=True)
                api_thread.start()
            
            # Initialize video capture
            capture_config = self.config.get('capture', {})
            self.capture = VideoCapture(capture_config)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the detection and tracking pipeline."""
        try:
            # Detect objects and people
            detections = self.detector.detect(frame)
            self.stats['total_detections'] += len(detections)
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Update counter
            crossings = self.counter.update(tracked_objects)
            self.stats['total_crossings'] += len(crossings)
            
            # Classify activities for people
            for obj in tracked_objects:
                if obj.class_name == "person":
                    # Get pose-based activity
                    activity = self.activity_classifier.classify(frame, obj.bbox)
                    obj.activity = activity
            
            # Update interval classifier
            current_time = time.time()
            interval_result = self.interval_classifier.update(tracked_objects, current_time)
            
            # Visualize results
            vis_frame = self._visualize_results(
                frame, tracked_objects, crossings, interval_result
            )
            
            # Update API server data
            if self.api_server:
                self.api_server.update_data({
                    'counts': self.counter.get_counts(),
                    'tracked_objects': [
                        {
                            'id': obj.id,
                            'class_name': obj.class_name,
                            'bbox': obj.bbox.tolist(),
                            'confidence': getattr(obj, 'confidence', 0.0),
                            'activity': getattr(obj, 'activity', 'unknown')
                        }
                        for obj in tracked_objects
                    ],
                    'crossings': [
                        {
                            'object_id': c.object_id,
                            'direction': c.direction,
                            'timestamp': c.timestamp.isoformat()
                        }
                        for c in crossings
                    ],
                    'interval_summary': interval_result
                })
            
            # Save to database
            if self.db:
                for crossing in crossings:
                    self.db.log_crossing(crossing)
            
            self.stats['frames_processed'] += 1
            return vis_frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame
    
    def _visualize_results(self, frame: np.ndarray, tracked_objects, crossings, interval_result) -> np.ndarray:
        """Visualize detection and tracking results on frame."""
        vis_frame = frame.copy()
        
        # Draw crossing lines
        self.counter.draw_lines(vis_frame)
        
        # Draw tracked objects
        for obj in tracked_objects:
            if obj.class_name == "person":
                # Draw bounding box for persons
                x1, y1, x2, y2 = obj.bbox.astype(int)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw person ID and activity
                label = f"Person {obj.id}"
                if hasattr(obj, 'activity') and obj.activity:
                    label += f" - {obj.activity}"
                
                cv2.putText(vis_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Draw blue dot for objects
                center_x = int((obj.bbox[0] + obj.bbox[2]) / 2)
                center_y = int((obj.bbox[1] + obj.bbox[3]) / 2)
                cv2.circle(vis_frame, (center_x, center_y), 8, (255, 0, 0), -1)
                
                # Draw object label
                label = f"{obj.class_name}"
                cv2.putText(vis_frame, label, (center_x + 15, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw counts
        counts = self.counter.get_counts()
        count_text = f"In: {counts['in']} | Out: {counts['out']} | Current: {counts['occupancy']}"
        cv2.putText(vis_frame, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw interval summary
        if interval_result and interval_result.get('summary'):
            summary = interval_result['summary']
            y_offset = 60
            
            # Activity summary
            if summary.get('dominant_activity'):
                activity_text = f"Activity: {summary['dominant_activity']}"
                cv2.putText(vis_frame, activity_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
            
            # Object summary
            if summary.get('dominant_objects'):
                objects_text = f"Objects: {', '.join(summary['dominant_objects'])}"
                cv2.putText(vis_frame, objects_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
            
            # Interactions
            if summary.get('interactions'):
                interactions_text = f"Interactions: {len(summary['interactions'])}"
                cv2.putText(vis_frame, interactions_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw crossings
        for crossing in crossings:
            direction_color = (0, 255, 0) if crossing.direction == "in" else (0, 0, 255)
            cv2.putText(vis_frame, f"Crossing: {crossing.direction}",
                       (10, vis_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, direction_color, 2)
        
        return vis_frame
    
    def run(self):
        """Main processing loop."""
        self.logger.info("Starting People Counter System")
        self.running = True
        self.stats['start_time'] = time.time()
        
        try:
            # Initialize video capture
            if not self.capture.start():
                self.logger.error("Failed to start video capture")
                return
            
            self.logger.info("Processing started - Press 'q' to quit")
            
            while self.running:
                # Read frame
                ret, frame = self.capture.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("People Counter", processed_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                
                # Print stats periodically
                if self.stats['frames_processed'] % 100 == 0:
                    self._print_stats()
        
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system and cleanup resources."""
        self.logger.info("Stopping People Counter System")
        self.running = False
        
        # Stop video capture
        if self.capture:
            self.capture.stop()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Print final stats
        self._print_final_stats()
        
        self.logger.info("Processing loop stopped")
    
    def _print_stats(self):
        """Print current statistics."""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            fps = self.stats['frames_processed'] / runtime if runtime > 0 else 0
            
            self.logger.info(f"Stats: Frames={self.stats['frames_processed']}, "
                           f"FPS={fps:.1f}, Detections={self.stats['total_detections']}, "
                           f"Crossings={self.stats['total_crossings']}")
    
    def _print_final_stats(self):
        """Print final statistics."""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            avg_fps = self.stats['frames_processed'] / runtime if runtime > 0 else 0
            counts = self.counter.get_counts() if self.counter else {'in': 0, 'out': 0, 'occupancy': 0}
            
            self.logger.info("Final Statistics:")
            self.logger.info(f"  Runtime: {runtime:.1f} seconds")
            self.logger.info(f"  Frames processed: {self.stats['frames_processed']}")
            self.logger.info(f"  Average FPS: {avg_fps:.1f}")
            self.logger.info(f"  Total detections: {self.stats['total_detections']}")
            self.logger.info(f"  Total crossings: {self.stats['total_crossings']}")
            self.logger.info(f"  Final counts: In={counts['in']}, Out={counts['out']}, Occupancy={counts['occupancy']}")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logging.getLogger(__name__).info(f"Received signal {signum}")
    sys.exit(0)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced People Counter System")
    parser.add_argument("--config", default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create and initialize system
        system = PeopleCounterSystem(args.config)
        system.initialize()
        
        # Run system
        system.run()
        
    except Exception as e:
        logging.error(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
