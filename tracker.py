"""
Multi-object tracking module with Norfair default and ByteTrack fallback.
Provides a unified interface for different tracking backends.
"""
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    import norfair
    NORFAIR_AVAILABLE = True
except ImportError:
    NORFAIR_AVAILABLE = False

try:
    import torch
    from yolox.tracker.byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False

from detector import Detection

logger = logging.getLogger(__name__)

@dataclass
class TrackedObject:
    """Unified tracked object representation."""
    track_id: int
    xyxy: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    age: int = 0
    
    @property
    def center(self) -> tuple:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

class BaseTracker:
    """Base class for tracking adapters."""
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with new detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of TrackedObject instances
        """
        raise NotImplementedError

class NorfairTracker(BaseTracker):
    """Norfair tracking adapter."""
    
    def __init__(self, 
                 distance_threshold: float = 0.7,
                 hit_counter_max: int = 15,
                 initialization_delay: int = 3,
                 distance_function: str = "iou"):
        """Initialize Norfair tracker.
        
        Args:
            distance_threshold: Maximum distance for track association
            hit_counter_max: Frames to keep track alive without detection
            initialization_delay: Frames before track is confirmed
            distance_function: Distance metric ('iou' or 'euclidean')
        """
        if not NORFAIR_AVAILABLE:
            raise ImportError("Norfair not available. Install with: pip install norfair")
        
        self.distance_threshold = distance_threshold
        self.hit_counter_max = hit_counter_max
        self.initialization_delay = initialization_delay
        
        # Create custom distance function that handles Detection objects properly
        def custom_distance_function(detection, tracked_object):
            """Custom distance function that converts Detection objects to proper format."""
            try:
                # Extract bounding boxes from Detection objects
                if hasattr(detection, 'points') and len(detection.points) >= 2:
                    det_x1, det_y1 = detection.points[0]
                    det_x2, det_y2 = detection.points[1]
                    det_bbox = np.array([det_x1, det_y1, det_x2, det_y2], dtype=np.float32)
                else:
                    return float('inf')  # Invalid detection
                
                if hasattr(tracked_object, 'last_detection') and tracked_object.last_detection is not None:
                    if hasattr(tracked_object.last_detection, 'points') and len(tracked_object.last_detection.points) >= 2:
                        track_x1, track_y1 = tracked_object.last_detection.points[0]
                        track_x2, track_y2 = tracked_object.last_detection.points[1]
                        track_bbox = np.array([track_x1, track_y1, track_x2, track_y2], dtype=np.float32)
                    else:
                        return float('inf')  # Invalid tracked object
                else:
                    return float('inf')  # No last detection
                
                # Calculate IoU distance
                if distance_function == "iou":
                    return self._calculate_iou_distance(det_bbox, track_bbox)
                else:
                    return self._calculate_euclidean_distance(det_bbox, track_bbox)
                    
            except Exception as e:
                logger.warning(f"Distance calculation failed: {e}")
                return float('inf')
        
        self.distance_function = custom_distance_function
        
        self.tracker = norfair.Tracker(
            distance_function=self.distance_function,
            distance_threshold=distance_threshold,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
        )
        
        logger.info(f"Initialized Norfair tracker with custom {distance_function} distance")
    
    def _calculate_iou_distance(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU-based distance between two bounding boxes."""
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # Calculate intersection
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 1.0  # No intersection, maximum distance
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            
            # Calculate union
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            if union <= 0:
                return 1.0
            
            iou = intersection / union
            return 1.0 - iou  # Convert IoU to distance (0 = perfect match, 1 = no match)
            
        except Exception:
            return 1.0
    
    def _calculate_euclidean_distance(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate Euclidean distance between bounding box centers."""
        try:
            # Calculate centers
            center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
            center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(center1 - center2)
            
            # Normalize by image diagonal (assuming 640x640 for now)
            max_distance = np.sqrt(640**2 + 640**2)
            return distance / max_distance
            
        except Exception:
            return 1.0
    
    def _detections_to_norfair(self, detections: List[Detection]) -> List[norfair.Detection]:
        """Convert Detection objects to Norfair format."""
        norfair_detections = []
        
        for det in detections:
            try:
                # Convert xyxy to points format for Norfair
                x1, y1, x2, y2 = det.xyxy.astype(np.float32)
                
                # Norfair expects points as numpy array with shape (n_points, 2)
                # For bounding boxes, we use 2 points: top-left and bottom-right
                points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
                
                # Ensure confidence is a float
                confidence = float(det.confidence)
                
                norfair_det = norfair.Detection(
                    points=points,
                    scores=np.array([confidence, confidence], dtype=np.float32)
                )
                norfair_detections.append(norfair_det)
                
            except Exception as e:
                logger.warning(f"Failed to convert detection to Norfair format: {e}")
                continue
        
        return norfair_detections
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with new detections."""
        try:
            if not detections:
                # Update with empty detections to age existing tracks
                tracked_objects = self.tracker.update(detections=[])
                return self._convert_norfair_results(tracked_objects)
                
            # Convert to Norfair format
            norfair_detections = self._detections_to_norfair(detections)
            
            # Skip update if no valid detections after conversion
            if not norfair_detections:
                tracked_objects = self.tracker.update(detections=[])
                return self._convert_norfair_results(tracked_objects)
            
            # Update tracker with converted detections
            tracked_objects = self.tracker.update(detections=norfair_detections)
            
            return self._convert_norfair_results(tracked_objects)
            
        except Exception as e:
            logger.error(f"Error in Norfair tracker update: {e}")
            # Return empty list on error to prevent system crash
            return []
    
    def _convert_norfair_results(self, tracked_objects) -> List[TrackedObject]:
        """Convert Norfair tracked objects to unified format."""
        results = []
        
        for obj in tracked_objects:
            try:
                if obj.last_detection is not None and len(obj.last_detection.points) >= 2:
                    # Extract bounding box from points
                    points = obj.last_detection.points
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                    # Ensure coordinates are valid
                    if not all(np.isfinite([x1, y1, x2, y2])):
                        logger.warning(f"Invalid coordinates for track {obj.id}")
                        continue
                    
                    # Get confidence from scores
                    confidence = float(np.mean(obj.last_detection.scores))
                    
                    tracked_obj = TrackedObject(
                        track_id=obj.id,
                        xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                        confidence=confidence,
                        age=obj.age
                    )
                    results.append(tracked_obj)
                    
            except Exception as e:
                logger.warning(f"Failed to convert Norfair result for track {getattr(obj, 'id', 'unknown')}: {e}")
                continue
        
        return results

class ByteTrackAdapter(BaseTracker):
    """ByteTrack tracking adapter."""
    
    def __init__(self,
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30):
        """Initialize ByteTrack adapter.
        
        Args:
            track_thresh: Detection threshold for tracking
            track_buffer: Frames to buffer for re-identification
            match_thresh: Matching threshold for association
            frame_rate: Video frame rate
        """
        if not BYTETRACK_AVAILABLE:
            raise ImportError("ByteTrack not available. Install PyTorch and yolox")
        
        # ByteTracker arguments
        class Args:
            def __init__(self):
                self.track_thresh = track_thresh
                self.track_buffer = track_buffer
                self.match_thresh = match_thresh
                self.mot20 = False
        
        self.args = Args()
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)
        self.frame_id = 0
        
        logger.info("Initialized ByteTrack adapter")
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with new detections."""
        self.frame_id += 1
        
        if not detections:
            # Update with empty detections
            online_targets = self.tracker.update(
                output_results=np.empty((0, 5)),
                img_info=(640, 640),  # dummy size
                img_size=(640, 640)
            )
        else:
            # Convert detections to ByteTrack format
            det_array = np.zeros((len(detections), 5))
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det.xyxy
                det_array[i] = [x1, y1, x2, y2, det.confidence]
            
            # Update tracker
            online_targets = self.tracker.update(
                output_results=det_array,
                img_info=(640, 640),  # dummy size
                img_size=(640, 640)
            )
        
        # Convert to unified format
        results = []
        for track in online_targets:
            if hasattr(track, 'tlbr') and hasattr(track, 'track_id'):
                x1, y1, x2, y2 = track.tlbr
                
                tracked_obj = TrackedObject(
                    track_id=int(track.track_id),
                    xyxy=np.array([x1, y1, x2, y2]),
                    confidence=float(track.score) if hasattr(track, 'score') else 0.5,
                    age=getattr(track, 'frame_id', 0)
                )
                results.append(tracked_obj)
        
        return results

class SimpleTracker(BaseTracker):
    """Simple centroid-based tracker as fallback."""
    
    def __init__(self, max_distance: float = 50.0, max_age: int = 10):
        """Initialize simple tracker.
        
        Args:
            max_distance: Maximum distance for track association
            max_age: Maximum frames to keep track without detection
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
        
        logger.info("Initialized simple centroid tracker")
    
    def _calculate_distance(self, center1: tuple, center2: tuple) -> float:
        """Calculate Euclidean distance between centers."""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with new detections."""
        # Get detection centers
        det_centers = [det.center for det in detections]
        
        # Update existing tracks
        updated_tracks = {}
        used_detections = set()
        
        for track_id, track in self.tracks.items():
            best_match = None
            best_distance = float('inf')
            
            # Find closest detection
            for i, center in enumerate(det_centers):
                if i in used_detections:
                    continue
                
                distance = self._calculate_distance(track.center, center)
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                # Update track with matched detection
                det = detections[best_match]
                track.xyxy = det.xyxy
                track.confidence = det.confidence
                track.age = 0
                updated_tracks[track_id] = track
                used_detections.add(best_match)
            else:
                # Age track
                track.age += 1
                if track.age < self.max_age:
                    updated_tracks[track_id] = track
        
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_track = TrackedObject(
                    track_id=self.next_id,
                    xyxy=det.xyxy,
                    confidence=det.confidence,
                    age=0
                )
                updated_tracks[self.next_id] = new_track
                self.next_id += 1
        
        self.tracks = updated_tracks
        return list(self.tracks.values())

class TrackerManager:
    """Manager for different tracking backends."""
    
    def __init__(self, tracker_type: str = "norfair", **kwargs):
        """Initialize tracker manager.
        
        Args:
            tracker_type: Type of tracker ('norfair', 'bytetrack', 'simple')
            **kwargs: Tracker-specific parameters
        """
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker(tracker_type, **kwargs)
    
    def _create_tracker(self, tracker_type: str, **kwargs) -> BaseTracker:
        """Create tracker instance based on type."""
        if tracker_type == "norfair" and NORFAIR_AVAILABLE:
            return NorfairTracker(**kwargs)
        elif tracker_type == "bytetrack" and BYTETRACK_AVAILABLE:
            return ByteTrackAdapter(**kwargs)
        elif tracker_type == "simple":
            return SimpleTracker(**kwargs)
        else:
            # Fallback logic
            if NORFAIR_AVAILABLE:
                logger.warning(f"Tracker '{tracker_type}' not available, using Norfair")
                return NorfairTracker(**kwargs)
            elif BYTETRACK_AVAILABLE:
                logger.warning(f"Tracker '{tracker_type}' not available, using ByteTrack")
                return ByteTrackAdapter(**kwargs)
            else:
                logger.warning(f"Advanced trackers not available, using simple tracker")
                return SimpleTracker(**kwargs)
    
    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Update tracker with new detections."""
        return self.tracker.update(detections)
    
    def get_info(self) -> Dict[str, Any]:
        """Get tracker information."""
        return {
            'type': self.tracker_type,
            'class': self.tracker.__class__.__name__,
            'available_backends': {
                'norfair': NORFAIR_AVAILABLE,
                'bytetrack': BYTETRACK_AVAILABLE,
                'simple': True
            }
        }
