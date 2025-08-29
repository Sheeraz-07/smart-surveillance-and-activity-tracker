"""
Line/region crossing counter with per-track debouncing and calibration.
Implements robust counting logic with configurable crossing lines and regions.
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time

from tracker import TrackedObject

logger = logging.getLogger(__name__)

class CrossingDirection(Enum):
    """Crossing direction enumeration."""
    IN = "in"
    OUT = "out"

@dataclass
class CrossingLine:
    """Represents a counting line with direction logic."""
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    name: str = "default"
    
    def __post_init__(self):
        """Calculate line parameters after initialization."""
        x1, y1 = self.start_point
        x2, y2 = self.end_point
        
        # Line equation: ax + by + c = 0
        self.a = y2 - y1
        self.b = x1 - x2
        self.c = x2 * y1 - x1 * y2
        
        # Normalize for consistent distance calculation
        norm = np.sqrt(self.a**2 + self.b**2)
        if norm > 0:
            self.a /= norm
            self.b /= norm
            self.c /= norm
    
    def signed_distance(self, point: Tuple[float, float]) -> float:
        """Calculate signed distance from point to line.
        
        Args:
            point: (x, y) coordinates
            
        Returns:
            Signed distance (positive/negative indicates side)
        """
        x, y = point
        return self.a * x + self.b * y + self.c
    
    def get_crossing_direction(self, prev_distance: float, curr_distance: float) -> Optional[CrossingDirection]:
        """Determine crossing direction based on distance signs.
        
        Args:
            prev_distance: Previous signed distance
            curr_distance: Current signed distance
            
        Returns:
            CrossingDirection or None if no crossing
        """
        # Check if crossed the line (sign change)
        if prev_distance * curr_distance < 0:
            # Determine direction based on sign change
            if prev_distance < 0 and curr_distance > 0:
                return CrossingDirection.IN
            elif prev_distance > 0 and curr_distance < 0:
                return CrossingDirection.OUT
        
        return None

@dataclass
class TrackHistory:
    """Track history for debouncing and crossing detection."""
    track_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    last_crossing_time: float = 0.0
    last_crossing_direction: Optional[CrossingDirection] = None
    crossing_count: int = 0
    
    def add_position(self, position: Tuple[float, float], distance: float, max_history: int = 10):
        """Add new position to history.
        
        Args:
            position: (x, y) coordinates
            distance: Signed distance to counting line
            max_history: Maximum history length to maintain
        """
        self.positions.append(position)
        self.distances.append(distance)
        
        # Limit history size
        if len(self.positions) > max_history:
            self.positions = self.positions[-max_history:]
            self.distances = self.distances[-max_history:]

class PeopleCounter:
    """Main people counting class with line crossing detection."""
    
    def __init__(self,
                 counting_line: CrossingLine,
                 debounce_frames: int = 15,
                 debounce_time: float = 2.0,
                 min_track_length: int = 3,
                 max_track_age: int = 30):
        """Initialize people counter.
        
        Args:
            counting_line: Line for crossing detection
            debounce_frames: Minimum frames between crossings for same track
            debounce_time: Minimum time (seconds) between crossings for same track
            min_track_length: Minimum track history before counting
            max_track_age: Maximum frames to keep track history
        """
        self.counting_line = counting_line
        self.debounce_frames = debounce_frames
        self.debounce_time = debounce_time
        self.min_track_length = min_track_length
        self.max_track_age = max_track_age
        
        # Track histories for debouncing
        self.track_histories: Dict[int, TrackHistory] = {}
        
        # Counters
        self.in_count = 0
        self.out_count = 0
        
        # Frame counter for debouncing
        self.frame_count = 0
        
        logger.info(f"Initialized counter with line from {counting_line.start_point} to {counting_line.end_point}")
    
    def update(self, tracked_objects: List[TrackedObject]) -> List[Dict]:
        """Update counter with new tracked objects.
        
        Args:
            tracked_objects: List of TrackedObject instances
            
        Returns:
            List of crossing events that occurred this frame
        """
        self.frame_count += 1
        current_time = time.time()
        crossings = []
        
        # Get current track IDs
        current_track_ids = {obj.track_id for obj in tracked_objects}
        
        # Update existing tracks and detect crossings
        for obj in tracked_objects:
            track_id = obj.track_id
            center = obj.center
            distance = self.counting_line.signed_distance(center)
            
            # Get or create track history
            if track_id not in self.track_histories:
                self.track_histories[track_id] = TrackHistory(track_id=track_id)
            
            history = self.track_histories[track_id]
            
            # Check for crossing if we have previous distance
            if len(history.distances) > 0:
                prev_distance = history.distances[-1]
                crossing_direction = self.counting_line.get_crossing_direction(prev_distance, distance)
                
                if crossing_direction is not None:
                    # Check debouncing conditions
                    frames_since_last = self.frame_count - history.crossing_count
                    time_since_last = current_time - history.last_crossing_time
                    
                    # Ensure minimum track length and debouncing
                    if (len(history.positions) >= self.min_track_length and
                        frames_since_last >= self.debounce_frames and
                        time_since_last >= self.debounce_time):
                        
                        # Valid crossing detected
                        if crossing_direction == CrossingDirection.IN:
                            self.in_count += 1
                        else:
                            self.out_count += 1
                        
                        # Update history
                        history.last_crossing_time = current_time
                        history.last_crossing_direction = crossing_direction
                        history.crossing_count = self.frame_count
                        
                        # Create crossing event
                        crossing_event = {
                            'track_id': track_id,
                            'direction': crossing_direction.value,
                            'position': center,
                            'confidence': obj.confidence,
                            'timestamp': current_time,
                            'frame_number': self.frame_count
                        }
                        crossings.append(crossing_event)
                        
                        logger.info(f"Crossing detected: Track {track_id} went {crossing_direction.value} "
                                  f"at {center} (conf: {obj.confidence:.2f})")
            
            # Add current position to history
            history.add_position(center, distance)
        
        # Clean up old track histories
        self._cleanup_old_tracks(current_track_ids)
        
        return crossings
    
    def _cleanup_old_tracks(self, current_track_ids: Set[int]):
        """Remove old track histories that are no longer active.
        
        Args:
            current_track_ids: Set of currently active track IDs
        """
        # Remove tracks that haven't been seen for max_track_age frames
        to_remove = []
        for track_id, history in self.track_histories.items():
            if track_id not in current_track_ids:
                frames_since_seen = self.frame_count - history.crossing_count
                if frames_since_seen > self.max_track_age:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.track_histories[track_id]
            logger.debug(f"Removed old track history for ID {track_id}")
    
    def get_counts(self) -> Dict[str, int]:
        """Get current crossing counts.
        
        Returns:
            Dictionary with 'in', 'out', and 'occupancy' counts
        """
        return {
            'in': self.in_count,
            'out': self.out_count,
            'occupancy': self.in_count - self.out_count
        }
    
    def reset_counts(self):
        """Reset all counters to zero."""
        self.in_count = 0
        self.out_count = 0
        logger.info("Counter reset to zero")
    
    def get_line_info(self) -> Dict:
        """Get counting line information for visualization."""
        return {
            'start_point': self.counting_line.start_point,
            'end_point': self.counting_line.end_point,
            'name': self.counting_line.name
        }
    
    def get_active_tracks_info(self) -> List[Dict]:
        """Get information about currently tracked objects near the line.
        
        Returns:
            List of track information dictionaries
        """
        tracks_info = []
        
        for track_id, history in self.track_histories.items():
            if len(history.positions) > 0:
                last_pos = history.positions[-1]
                last_distance = history.distances[-1]
                
                track_info = {
                    'track_id': track_id,
                    'position': last_pos,
                    'distance_to_line': abs(last_distance),
                    'side': 'positive' if last_distance > 0 else 'negative',
                    'history_length': len(history.positions),
                    'last_crossing': history.last_crossing_direction.value if history.last_crossing_direction else None
                }
                tracks_info.append(track_info)
        
        return tracks_info

class RegionCounter:
    """Alternative region-based counter for complex counting scenarios."""
    
    def __init__(self, 
                 entry_region: List[Tuple[float, float]],
                 exit_region: List[Tuple[float, float]],
                 debounce_time: float = 2.0):
        """Initialize region-based counter.
        
        Args:
            entry_region: Polygon points defining entry region
            exit_region: Polygon points defining exit region  
            debounce_time: Minimum time between counts for same track
        """
        self.entry_region = np.array(entry_region)
        self.exit_region = np.array(exit_region)
        self.debounce_time = debounce_time
        
        # Track states
        self.track_states: Dict[int, Dict] = {}
        
        # Counters
        self.in_count = 0
        self.out_count = 0
        
        logger.info(f"Initialized region counter with {len(entry_region)} entry points "
                   f"and {len(exit_region)} exit points")
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: np.ndarray) -> bool:
        """Check if point is inside polygon using ray casting algorithm.
        
        Args:
            point: (x, y) coordinates
            polygon: Array of polygon vertices
            
        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def update(self, tracked_objects: List[TrackedObject]) -> List[Dict]:
        """Update region counter with tracked objects.
        
        Args:
            tracked_objects: List of TrackedObject instances
            
        Returns:
            List of crossing events
        """
        current_time = time.time()
        crossings = []
        
        for obj in tracked_objects:
            track_id = obj.track_id
            center = obj.center
            
            # Initialize track state if new
            if track_id not in self.track_states:
                self.track_states[track_id] = {
                    'in_entry': False,
                    'in_exit': False,
                    'last_count_time': 0.0
                }
            
            state = self.track_states[track_id]
            
            # Check region occupancy
            in_entry = self._point_in_polygon(center, self.entry_region)
            in_exit = self._point_in_polygon(center, self.exit_region)
            
            # Detect entry (entry region -> exit region)
            if state['in_entry'] and in_exit and not state['in_exit']:
                time_since_last = current_time - state['last_count_time']
                if time_since_last >= self.debounce_time:
                    self.in_count += 1
                    state['last_count_time'] = current_time
                    
                    crossings.append({
                        'track_id': track_id,
                        'direction': 'in',
                        'position': center,
                        'confidence': obj.confidence,
                        'timestamp': current_time
                    })
            
            # Detect exit (exit region -> entry region)
            elif state['in_exit'] and in_entry and not state['in_entry']:
                time_since_last = current_time - state['last_count_time']
                if time_since_last >= self.debounce_time:
                    self.out_count += 1
                    state['last_count_time'] = current_time
                    
                    crossings.append({
                        'track_id': track_id,
                        'direction': 'out',
                        'position': center,
                        'confidence': obj.confidence,
                        'timestamp': current_time
                    })
            
            # Update state
            state['in_entry'] = in_entry
            state['in_exit'] = in_exit
        
        return crossings
    
    def get_counts(self) -> Dict[str, int]:
        """Get current counts."""
        return {
            'in': self.in_count,
            'out': self.out_count,
            'occupancy': self.in_count - self.out_count
        }
