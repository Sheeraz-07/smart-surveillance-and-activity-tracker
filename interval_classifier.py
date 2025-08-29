"""
Interval-based activity classification system.
Analyzes 5-second intervals of activity data to classify activities based on
object presence, person-object interactions, and temporal patterns.
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import numpy as np

try:
    from activity_classifier import ActivityType
except ImportError:
    # Define ActivityType locally if import fails
    class ActivityType(Enum):
        UNKNOWN = "unknown"
        STANDING = "standing"
        SITTING = "sitting"
        WALKING = "walking"
        USING_PHONE = "using_phone"
        USING_LAPTOP = "using_laptop"
        DRINKING = "drinking"

logger = logging.getLogger(__name__)

class InteractionType(Enum):
    """Types of person-object interactions."""
    HOLDING = "holding"
    USING = "using"
    NEAR = "near"
    WRITING = "writing"
    DRINKING = "drinking"
    TYPING = "typing"

@dataclass
class ObjectPresence:
    """Object presence data in a frame."""
    object_type: str
    position: Tuple[float, float]
    confidence: float
    bbox: np.ndarray

@dataclass
class PersonObjectInteraction:
    """Person-object interaction data."""
    person_id: int
    object_type: str
    interaction_type: InteractionType
    confidence: float
    distance: float
    timestamp: float

@dataclass
class IntervalFrame:
    """Single frame data within an interval."""
    timestamp: float
    persons: Dict[int, Tuple[float, float]]  # person_id -> center position
    objects: List[ObjectPresence]
    interactions: List[PersonObjectInteraction]
    activities: Dict[int, Tuple[ActivityType, float]]  # person_id -> (activity, confidence)

@dataclass
class ActivityInterval:
    """5-second activity analysis interval."""
    start_time: float
    end_time: float
    frames: List[IntervalFrame] = field(default_factory=list)
    dominant_objects: Dict[str, int] = field(default_factory=dict)  # object_type -> count
    person_activities: Dict[int, List[Tuple[ActivityType, float]]] = field(default_factory=lambda: defaultdict(list))
    interactions: List[PersonObjectInteraction] = field(default_factory=list)
    classified_activities: Dict[int, Tuple[ActivityType, float]] = field(default_factory=dict)

class IntervalActivityClassifier:
    """Enhanced activity classifier using 5-second intervals."""
    
    def __init__(self, interval_duration: float = 5.0, overlap_duration: float = 1.0):
        """Initialize interval classifier.
        
        Args:
            interval_duration: Duration of each analysis interval in seconds
            overlap_duration: Overlap between consecutive intervals in seconds
        """
        self.interval_duration = interval_duration
        self.overlap_duration = overlap_duration
        
        # Sliding window of intervals
        self.current_interval: Optional[ActivityInterval] = None
        self.previous_interval: Optional[ActivityInterval] = None
        self.interval_history: deque = deque(maxlen=10)  # Keep last 10 intervals
        
        # Frame buffer for current interval
        self.frame_buffer: List[IntervalFrame] = []
        
        # Activity classification rules
        self.object_activity_rules = {
            'laptop': {InteractionType.TYPING: ActivityType.USING_LAPTOP},
            'cell phone': {InteractionType.HOLDING: ActivityType.USING_PHONE, InteractionType.USING: ActivityType.USING_PHONE},
            'book': {InteractionType.HOLDING: ActivityType.UNKNOWN, InteractionType.USING: ActivityType.UNKNOWN},
            'bottle': {InteractionType.DRINKING: ActivityType.DRINKING, InteractionType.HOLDING: ActivityType.DRINKING},
            'cup': {InteractionType.DRINKING: ActivityType.DRINKING, InteractionType.HOLDING: ActivityType.DRINKING}
        }
        
        # Interaction detection thresholds
        self.interaction_distance_threshold = 100  # pixels
        self.holding_distance_threshold = 50  # pixels
        
    def update(self, tracked_objects, current_time: float) -> Dict:
        """Update interval classifier with new frame data.
        
        Args:
            tracked_objects: List of tracked objects from tracker
            current_time: Current timestamp
            
        Returns:
            Dictionary with interval analysis results
        """
        # Extract activities from tracked objects
        activities = {}
        for obj in tracked_objects:
            if hasattr(obj, 'class_name') and obj.class_name == 'person':
                if hasattr(obj, 'activity') and obj.activity:
                    # Convert activity string to ActivityType if needed
                    try:
                        activity_type = ActivityType(obj.activity) if isinstance(obj.activity, str) else obj.activity
                        confidence = getattr(obj, 'activity_confidence', 0.7)
                        activities[getattr(obj, 'id', getattr(obj, 'track_id', 0))] = (activity_type, confidence)
                    except (ValueError, AttributeError):
                        # Handle unknown activity types
                        activities[getattr(obj, 'id', getattr(obj, 'track_id', 0))] = (ActivityType.UNKNOWN, 0.5)
        
        # Convert tracked objects to detections format for compatibility
        detections = []
        for obj in tracked_objects:
            if hasattr(obj, 'bbox') and hasattr(obj, 'class_name'):
                detections.append(obj)
        
        # Call the existing add_frame_data method
        self.add_frame_data(tracked_objects, detections, activities)
        
        # Return current interval summary
        return self.get_interval_summary()

    def add_frame_data(self, tracked_objects, detections, activities: Dict[int, Tuple[ActivityType, float]]):
        """Add frame data to current interval."""
        current_time = time.time()
        
        # Initialize new interval if needed
        if self.current_interval is None:
            self.current_interval = ActivityInterval(
                start_time=current_time,
                end_time=current_time + self.interval_duration
            )
        
        # Extract persons and objects from tracked objects
        persons = {}
        objects = []
        
        for obj in tracked_objects:
            if hasattr(obj, 'class_name') and obj.class_name == 'person':
                # Get object ID
                obj_id = getattr(obj, 'id', getattr(obj, 'track_id', 0))
                # Get center position
                if hasattr(obj, 'center'):
                    persons[obj_id] = obj.center
                elif hasattr(obj, 'bbox'):
                    # Calculate center from bbox
                    bbox = obj.bbox
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    persons[obj_id] = (center_x, center_y)
            elif hasattr(obj, 'class_name') and obj.class_name != 'person':
                # Get center position for objects
                if hasattr(obj, 'center'):
                    position = obj.center
                elif hasattr(obj, 'bbox'):
                    bbox = obj.bbox
                    position = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                else:
                    continue
                
                objects.append(ObjectPresence(
                    object_type=obj.class_name,
                    position=position,
                    confidence=getattr(obj, 'confidence', 0.7),
                    bbox=getattr(obj, 'bbox', np.array([0, 0, 0, 0]))
                ))
        
        # Detect interactions
        interactions = self._detect_interactions(persons, objects, current_time)
        
        # Create interval frame
        interval_frame = IntervalFrame(
            timestamp=current_time,
            persons=persons,
            objects=objects,
            interactions=interactions,
            activities=activities
        )
        
        # Add to current interval
        self.current_interval.frames.append(interval_frame)
        self.current_interval.interactions.extend(interactions)
        
        # Update person activities
        for person_id, (activity, confidence) in activities.items():
            self.current_interval.person_activities[person_id].append((activity, confidence))
        
        # Check if interval is complete
        if current_time >= self.current_interval.end_time:
            self._finalize_interval()
    
    def _detect_interactions(self, persons: Dict[int, Tuple[float, float]], 
                           objects: List[ObjectPresence], timestamp: float) -> List[PersonObjectInteraction]:
        """Detect person-object interactions in current frame."""
        interactions = []
        
        for person_id, person_pos in persons.items():
            for obj in objects:
                distance = np.sqrt((person_pos[0] - obj.position[0])**2 + 
                                 (person_pos[1] - obj.position[1])**2)
                
                if distance <= self.interaction_distance_threshold:
                    # Determine interaction type based on distance and object type
                    interaction_type = self._classify_interaction(obj.object_type, distance)
                    
                    if interaction_type:
                        confidence = max(0.1, 1.0 - (distance / self.interaction_distance_threshold))
                        interactions.append(PersonObjectInteraction(
                            person_id=person_id,
                            object_type=obj.object_type,
                            interaction_type=interaction_type,
                            confidence=confidence,
                            distance=distance,
                            timestamp=timestamp
                        ))
        
        return interactions
    
    def _classify_interaction(self, object_type: str, distance: float) -> Optional[InteractionType]:
        """Classify the type of interaction based on object and distance."""
        if distance <= self.holding_distance_threshold:
            if object_type in ['bottle', 'cup']:
                return InteractionType.DRINKING
            elif object_type == 'cell phone':
                return InteractionType.HOLDING
            elif object_type == 'book':
                return InteractionType.HOLDING
            elif object_type == 'laptop':
                return InteractionType.TYPING
        elif distance <= self.interaction_distance_threshold:
            if object_type == 'laptop':
                return InteractionType.USING
            elif object_type in ['book', 'cell phone']:
                return InteractionType.USING
            else:
                return InteractionType.NEAR
        
        return None
    
    def _finalize_interval(self):
        """Finalize current interval and classify activities."""
        if not self.current_interval:
            return
        
        # Analyze object presence
        object_counts = defaultdict(int)
        for frame in self.current_interval.frames:
            for obj in frame.objects:
                object_counts[obj.object_type] += 1
        
        self.current_interval.dominant_objects = dict(object_counts)
        
        # Classify activities for each person using majority voting and interactions
        for person_id in self.current_interval.person_activities:
            classified_activity = self._classify_person_activity(person_id)
            if classified_activity:
                self.current_interval.classified_activities[person_id] = classified_activity
        
        # Move to history
        self.interval_history.append(self.current_interval)
        self.previous_interval = self.current_interval
        
        # Start new interval with overlap
        overlap_start = self.current_interval.end_time - self.overlap_duration
        self.current_interval = ActivityInterval(
            start_time=overlap_start,
            end_time=overlap_start + self.interval_duration
        )
        
        logger.info(f"Finalized interval with {len(self.previous_interval.frames)} frames, "
                   f"{len(self.previous_interval.classified_activities)} classified activities")
    
    def _classify_person_activity(self, person_id: int) -> Optional[Tuple[ActivityType, float]]:
        """Classify activity for a specific person using interval data."""
        if not self.current_interval:
            return None
        
        # Get person's interactions in this interval
        person_interactions = [i for i in self.current_interval.interactions if i.person_id == person_id]
        
        # Priority 1: Object-based activity classification
        if person_interactions:
            # Group interactions by object type and interaction type
            interaction_votes = defaultdict(list)
            
            for interaction in person_interactions:
                if interaction.object_type in self.object_activity_rules:
                    rules = self.object_activity_rules[interaction.object_type]
                    if interaction.interaction_type in rules:
                        activity = rules[interaction.interaction_type]
                        interaction_votes[activity].append(interaction.confidence)
            
            # Find most confident object-based activity
            if interaction_votes:
                best_activity = None
                best_confidence = 0
                
                for activity, confidences in interaction_votes.items():
                    avg_confidence = np.mean(confidences)
                    if avg_confidence > best_confidence:
                        best_activity = activity
                        best_confidence = avg_confidence
                
                if best_activity and best_confidence > 0.3:
                    return (best_activity, best_confidence)
        
        # Priority 2: Pose-based activity classification (majority voting)
        person_activities = self.current_interval.person_activities.get(person_id, [])
        if person_activities:
            # Count activity votes
            activity_votes = defaultdict(list)
            for activity, confidence in person_activities:
                activity_votes[activity].append(confidence)
            
            # Find majority activity
            best_activity = None
            best_score = 0
            
            for activity, confidences in activity_votes.items():
                # Score = count * average confidence
                score = len(confidences) * np.mean(confidences)
                if score > best_score:
                    best_activity = activity
                    best_score = score
            
            if best_activity:
                avg_confidence = np.mean(activity_votes[best_activity])
                return (best_activity, avg_confidence)
        
        return None
    
    def get_current_activities(self) -> Dict[int, Tuple[ActivityType, float]]:
        """Get current classified activities for all persons."""
        if self.previous_interval:
            return self.previous_interval.classified_activities.copy()
        return {}
    
    def get_interval_summary(self) -> Dict:
        """Get summary of recent interval analysis."""
        if not self.previous_interval:
            return {}
        
        return {
            'interval_duration': self.previous_interval.end_time - self.previous_interval.start_time,
            'frames_analyzed': len(self.previous_interval.frames),
            'dominant_objects': self.previous_interval.dominant_objects,
            'interactions_detected': len(self.previous_interval.interactions),
            'activities_classified': len(self.previous_interval.classified_activities),
            'persons_tracked': len(self.previous_interval.person_activities)
        }
    
    def get_activity_transitions(self) -> Dict[int, Tuple[ActivityType, ActivityType]]:
        """Get activity transitions between previous and current intervals."""
        transitions = {}
        
        if not self.previous_interval or len(self.interval_history) < 2:
            return transitions
        
        prev_prev_interval = self.interval_history[-2]
        
        for person_id in self.previous_interval.classified_activities:
            if person_id in prev_prev_interval.classified_activities:
                old_activity = prev_prev_interval.classified_activities[person_id][0]
                new_activity = self.previous_interval.classified_activities[person_id][0]
                
                if old_activity != new_activity:
                    transitions[person_id] = (old_activity, new_activity)
        
        return transitions
