"""
Human Activity Classification Module
Analyzes human poses and behaviors over multiple frames to classify activities.
"""
import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Deque, Any
from collections import deque, Counter
from dataclasses import dataclass, field
from enum import Enum
import time

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from tracker import TrackedObject

logger = logging.getLogger(__name__)

class ActivityType(Enum):
    """Human activity categories."""
    UNKNOWN = "unknown"
    STANDING = "standing"
    SITTING = "sitting" 
    WALKING = "walking"
    DRINKING = "drinking"
    WRITING = "writing"
    USING_PHONE = "using_phone"
    USING_LAPTOP = "using_laptop"
    TALKING = "talking"
    COMING_IN = "coming_in"
    GOING_OUT = "going_out"

@dataclass
class PoseKeypoints:
    """Human pose keypoints from MediaPipe."""
    nose: Optional[Tuple[float, float]] = None
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    left_elbow: Optional[Tuple[float, float]] = None
    right_elbow: Optional[Tuple[float, float]] = None
    left_wrist: Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None
    left_hip: Optional[Tuple[float, float]] = None
    right_hip: Optional[Tuple[float, float]] = None
    left_knee: Optional[Tuple[float, float]] = None
    right_knee: Optional[Tuple[float, float]] = None
    left_ankle: Optional[Tuple[float, float]] = None
    right_ankle: Optional[Tuple[float, float]] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if pose has minimum required keypoints."""
        required = [self.nose, self.left_shoulder, self.right_shoulder, 
                   self.left_hip, self.right_hip]
        return sum(1 for kp in required if kp is not None) >= 3

@dataclass
class ActivityFrame:
    """Single frame analysis for activity classification."""
    timestamp: float
    pose: Optional[PoseKeypoints]
    bbox: np.ndarray  # [x1, y1, x2, y2]
    movement_vector: Optional[Tuple[float, float]] = None
    hand_positions: List[Tuple[float, float]] = field(default_factory=list)
    head_position: Optional[Tuple[float, float]] = None
    body_posture: str = "unknown"

@dataclass
class ActivityHistory:
    """Track activity history for temporal analysis."""
    track_id: int
    frames: Deque[ActivityFrame] = field(default_factory=lambda: deque(maxlen=30))
    current_activity: ActivityType = ActivityType.UNKNOWN
    activity_confidence: float = 0.0
    activity_start_time: float = 0.0
    last_update: float = 0.0
    position_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=10))

class PoseDetector:
    """MediaPipe-based pose detection."""
    
    def __init__(self):
        """Initialize pose detector."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        
        # Suppress TensorFlow warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
        
        # Suppress MediaPipe warnings and ABSL logs
        import logging
        logging.getLogger('mediapipe').setLevel(logging.ERROR)
        logging.getLogger('absl').setLevel(logging.ERROR)
        
        # Suppress TensorFlow Lite warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['GLOG_minloglevel'] = '3'
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Use lighter model for better performance
            enable_segmentation=False,
            min_detection_confidence=0.25,  # Lower threshold for better performance
            min_tracking_confidence=0.25,   # Lower threshold for better performance
            smooth_landmarks=True,
            smooth_segmentation=False
        )
        
        logger.info("MediaPipe pose detector initialized")
    
    def detect_pose(self, frame: np.ndarray, bbox: np.ndarray) -> Optional[PoseKeypoints]:
        """Extract pose keypoints from person bounding box."""
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Extract person region
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                return None
            
            # Convert BGR to RGB
            rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(rgb_roi)
            
            if not results.pose_landmarks:
                return None
            
            # Extract keypoints and convert to original frame coordinates
            landmarks = results.pose_landmarks.landmark
            roi_h, roi_w = person_roi.shape[:2]
            
            def get_keypoint(idx: int) -> Optional[Tuple[float, float]]:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    if lm.visibility > 0.5:  # Only use visible keypoints
                        x = x1 + lm.x * roi_w
                        y = y1 + lm.y * roi_h
                        return (x, y)
                return None
            
            # MediaPipe pose landmark indices
            pose_kp = PoseKeypoints(
                nose=get_keypoint(0),
                left_shoulder=get_keypoint(11),
                right_shoulder=get_keypoint(12),
                left_elbow=get_keypoint(13),
                right_elbow=get_keypoint(14),
                left_wrist=get_keypoint(15),
                right_wrist=get_keypoint(16),
                left_hip=get_keypoint(23),
                right_hip=get_keypoint(24),
                left_knee=get_keypoint(25),
                right_knee=get_keypoint(26),
                left_ankle=get_keypoint(27),
                right_ankle=get_keypoint(28)
            )
            
            return pose_kp if pose_kp.is_valid else None
            
        except Exception as e:
            logger.debug(f"Pose detection failed: {e}")
            return None

class ActivityClassifier:
    """Multi-frame activity classification system."""
    
    def __init__(self, history_length: int = 30, min_confidence: float = 0.6):
        """Initialize activity classifier.
        
        Args:
            history_length: Number of frames to analyze for temporal patterns
            min_confidence: Minimum confidence threshold for activity classification
        """
        self.history_length = history_length
        self.min_confidence = min_confidence
        
        # Initialize pose detector if available
        self.pose_detector = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.pose_detector = PoseDetector()
            except Exception as e:
                logger.warning(f"Failed to initialize pose detector: {e}")
        
        # Track histories for each person
        self.activity_histories: Dict[int, ActivityHistory] = {}
        
        logger.info(f"Activity classifier initialized (pose_detection={'enabled' if self.pose_detector else 'disabled'})")
    
    def analyze_posture(self, pose: PoseKeypoints) -> str:
        """Analyze body posture from pose keypoints."""
        if not pose.is_valid:
            return "unknown"
        
        try:
            # Calculate key body ratios and angles
            posture_features = []
            
            # Shoulder-hip alignment (standing vs sitting)
            if pose.left_shoulder and pose.right_shoulder and pose.left_hip and pose.right_hip:
                shoulder_y = (pose.left_shoulder[1] + pose.right_shoulder[1]) / 2
                hip_y = (pose.left_hip[1] + pose.right_hip[1]) / 2
                torso_length = abs(hip_y - shoulder_y)
                
                # Head-shoulder distance
                if pose.nose:
                    head_shoulder_dist = abs(pose.nose[1] - shoulder_y)
                    head_torso_ratio = head_shoulder_dist / torso_length if torso_length > 0 else 0
                    posture_features.append(head_torso_ratio)
            
            # Knee position relative to hips (sitting detection)
            if pose.left_knee and pose.right_knee and pose.left_hip and pose.right_hip:
                avg_knee_y = (pose.left_knee[1] + pose.right_knee[1]) / 2
                avg_hip_y = (pose.left_hip[1] + pose.right_hip[1]) / 2
                knee_hip_ratio = (avg_knee_y - avg_hip_y) / abs(avg_hip_y) if avg_hip_y != 0 else 0
                
                # If knees are significantly above hips, likely sitting
                if knee_hip_ratio < -0.1:
                    return "sitting"
            
            # Default to standing if upright posture detected
            if len(posture_features) > 0:
                return "standing"
            
            return "unknown"
            
        except Exception as e:
            logger.debug(f"Posture analysis failed: {e}")
            return "unknown"
    
    def analyze_hand_activity(self, pose: PoseKeypoints) -> List[str]:
        """Analyze hand positions for activity detection."""
        activities = []
        
        if not pose.is_valid:
            return activities
        
        try:
            # Phone usage detection (hand near head)
            if pose.right_wrist and pose.nose:
                wrist_head_dist = np.sqrt(
                    (pose.right_wrist[0] - pose.nose[0])**2 + 
                    (pose.right_wrist[1] - pose.nose[1])**2
                )
                if wrist_head_dist < 100:  # Threshold for phone usage
                    activities.append("phone_gesture")
            
            # Laptop/writing detection (hands in front of body)
            if pose.left_wrist and pose.right_wrist and pose.left_shoulder and pose.right_shoulder:
                shoulder_center_x = (pose.left_shoulder[0] + pose.right_shoulder[0]) / 2
                shoulder_center_y = (pose.left_shoulder[1] + pose.right_shoulder[1]) / 2
                
                # Check if both hands are in front and below shoulders
                left_in_front = pose.left_wrist[1] > shoulder_center_y
                right_in_front = pose.right_wrist[1] > shoulder_center_y
                
                if left_in_front and right_in_front:
                    # Check hand distance (typing vs writing)
                    hand_distance = abs(pose.left_wrist[0] - pose.right_wrist[0])
                    if hand_distance > 150:  # Wide hand position suggests laptop
                        activities.append("laptop_gesture")
                    else:  # Close hand position suggests writing
                        activities.append("writing_gesture")
            
            # Drinking gesture (hand raised toward head)
            if pose.right_wrist and pose.right_elbow and pose.right_shoulder:
                # Check if wrist is above elbow and elbow is bent
                wrist_above_elbow = pose.right_wrist[1] < pose.right_elbow[1]
                elbow_above_shoulder = pose.right_elbow[1] < pose.right_shoulder[1]
                
                if wrist_above_elbow and elbow_above_shoulder:
                    activities.append("drinking_gesture")
            
            return activities
            
        except Exception as e:
            logger.debug(f"Hand activity analysis failed: {e}")
            return activities
    
    def classify_activity(self, history: ActivityHistory) -> Tuple[ActivityType, float]:
        """Classify activity based on temporal analysis of frames."""
        if len(history.frames) < 5:  # Need minimum frames for analysis
            return ActivityType.UNKNOWN, 0.0
        
        # Collect features from recent frames
        recent_frames = list(history.frames)[-15:]  # Last 15 frames
        
        # Movement analysis
        movement_vectors = []
        for frame in recent_frames:
            if frame.movement_vector:
                movement_vectors.append(frame.movement_vector)
        
        # Calculate average movement
        avg_movement = 0.0
        if movement_vectors:
            movements = [np.sqrt(mv[0]**2 + mv[1]**2) for mv in movement_vectors]
            avg_movement = np.mean(movements)
        
        # Posture analysis
        postures = [frame.body_posture for frame in recent_frames if frame.body_posture != "unknown"]
        posture_counter = Counter(postures)
        dominant_posture = posture_counter.most_common(1)[0][0] if postures else "unknown"
        
        # Hand gesture analysis
        all_gestures = []
        for frame in recent_frames:
            if hasattr(frame, 'hand_gestures'):
                all_gestures.extend(frame.hand_gestures)
        
        gesture_counter = Counter(all_gestures)
        dominant_gesture = gesture_counter.most_common(1)[0][0] if all_gestures else None
        
        # Activity classification logic
        confidence = 0.0
        activity = ActivityType.UNKNOWN
        
        # Movement-based activities
        if avg_movement > 20:  # High movement threshold
            if len(history.position_history) >= 2:
                # Check direction for coming in / going out
                start_pos = history.position_history[0]
                end_pos = history.position_history[-1]
                direction_x = end_pos[0] - start_pos[0]
                
                if abs(direction_x) > 50:  # Significant horizontal movement
                    activity = ActivityType.COMING_IN if direction_x > 0 else ActivityType.GOING_OUT
                    confidence = min(0.9, avg_movement / 50)
                else:
                    activity = ActivityType.WALKING
                    confidence = min(0.8, avg_movement / 30)
        
        # Gesture-based activities (override movement if confident)
        elif dominant_gesture:
            gesture_confidence = gesture_counter[dominant_gesture] / len(all_gestures)
            
            if gesture_confidence > 0.6:  # High gesture consistency
                if dominant_gesture == "phone_gesture":
                    activity = ActivityType.USING_PHONE
                    confidence = gesture_confidence * 0.9
                elif dominant_gesture == "laptop_gesture":
                    activity = ActivityType.USING_LAPTOP
                    confidence = gesture_confidence * 0.85
                elif dominant_gesture == "writing_gesture":
                    activity = ActivityType.WRITING
                    confidence = gesture_confidence * 0.8
                elif dominant_gesture == "drinking_gesture":
                    activity = ActivityType.DRINKING
                    confidence = gesture_confidence * 0.75
        
        # Posture-based activities (fallback)
        elif dominant_posture != "unknown":
            posture_confidence = posture_counter[dominant_posture] / len(postures)
            
            if posture_confidence > 0.7:
                if dominant_posture == "sitting":
                    activity = ActivityType.SITTING
                    confidence = posture_confidence * 0.7
                elif dominant_posture == "standing":
                    activity = ActivityType.STANDING
                    confidence = posture_confidence * 0.6
        
        return activity, confidence
    
    def update(self, tracked_objects: List[TrackedObject], frame: np.ndarray) -> Dict[int, Tuple[ActivityType, float]]:
        """Update activity classification for all tracked objects."""
        current_time = time.time()
        activities = {}
        
        # Minimal pose detection skipping - only when many objects present
        skip_pose = len(tracked_objects) > 5 and int(current_time * 10) % 2 != 0
        
        # Process each tracked object
        for obj in tracked_objects:
            track_id = obj.track_id
            
            # Initialize history if new track
            if track_id not in self.activity_histories:
                self.activity_histories[track_id] = ActivityHistory(
                    track_id=track_id,
                    last_update=current_time
                )
            
            history = self.activity_histories[track_id]
            
            # Detect pose if available and not skipping
            pose = None
            if self.pose_detector and not skip_pose:
                pose = self.pose_detector.detect_pose(frame, obj.xyxy)
            
            # Calculate movement vector
            movement_vector = None
            if len(history.position_history) > 0:
                prev_pos = history.position_history[-1]
                curr_pos = obj.center
                movement_vector = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            
            # Analyze current frame
            body_posture = "unknown"
            hand_gestures = []
            
            if pose:
                body_posture = self.analyze_posture(pose)
                hand_gestures = self.analyze_hand_activity(pose)
            
            # Create activity frame
            activity_frame = ActivityFrame(
                timestamp=current_time,
                pose=pose,
                bbox=obj.xyxy,
                movement_vector=movement_vector,
                body_posture=body_posture
            )
            activity_frame.hand_gestures = hand_gestures  # Add gestures dynamically
            
            # Add to history
            history.frames.append(activity_frame)
            history.position_history.append(obj.center)
            history.last_update = current_time
            
            # Classify activity more frequently for better accuracy
            if len(history.frames) % 3 == 0 or len(history.frames) < 10:
                activity, confidence = self.classify_activity(history)
                
                # Update activity if confidence is high enough
                if confidence >= self.min_confidence:
                    if history.current_activity != activity:
                        history.current_activity = activity
                        history.activity_confidence = confidence
                        history.activity_start_time = current_time
                        logger.debug(f"Track {track_id}: Activity changed to {activity.value} (conf: {confidence:.2f})")
            
            activities[track_id] = (history.current_activity, history.activity_confidence)
        
        # Clean up old histories less frequently
        if int(current_time) % 10 == 0:  # Only every 10 seconds
            active_track_ids = {obj.track_id for obj in tracked_objects}
            to_remove = []
            for track_id in self.activity_histories:
                if track_id not in active_track_ids:
                    # Remove if not seen for 30 seconds
                    if current_time - self.activity_histories[track_id].last_update > 30:
                        to_remove.append(track_id)
            
            for track_id in to_remove:
                del self.activity_histories[track_id]
        
        return activities
    
    def get_current_activity(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get current activity for a track ID."""
        if track_id in self.activity_histories:
            history = self.activity_histories[track_id]
            # Get the most recent activity from history
            if hasattr(history, 'poses') and history.poses:
                # Use the last classified activity if available
                if hasattr(history, 'last_activity') and hasattr(history, 'last_confidence'):
                    return {
                        'activity': history.last_activity.value if history.last_activity != ActivityType.UNKNOWN else 'unknown',
                        'confidence': history.last_confidence
                    }
        return None
    
    def get_activity_summary(self) -> Dict[str, int]:
        """Get summary of current activities across all tracks."""
        summary = {}
        for track_id, history in self.activity_histories.items():
            if hasattr(history, 'last_activity') and hasattr(history, 'last_confidence'):
                if history.last_activity != ActivityType.UNKNOWN and history.last_confidence >= self.min_confidence:
                    activity_name = history.last_activity.value
                    summary[activity_name] = summary.get(activity_name, 0) + 1
        return summary
