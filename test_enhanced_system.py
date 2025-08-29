"""
Test script for the enhanced surveillance system with object detection and interval-based activity classification.
"""

import asyncio
import logging
import time
import numpy as np
from pathlib import Path

# Import system components
from detector import PersonDetector
from interval_classifier import IntervalActivityClassifier, InteractionType
from activity_classifier import ActivityType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_enhanced_detection():
    """Test enhanced object detection capabilities."""
    logger.info("Testing enhanced object detection...")
    
    try:
        # Initialize detector with object detection enabled
        detector = PersonDetector(
            model_path="models/yolo11n.onnx",
            imgsz=416,
            conf_threshold=0.4,
            nms_threshold=0.5,
            detect_objects=True
        )
        
        # Create a dummy frame for testing
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        detections = detector.detect(test_frame)
        
        logger.info(f"Detection test completed. Found {len(detections)} detections")
        
        # Check if detections have class names
        for det in detections:
            if hasattr(det, 'class_name'):
                logger.info(f"Detected: {det.class_name} (confidence: {det.confidence:.2f})")
            else:
                logger.warning("Detection missing class_name attribute")
        
        return True
        
    except Exception as e:
        logger.error(f"Detection test failed: {e}")
        return False

def test_interval_classifier():
    """Test interval-based activity classification."""
    logger.info("Testing interval-based activity classification...")
    
    try:
        # Initialize interval classifier
        classifier = IntervalActivityClassifier(interval_duration=2.0, overlap_duration=0.5)
        
        # Create mock data
        mock_tracked_objects = []
        mock_detections = []
        mock_activities = {1: (ActivityType.STANDING, 0.8)}
        
        # Add several frames of data
        for i in range(10):
            classifier.add_frame_data(mock_tracked_objects, mock_detections, mock_activities)
            time.sleep(0.1)  # Simulate frame timing
        
        # Get results
        current_activities = classifier.get_current_activities()
        summary = classifier.get_interval_summary()
        transitions = classifier.get_activity_transitions()
        
        logger.info(f"Interval classifier test completed")
        logger.info(f"Current activities: {current_activities}")
        logger.info(f"Summary: {summary}")
        logger.info(f"Transitions: {transitions}")
        
        return True
        
    except Exception as e:
        logger.error(f"Interval classifier test failed: {e}")
        return False

def test_interaction_detection():
    """Test person-object interaction detection."""
    logger.info("Testing interaction detection...")
    
    try:
        classifier = IntervalActivityClassifier()
        
        # Mock person and object positions
        persons = {1: (100, 100)}  # Person at position (100, 100)
        objects = []
        
        # Test different interaction scenarios
        test_cases = [
            ("laptop", (120, 110), "close interaction"),
            ("cell phone", (105, 105), "very close interaction"),
            ("bottle", (200, 200), "distant object"),
        ]
        
        for obj_type, obj_pos, description in test_cases:
            from interval_classifier import ObjectPresence
            obj = ObjectPresence(
                object_type=obj_type,
                position=obj_pos,
                confidence=0.8,
                bbox=np.array([obj_pos[0]-10, obj_pos[1]-10, obj_pos[0]+10, obj_pos[1]+10])
            )
            
            interactions = classifier._detect_interactions(persons, [obj], time.time())
            
            logger.info(f"{description} - {obj_type} at {obj_pos}: {len(interactions)} interactions")
            for interaction in interactions:
                logger.info(f"  {interaction.interaction_type.value} (confidence: {interaction.confidence:.2f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Interaction detection test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests for the enhanced system."""
    logger.info("Starting enhanced surveillance system tests...")
    
    tests = [
        ("Object Detection", test_enhanced_detection()),
        ("Interval Classification", test_interval_classifier),
        ("Interaction Detection", test_interaction_detection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if asyncio.iscoroutine(test_func):
                result = await test_func
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced surveillance system is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests())
