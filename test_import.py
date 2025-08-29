#!/usr/bin/env python3
"""Test script to verify imports are working correctly."""

try:
    print("Testing imports...")
    
    # Test the problematic import
    from api import PeopleCounterAPI
    print("✅ Successfully imported PeopleCounterAPI from api")
    
    # Test other imports
    from detector import PersonDetector
    from tracker import TrackerManager
    from counter import PeopleCounter
    from interval_classifier import IntervalActivityClassifier
    print("✅ All core imports successful")
    
    print("\n🎉 All imports are working correctly!")
    print("The surveillance system should now start without import errors.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease check that main.py line 32-34 contains:")
    print("from api import PeopleCounterAPI")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
