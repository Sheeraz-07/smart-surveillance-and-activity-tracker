#!/usr/bin/env python3
"""
Test script to identify startup issues with the people counter system.
"""
import sys
import traceback

def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    
    try:
        import yaml
        print("✓ yaml imported")
    except Exception as e:
        print(f"✗ yaml import failed: {e}")
        return False

    try:
        import cv2
        print("✓ cv2 imported")
    except Exception as e:
        print(f"✗ cv2 import failed: {e}")
        return False

    try:
        import numpy as np
        print("✓ numpy imported")
    except Exception as e:
        print(f"✗ numpy import failed: {e}")
        return False

    try:
        from capture import VideoCapture
        print("✓ capture module imported")
    except Exception as e:
        print(f"✗ capture import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from detector import PersonDetector
        print("✓ detector module imported")
    except Exception as e:
        print(f"✗ detector import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from tracker import TrackerManager
        print("✓ tracker module imported")
    except Exception as e:
        print(f"✗ tracker import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from counter import PeopleCounter, CrossingLine
        print("✓ counter module imported")
    except Exception as e:
        print(f"✗ counter import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from db import CounterDB
        print("✓ db module imported")
    except Exception as e:
        print(f"✗ db import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from api import PeopleCounterAPI
        print("✓ api module imported")
    except Exception as e:
        print(f"✗ api import failed: {e}")
        traceback.print_exc()
        return False

    return True

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path("config.yaml")
        if not config_path.exists():
            print("✗ config.yaml not found")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        print("✓ Configuration loaded successfully")
        print(f"  Video source: {config.get('video', {}).get('source', 'Not set')}")
        print(f"  API port: {config.get('api', {}).get('port', 'Not set')}")
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        traceback.print_exc()
        return False

def test_system_creation():
    """Test system creation."""
    print("\nTesting system creation...")
    
    try:
        from main import PeopleCounterSystem
        system = PeopleCounterSystem()
        print("✓ PeopleCounterSystem created successfully")
        return True
    except Exception as e:
        print(f"✗ System creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("People Counter System Startup Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return 1
    
    # Test configuration
    if not test_config():
        print("\n❌ Configuration tests failed")
        return 1
    
    # Test system creation
    if not test_system_creation():
        print("\n❌ System creation tests failed")
        return 1
    
    print("\n✅ All tests passed! System should be ready to run.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
