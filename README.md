# People Counter System

A production-ready, Windows-optimized people counting system using ONNX-based person detection, multi-object tracking, and line crossing analysis.

## Project Structure

```
CAMERA_SURVEILLANCE_SYSTEM/
├── main.py                 # Main application entry point
├── api.py                  # FastAPI REST and WebSocket endpoints
├── capture.py              # Video capture and RTSP handling
├── counter.py              # People counting logic and line crossing
├── db.py                   # SQLite database operations
├── detector.py             # YOLO11 person detection
├── tracker.py              # Multi-object tracking (Norfair/ByteTrack)
├── config.yaml             # Main configuration file
├── config.example.yaml     # Example configuration template
├── requirements.txt        # Python dependencies
├── run_windows.ps1         # Windows PowerShell startup script
├── logs/                   # Log files directory
├── models/                 # ONNX model files directory
└── venv/                   # Python virtual environment
```

## Quick Start

1. **Navigate to project directory:**
   ```powershell
   cd "e:\DIGITAL_SOFTS_OFFICIAL_PROJECT_DIRECTORY\CAMERA_SURVEILLANCE_SYSTEM"
   ```

2. **Activate virtual environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Start the system:**
   ```powershell
   python main.py
   ```

## Alternative: Use PowerShell Script

```powershell
.\run_windows.ps1
```

This script will automatically:
- Activate the virtual environment
- Install dependencies
- Create necessary directories
- Start the people counting system

## Configuration

Edit `config.yaml` to customize:

```yaml
video:
  source: 0  # Webcam (0, 1, 2...) or RTSP URL or video file

counter:
  line_start: [640, 0]    # Counting line start point
  line_end: [640, 720]    # Counting line end point

detector:
  conf_threshold: 0.35    # Detection confidence threshold
  imgsz: 512             # Input image size
```

## API Endpoints

- **Health Check**: `http://localhost:8000/api/health`
- **Live Counts**: `http://localhost:8000/api/live`
- **WebSocket**: `ws://localhost:8000/ws/live`

## System Requirements

- Windows 10/11
- Python 3.10+
- 4GB RAM
- Intel i5 or equivalent CPU
- Webcam or IP camera (optional)

## License

YOLO11 models are subject to AGPL-3.0 license terms. See LICENSE_NOTICE.md for details.
