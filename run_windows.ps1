# People Counter System - Windows PowerShell Run Script
# This script sets up the virtual environment and starts the people counting system

# Stop on any error
$ErrorActionPreference = "Stop"

# Get script directory and set as working directory
$ScriptRoot = $PSScriptRoot
Set-Location -Path $ScriptRoot

Write-Host "People Counter System - Starting..." -ForegroundColor Green
Write-Host "Working directory: $ScriptRoot" -ForegroundColor Cyan

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found in PATH. Please install Python 3.10 or later." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if existing venv directory exists
$venvPath = ".\venv"
if (!(Test-Path $venvPath)) {
    Write-Host "ERROR: Virtual environment not found at $venvPath" -ForegroundColor Red
    Write-Host "Please ensure the 'venv' directory exists in the project root." -ForegroundColor Yellow
    Write-Host "If you need to create it, run: python -m venv venv" -ForegroundColor Yellow
    exit 1
} else {
    Write-Host "Found existing virtual environment at $venvPath" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
try {
    & "$venvPath\Scripts\Activate.ps1"
    Write-Host "Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Make sure the venv directory contains a valid Python virtual environment" -ForegroundColor Yellow
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
try {
    python -m pip install --upgrade pip --quiet
    Write-Host "Pip upgraded successfully" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Failed to upgrade pip, continuing anyway..." -ForegroundColor Yellow
}

# Install requirements
Write-Host "Installing/updating dependencies..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    try {
        pip install -r requirements.txt --quiet
        Write-Host "Dependencies installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        Write-Host "Check requirements.txt and your internet connection" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "WARNING: requirements.txt not found, installing core packages..." -ForegroundColor Yellow
    try {
        pip install ultralytics opencv-python fastapi uvicorn aiosqlite pyyaml numpy norfair mediapipe --quiet
        Write-Host "Core packages installed" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Failed to install core packages" -ForegroundColor Red
        exit 1
    }
}

# Create necessary directories
$directories = @("logs", "models")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# Copy example config if config.yaml doesn't exist
if (!(Test-Path "config.yaml") -and (Test-Path "config.example.yaml")) {
    Write-Host "Creating config.yaml from example..." -ForegroundColor Yellow
    Copy-Item "config.example.yaml" "config.yaml"
    Write-Host "Config file created. Please review and modify config.yaml as needed." -ForegroundColor Green
}

# Check for model file
$modelPath = "models\yolo11n.onnx"
if (!(Test-Path $modelPath)) {
    Write-Host "WARNING: Model file not found at $modelPath" -ForegroundColor Yellow
    Write-Host "The system will attempt to download YOLO11n automatically on first run" -ForegroundColor Yellow
    Write-Host "Or place your own ONNX model file in the models directory" -ForegroundColor Yellow
}

# Display system information
Write-Host "`nSystem Information:" -ForegroundColor Cyan
Write-Host "- Working Directory: $ScriptRoot" -ForegroundColor White
Write-Host "- Virtual Environment: $venvPath" -ForegroundColor White
Write-Host "- Python Version: $pythonVersion" -ForegroundColor White
Write-Host "- Config File: $(if (Test-Path 'config.yaml') { 'Found' } else { 'Not found (using defaults)' })" -ForegroundColor White
Write-Host "- Model File: $(if (Test-Path $modelPath) { 'Found' } else { 'Will download automatically' })" -ForegroundColor White

# Check if FFmpeg is available (optional but recommended)
try {
    $ffmpegVersion = ffmpeg -version 2>&1 | Select-String "ffmpeg version" | Select-Object -First 1
    Write-Host "- FFmpeg: Available ($ffmpegVersion)" -ForegroundColor White
} catch {
    Write-Host "- FFmpeg: Not found (RTSP streams may have issues)" -ForegroundColor Yellow
    Write-Host "  Install FFmpeg from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
}

Write-Host "`nStarting People Counter System..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the system" -ForegroundColor Yellow
Write-Host "Press 'q' or ESC in the camera window to quit" -ForegroundColor Yellow
Write-Host "API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "WebSocket endpoint: ws://localhost:8000/ws/live" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Gray

# Start the application
try {
    python main.py
} catch {
    Write-Host "`nERROR: Failed to start the application" -ForegroundColor Red
    Write-Host "Check the error messages above for details" -ForegroundColor Yellow
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "- Camera/video source not accessible" -ForegroundColor Yellow
    Write-Host "- Missing model file" -ForegroundColor Yellow
    Write-Host "- Port 8000 already in use" -ForegroundColor Yellow
    Write-Host "- Configuration errors in config.yaml" -ForegroundColor Yellow
    exit 1
}

Write-Host "`nPeople Counter System stopped." -ForegroundColor Green
