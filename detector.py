"""
ONNX-based person detection module optimized for CPU inference.
Supports both Ultralytics YOLO and raw ONNXRuntime approaches.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU usage

import warnings
warnings.filterwarnings('ignore')

import logging
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

logger = logging.getLogger(__name__)

class Detection:
    """Single detection result."""
    
    def __init__(self, xyxy: np.ndarray, confidence: float, class_id: int = 0, class_name: str = "person"):
        """Initialize detection.
        
        Args:
            xyxy: Bounding box coordinates [x1, y1, x2, y2]
            confidence: Detection confidence score
            class_id: Class ID (0 for person)
            class_name: Human-readable class name
        """
        self.xyxy = xyxy.astype(np.float32)
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get bounding box area."""
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1) * (y2 - y1)

class PersonDetector:
    """ONNX-based multi-object detector with CPU optimization."""
    
    # COCO class mapping for relevant objects
    COCO_CLASSES = {
        0: 'person', 39: 'bottle', 41: 'cup', 63: 'laptop', 67: 'cell phone',
        73: 'book', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }
    
    # Activity-relevant object mapping
    ACTIVITY_OBJECTS = {
        'person': 0, 'bottle': 39, 'cup': 41, 'laptop': 63, 'cell phone': 67, 'book': 73
    }
    
    def __init__(self, 
                 model_path: str = "models/yolo11n.onnx",
                 imgsz: int = 416,  # Optimized for performance
                 conf_threshold: float = 0.4,
                 nms_threshold: float = 0.5,
                 use_ultralytics: bool = True,
                 num_threads: int = 6,
                 detect_objects: bool = True):  # Enable object detection
        """Initialize person detector.
        
        Args:
            model_path: Path to ONNX model file
            imgsz: Input image size (512 chosen for i5-6th-gen CPU sweet spot)
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS IoU threshold
            use_ultralytics: Use Ultralytics wrapper if available
            num_threads: Number of CPU threads for inference
        """
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.num_threads = num_threads
        self.detect_objects = detect_objects
        
        self.model = None
        self.session = None
        self.input_name = None
        self.output_names = None
        
        # Try Ultralytics first, fallback to raw ONNXRuntime
        if use_ultralytics and ULTRALYTICS_AVAILABLE:
            self._init_ultralytics()
        elif ONNXRUNTIME_AVAILABLE:
            self._init_onnxruntime()
        else:
            raise RuntimeError("Neither Ultralytics nor ONNXRuntime available")
    
    def _init_ultralytics(self) -> None:
        """Initialize using Ultralytics YOLO wrapper."""
        try:
            if not self.model_path.exists():
                logger.warning(f"Model not found at {self.model_path}, using default yolo11n")
                self.model = YOLO("yolo11n.pt")  # Will download if needed
                # Export to ONNX for future use
                self.model.export(format="onnx", imgsz=self.imgsz)
            else:
                self.model = YOLO(str(self.model_path))
            
            logger.info(f"Initialized Ultralytics detector with {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ultralytics: {e}")
            if ONNXRUNTIME_AVAILABLE:
                logger.info("Falling back to raw ONNXRuntime")
                self._init_onnxruntime()
            else:
                raise
    
    def _init_onnxruntime(self) -> None:
        """Initialize using raw ONNXRuntime with CPU optimization."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            
            # CPU-optimized session options
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self.num_threads  # 4 cores for i5-6th gen
            sess_options.inter_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create inference session
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(
                str(self.model_path), 
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            input_shape = self.session.get_inputs()[0].shape
            logger.info(f"Initialized ONNXRuntime detector: {input_shape} -> {len(self.output_names)} outputs")
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNXRuntime: {e}")
            raise
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess frame for inference.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (preprocessed tensor, scale, offset)
        """
        # Resize maintaining aspect ratio
        h, w = frame.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize and pad
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create padded image
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        
        # Center the resized image
        y_offset = (self.imgsz - new_h) // 2
        x_offset = (self.imgsz - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        
        return tensor, scale, (x_offset, y_offset)
    
    def postprocess(self, outputs: np.ndarray, scale: float, 
                   offset: Tuple[int, int], orig_shape: Tuple[int, int]) -> List[Detection]:
        """Postprocess model outputs to detections.
        
        Args:
            outputs: Raw model outputs
            scale: Scaling factor used in preprocessing
            offset: Padding offset (x, y)
            orig_shape: Original frame shape (h, w)
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        # Handle different output formats
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # Remove batch dimension
        
        # Extract confidence scores and class IDs
        confidences = outputs[:, 4]
        class_ids = outputs[:, 5].astype(int)
        
        # Filter based on detection mode
        if self.detect_objects:
            # Keep activity-relevant objects
            relevant_classes = list(self.ACTIVITY_OBJECTS.values())
            valid_mask = np.isin(class_ids, relevant_classes) & (confidences >= self.conf_threshold)
        else:
            # Only keep person detections (class_id = 0)
            valid_mask = (class_ids == 0) & (confidences >= self.conf_threshold)
        
        if not np.any(valid_mask):
            return detections
        
        # Filter detections
        valid_outputs = outputs[valid_mask]
        valid_scores = confidences[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        
        # Convert to xyxy format
        boxes = valid_outputs[:, :4].copy()
        if boxes.shape[1] == 4:  # xywh format
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x + w
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y + h
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            valid_scores.tolist(),
            self.conf_threshold,
            self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            
            for idx in indices:
                x1, y1, x2, y2 = boxes[idx]
                conf = valid_scores[idx]
                class_id = valid_class_ids[idx]
                
                # Apply scale and offset corrections
                x1 = (x1 - offset[0]) / scale
                y1 = (y1 - offset[1]) / scale
                x2 = (x2 - offset[0]) / scale
                y2 = (y2 - offset[1]) / scale
                
                # Clamp to frame boundaries
                h, w = orig_shape
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                class_name = self.COCO_CLASSES.get(int(class_id), f"class_{int(class_id)}")
                detection = Detection(
                    xyxy=np.array([x1, y1, x2, y2]),
                    confidence=float(conf),
                    class_id=int(class_id),
                    class_name=class_name
                )
                detections.append(detection)
        
        return detections
    
    def _postprocess_ultralytics(self, results) -> List[Detection]:
        """Postprocess Ultralytics YOLO results to Detection objects.
        
        Args:
            results: Ultralytics YOLO results
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    # Filter based on detection mode
                    if self.detect_objects:
                        # Keep activity-relevant objects
                        relevant_classes = list(self.ACTIVITY_OBJECTS.values())
                        if class_id not in relevant_classes:
                            continue
                    else:
                        # Only keep person detections (class_id = 0)
                        if class_id != 0:
                            continue
                    
                    # Skip low confidence detections
                    if conf < self.conf_threshold:
                        continue
                    
                    class_name = self.COCO_CLASSES.get(int(class_id), f"class_{int(class_id)}")
                    detection = Detection(
                        xyxy=box,
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=class_name
                    )
                    detections.append(detection)
        
        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of Detection objects
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            if self.model:
                # Use Ultralytics
                results = self.model(frame, imgsz=self.imgsz, conf=self.conf_threshold, 
                                   iou=self.nms_threshold, verbose=False)
                return self._postprocess_ultralytics(results)
            elif self.session:
                # Use raw ONNXRuntime
                tensor, scale, offset = self.preprocess(frame)
                orig_shape = frame.shape[:2]
                outputs = self.session.run(self.output_names, {self.input_name: tensor})
                return self.postprocess(outputs[0], scale, offset, orig_shape)
            else:
                logger.error("No model loaded")
                return []
                
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def get_model_info(self) -> dict:
        """Get model information."""
        info = {
            'model_path': str(self.model_path),
            'imgsz': self.imgsz,
            'conf_threshold': self.conf_threshold,
            'nms_threshold': self.nms_threshold,
            'num_threads': self.num_threads,
            'backend': 'ultralytics' if self.model else 'onnxruntime'
        }
        
        if self.session:
            info['providers'] = self.session.get_providers()
            
        return info
