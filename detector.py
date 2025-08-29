"""
ONNX-based person detection module optimized for CPU inference.
Supports both Ultralytics YOLO and raw ONNXRuntime approaches.
"""
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
    
    def __init__(self, xyxy: np.ndarray, confidence: float, class_id: int = 0):
        """Initialize detection.
        
        Args:
            xyxy: Bounding box coordinates [x1, y1, x2, y2]
            confidence: Detection confidence score
            class_id: Class ID (0 for person)
        """
        self.xyxy = xyxy.astype(np.float32)
        self.confidence = confidence
        self.class_id = class_id
    
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
    """ONNX-based person detector with CPU optimization."""
    
    def __init__(self, 
                 model_path: str = "models/yolo11n.onnx",
                 imgsz: int = 512,  # CPU sweet spot for i5-6th gen
                 conf_threshold: float = 0.35,
                 nms_threshold: float = 0.45,
                 use_ultralytics: bool = True,
                 num_threads: int = 4):  # 4 cores for i5-6th gen
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
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for inference.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed tensor ready for inference
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
        
        # Filter by confidence and class (person = 0)
        if outputs.shape[1] > 5:  # Has class predictions
            confidences = outputs[:, 4]
            class_probs = outputs[:, 5:]
            person_scores = confidences * class_probs[:, 0]  # Person class
            valid_mask = person_scores > self.conf_threshold
        else:
            # Simple format: [x, y, w, h, conf]
            person_scores = outputs[:, 4]
            valid_mask = person_scores > self.conf_threshold
        
        if not np.any(valid_mask):
            return detections
        
        # Filter detections
        valid_outputs = outputs[valid_mask]
        valid_scores = person_scores[valid_mask]
        
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
            
            for i in indices:
                box = boxes[i]
                conf = valid_scores[i]
                
                # Convert back to original coordinates
                x1, y1, x2, y2 = box
                x1 = (x1 - offset[0]) / scale
                y1 = (y1 - offset[1]) / scale
                x2 = (x2 - offset[0]) / scale
                y2 = (y2 - offset[1]) / scale
                
                # Clip to frame bounds
                h, w = orig_shape
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                detection = Detection(
                    xyxy=np.array([x1, y1, x2, y2]),
                    confidence=float(conf),
                    class_id=0
                )
                detections.append(detection)
        
        return detections
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect persons in frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of Detection objects
        """
        if frame is None or frame.size == 0:
            return []
        
        orig_shape = frame.shape[:2]
        
        try:
            if self.model is not None:  # Ultralytics
                results = self.model(frame, imgsz=self.imgsz, conf=self.conf_threshold, verbose=False)
                
                detections = []
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        
                        # Filter for person class (0)
                        person_mask = classes == 0
                        
                        for box, conf in zip(boxes[person_mask], confidences[person_mask]):
                            detection = Detection(
                                xyxy=box,
                                confidence=float(conf),
                                class_id=0
                            )
                            detections.append(detection)
                
                return detections
                
            elif self.session is not None:  # Raw ONNXRuntime
                # Preprocess
                tensor, scale, offset = self.preprocess(frame)
                
                # Inference
                outputs = self.session.run(self.output_names, {self.input_name: tensor})
                
                # Postprocess
                return self.postprocess(outputs[0], scale, offset, orig_shape)
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
        
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
