"""
Face Detection Module for Student Attendance System

This module provides face detection capabilities using OpenCV and deep neural networks.
It is designed as the first component of a multi-module attendance system.

Component Structure:
    1. FaceDetector: Detects faces in frames
    2. DetectionResult: Stores detection output (boxes, crops, confidence)
    3. FaceRecognizer: Placeholder for future recognition module
    
Data Flow:
    WebcamCapture -> FaceDetector -> DetectionResult -> [FaceRecognizer (future)]
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging
from pathlib import Path


# Configure logging for detection and recognition separately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)

detection_logger = logging.getLogger('DETECTION')
recognition_logger = logging.getLogger('RECOGNITION')


@dataclass
class Face:
    """
    Represents a detected face with its associated metadata.
    
    Attributes:
        bbox (Tuple[int, int, int, int]): Bounding box as (x, y, width, height)
        crop (np.ndarray): Cropped face image for further processing
        confidence (float): Detection confidence score (0.0 to 1.0)
        frame_index (int): Index of the frame in which face was detected
    """
    bbox: Tuple[int, int, int, int]
    crop: np.ndarray
    confidence: float
    frame_index: int
    
    # Placeholder for recognition results (will be populated by FaceRecognizer)
    recognized_student_id: Optional[str] = None
    recognized_name: Optional[str] = None
    recognition_confidence: Optional[float] = None
    
    def __post_init__(self):
        """Validate face data on initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.crop is None or self.crop.size == 0:
            raise ValueError("Face crop cannot be empty")


class DetectionResult:
    """
    Container for face detection results from a single frame.
    Stores both detection and recognition data separately for clarity.
    
    This class bridges detection output with recognition module input.
    """
    
    def __init__(self, frame_index: int, frame_shape: Tuple[int, int, int]):
        """
        Initialize detection result.
        
        Args:
            frame_index: Index of the processed frame
            frame_shape: Original frame shape (height, width, channels)
        """
        self.frame_index = frame_index
        self.frame_shape = frame_shape
        self.detected_faces: List[Face] = []
        self.detection_timestamps = []
    
    def add_face(self, face: Face) -> None:
        """Add a detected face to results."""
        self.detected_faces.append(face)
    
    def get_faces_for_recognition(self) -> List[Face]:
        """
        Get all detected faces formatted for recognition module input.
        
        Returns:
            List of Face objects ready for recognition processing.
            
        Note:
            This method is called by the FaceRecognizer to process detection results.
            Each Face object contains the crop and metadata needed for recognition.
        """
        return self.detected_faces
    
    def summary(self) -> str:
        """Return a summary of detection results."""
        return (
            f"Frame {self.frame_index}: "
            f"{len(self.detected_faces)} face(s) detected. "
            f"Avg confidence: "
            f"{np.mean([f.confidence for f in self.detected_faces]):.2f}" 
            if self.detected_faces else "No faces detected"
        )


class FaceDetector:
    """
    Face detection using OpenCV DNN module with pre-trained ResNet model.
    
    This detector uses a deep neural network trained on face detection,
    providing more reliable detections compared to Haar Cascades.
    
    The output (DetectionResult) is structured to easily pass to downstream
    recognition and attendance marking modules.
    """
    
    # Pre-trained model paths (included with OpenCV)
    PROTO_PATH = cv2.data.haarcascades + 'lbpcascade_frontalface_improved.xml'
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize the face detector.
        
        Args:
            confidence_threshold: Minimum confidence score to consider a detection valid (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        
        # Load cascade classifier (alternative method, fast but less accurate)
        try:
            self.cascade = cv2.CascadeClassifier(self.PROTO_PATH)
            if self.cascade.empty():
                detection_logger.warning("Cascade classifier failed to load, will use alternative method")
                self.cascade = None
        except Exception as e:
            detection_logger.error(f"Error loading cascade: {e}")
            self.cascade = None
        
        # Modern DNN-based detector setup (more accurate)
        self.setup_dnn_detector()
        
        detection_logger.info(
            f"FaceDetector initialized with confidence threshold: {confidence_threshold}"
        )
    
    def setup_dnn_detector(self) -> None:
        """
        Set up deep neural network-based face detector.
        Uses the Caffe model for face detection (included with OpenCV).
        """
        try:
            # OpenCV's pre-trained DNN models for face detection
            opencv_dir = cv2.data.haarcascades.replace('haarcascades', 'dnn')
            
            # Try to load DNN model
            model_file = Path(opencv_dir).parent / 'dnn' / 'opencv_face_detector_uint8.pb'
            config_file = Path(opencv_dir).parent / 'dnn' / 'opencv_face_detector.pbtxt'
            
            if model_file.exists() and config_file.exists():
                self.net = cv2.dnn.readNetFromTensorflow(str(model_file), str(config_file))
                self.use_dnn = True
                detection_logger.info("DNN detector loaded successfully")
            else:
                detection_logger.info("DNN model files not found, will use Cascade Classifier")
                self.use_dnn = False
                self.net = None
        except Exception as e:
            detection_logger.warning(f"DNN setup failed: {e}, falling back to Cascade Classifier")
            self.use_dnn = False
            self.net = None
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect faces in a single frame.
        
        Args:
            frame: Input frame (BGR color image from OpenCV)
        
        Returns:
            DetectionResult containing all detected faces with confidence scores
        
        Flow to Recognition:
            1. FaceDetector.detect() returns DetectionResult
            2. DetectionResult.get_faces_for_recognition() extracts Face objects
            3. Face.crop images are sent to FaceRecognizer for identity matching
            4. Recognition results populate Face.recognized_student_id, etc.
        """
        self.frame_count += 1
        result = DetectionResult(self.frame_count, frame.shape)
        
        # Convert to grayscale for cascade detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using best available method
        if self.use_dnn and self.net is not None:
            detected_boxes = self._detect_with_dnn(frame, gray)
        elif self.cascade is not None:
            detected_boxes = self._detect_with_cascade(gray)
        else:
            detection_logger.warning("No detection method available")
            return result
        
        # Process detections and create Face objects
        for (x, y, w, h, conf) in detected_boxes:
            if conf >= self.confidence_threshold:
                # Ensure bounding box is within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                # Extract face crop for recognition module
                face_crop = frame[y:y+h, x:x+w].copy()
                
                # Create Face object
                face = Face(
                    bbox=(x, y, w, h),
                    crop=face_crop,
                    confidence=conf,
                    frame_index=self.frame_count
                )
                
                result.add_face(face)
                detection_logger.debug(
                    f"Face detected - Box: ({x}, {y}, {w}, {h}), "
                    f"Confidence: {conf:.4f}"
                )
        
        detection_logger.info(f"Frame {self.frame_count}: {len(result.detected_faces)} face(s) detected")
        
        return result
    
    def _detect_with_cascade(self, gray: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using Haar Cascade Classifier.
        
        Returns:
            List of (x, y, w, h, confidence) tuples
        """
        boxes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list with normalized confidence (cascade doesn't return confidence)
        detections = []
        for (x, y, w, h) in boxes:
            detections.append((x, y, w, h, 0.85))  # Use default confidence for cascade
        
        return detections
    
    def _detect_with_dnn(self, frame: np.ndarray, gray: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using DNN (more accurate than Cascade).
        
        Returns:
            List of (x, y, w, h, confidence) tuples
        """
        detections = []
        
        try:
            h, w = frame.shape[:2]
            
            # Prepare blob for network input
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, (300, 300),
                [104.0, 117.0, 123.0], False, False
            )
            
            self.net.setInput(blob)
            result = self.net.forward()
            
            # Parse detections with confidence scores
            for detection in result[0][0]:
                confidence = detection[2]
                
                if confidence > self.confidence_threshold:
                    x1 = int(detection[3] * w)
                    y1 = int(detection[4] * h)
                    x2 = int(detection[5] * w)
                    y2 = int(detection[6] * h)
                    
                    box_w = x2 - x1
                    box_h = y2 - y1
                    
                    detections.append((x1, y1, box_w, box_h, float(confidence)))
        
        except Exception as e:
            detection_logger.error(f"Error in DNN detection: {e}")
        
        return detections
    
    def get_stats(self) -> dict:
        """Return detection statistics."""
        return {
            'total_frames_processed': self.frame_count,
            'confidence_threshold': self.confidence_threshold,
            'detector_type': 'DNN' if self.use_dnn else 'Cascade'
        }


class FaceRecognizer:
    """
    Placeholder for face recognition module (to be implemented later).
    
    This class will be implemented in the next phase to:
    1. Accept DetectionResult objects from FaceDetector
    2. Process Face.crop images for student identification
    3. Populate Face.recognized_student_id, Face.recognized_name, 
       and Face.recognition_confidence
    4. Log recognition results separately from detection
    
    Integration Flow:
        detector = FaceDetector()
        recognizer = FaceRecognizer(model_path='...')
        
        frame = capture_frame()
        detection_result = detector.detect(frame)
        
        # Pass detected faces to recognizer
        for face in detection_result.get_faces_for_recognition():
            student_id, name, rec_confidence = recognizer.recognize(face.crop)
            face.recognized_student_id = student_id
            face.recognized_name = name
            face.recognition_confidence = rec_confidence
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize face recognizer (stub for future implementation).
        
        Args:
            model_path: Path to pre-trained recognition model (e.g., face embedding model)
        """
        self.model_path = model_path
        self.model = None
        
        recognition_logger.info("FaceRecognizer initialized (placeholder - not yet implemented)")
    
    def recognize(self, face_crop: np.ndarray) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """
        Recognize a student from a face crop.
        
        This method will be implemented to:
        1. Extract face embeddings from the crop
        2. Compare against student database
        3. Return student ID, name, and confidence
        
        Args:
            face_crop: Cropped face image from FaceDetector.Face.crop
        
        Returns:
            Tuple of (student_id, student_name, confidence_score)
            Currently returns (None, None, None) as placeholder
        
        Example Future Implementation:
            def recognize(self, face_crop):
                embedding = self.model.get_embedding(face_crop)
                student_id, confidence = self.db.find_closest_match(embedding)
                student_name = self.db.get_student_name(student_id)
                recognition_logger.info(
                    f"Recognized: {student_name} (ID: {student_id}, "
                    f"Confidence: {confidence:.4f})"
                )
                return student_id, student_name, confidence
        """
        recognition_logger.warning("FaceRecognizer.recognize() not yet implemented")
        return None, None, None
    
    def batch_recognize(self, detection_result: DetectionResult) -> DetectionResult:
        """
        Batch process multiple detected faces (future implementation).
        
        This method will process entire DetectionResult objects, populating
        recognition data for each Face object.
        
        Args:
            detection_result: DetectionResult from FaceDetector.detect()
        
        Returns:
            Same DetectionResult with recognition fields populated
        """
        for face in detection_result.get_faces_for_recognition():
            student_id, name, rec_conf = self.recognize(face.crop)
            face.recognized_student_id = student_id
            face.recognized_name = name
            face.recognition_confidence = rec_conf
        
        return detection_result


class AttendanceLogger:
    """
    Centralized logger for attendance system events.
    Keeps detection and recognition logs separate for debugging and auditing.
    """
    
    @staticmethod
    def log_detection_event(frame_idx: int, num_faces: int, avg_confidence: float) -> None:
        """Log face detection event."""
        detection_logger.info(
            f"Frame {frame_idx}: Detected {num_faces} face(s), "
            f"avg confidence: {avg_confidence:.4f}"
        )
    
    @staticmethod
    def log_recognition_event(student_id: str, confidence: float) -> None:
        """Log face recognition event."""
        recognition_logger.info(
            f"Recognized student: {student_id}, confidence: {confidence:.4f}"
        )
    
    @staticmethod
    def log_attendance_mark(student_id: str, timestamp: str) -> None:
        """Log attendance marking event."""
        recognition_logger.info(f"Attendance marked for student: {student_id} at {timestamp}")
