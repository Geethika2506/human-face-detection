# Student Attendance System - Face Detection Module

A modular, production-ready face detection system designed as the first component of a complete student attendance system. This module detects faces in real-time webcam feeds and prepares the data for integration with face recognition and attendance marking modules.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   ATTENDANCE SYSTEM PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Webcam/Video] → [FaceDetector] → [FaceRecognizer] → [Logger] │
│                        ↓                                         │
│                  Detection Results                              │
│                  (confidence, bbox,                             │
│                   crop ready for                                │
│                   recognition)                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### ✅ Implemented (Detection Module)
- **Real-time face detection** using OpenCV DNN module
- **Confidence scoring** for each detected face (0.0 - 1.0)
- **Bounding box extraction** (x, y, width, height)
- **Face crops** ready for recognition
- **Frame-indexed detection results**
- **Separate logging** for detection events
- **Performance metrics** (FPS, processing stats)
- **Modular architecture** ready for extension

### 🔄 Planned (Recognition Module - Next Phase)
- Student face embedding extraction
- Student database matching
- Recognition confidence scores
- Separate recognition logging
- Batch processing of detected faces
- Attendance marking integration

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or video input device

### Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Run the Live Detection Demo

```bash
python demo.py
```

**Command-line options:**
```bash
python demo.py --camera 0 --confidence 0.5
```

**Controls:**
- Press `q` to quit
- Press `s` to save detected face crops to the `saved_faces/` directory

### Expected Output

```
============================================================
DETECTION SUMMARY - Frame 42
============================================================
Total faces detected: 2

Face 1:
  Position: (150, 200)
  Size: 120x140 pixels
  Detection Confidence: 0.9543
  Crop Shape: (140, 120, 3)
  Status: Ready for recognition module ✓

Face 2:
  Position: (500, 180)
  Size: 115x135 pixels
  Detection Confidence: 0.8721
  Crop Shape: (135, 115, 3)
  Status: Ready for recognition module ✓

============================================================
```

## Module Reference

### `face_detection.py`

#### Classes

##### `Face` (Dataclass)
Represents a detected face with its metadata.

**Attributes:**
- `bbox`: Tuple (x, y, width, height) - bounding box coordinates
- `crop`: np.ndarray - cropped face image (3-channel BGR)
- `confidence`: float - detection confidence (0.0-1.0)
- `frame_index`: int - frame number in which face was detected
- `recognized_student_id`: Optional[str] - populated by FaceRecognizer
- `recognized_name`: Optional[str] - populated by FaceRecognizer
- `recognition_confidence`: Optional[float] - populated by FaceRecognizer

```python
from face_detection import Face

# Example Face object created by detector
face = Face(
    bbox=(150, 200, 120, 140),
    crop=cropped_image_array,
    confidence=0.9543,
    frame_index=42
)
```

##### `DetectionResult`
Container for all faces detected in a single frame.

**Key Methods:**
- `add_face(face: Face)` - Add a detected face
- `get_faces_for_recognition()` - Get faces formatted for recognition module
- `summary()` - Get text summary of results

```python
detection_result = detector.detect(frame)
print(detection_result.summary())

# Extract faces for recognition
faces_for_recognition = detection_result.get_faces_for_recognition()
```

##### `FaceDetector`
Main detection engine using deep neural networks.

**Initialization:**
```python
from face_detection import FaceDetector

detector = FaceDetector(confidence_threshold=0.5)
```

**Key Methods:**
- `detect(frame: np.ndarray) -> DetectionResult` - Detect faces in frame
- `get_stats() -> dict` - Get detector statistics

```python
# Detect faces in a frame
detection_result = detector.detect(cv2_frame)

# Access detected faces
for face in detection_result.detected_faces:
    print(f"Confidence: {face.confidence:.4f}")
    print(f"Bbox: {face.bbox}")
    print(f"Crop size: {face.crop.shape}")
```

##### `FaceRecognizer` (Placeholder)
Stub for the recognition module (to be implemented in Phase 2).

**Future Integration:**
```python
from face_detection import FaceRecognizer

# Initialize recognizer with trained model
recognizer = FaceRecognizer(model_path='./models/face_recognition.pkl')

# Process detected faces
detection_result = detector.detect(frame)
detection_result = recognizer.batch_recognize(detection_result)

# Access recognition results
for face in detection_result.detected_faces:
    print(f"Student: {face.recognized_name}")
    print(f"Student ID: {face.recognized_student_id}")
    print(f"Recognition Confidence: {face.recognition_confidence:.4f}")
```

##### `AttendanceLogger`
Centralized logging for detection and recognition events.

```python
from face_detection import AttendanceLogger

AttendanceLogger.log_detection_event(frame_idx=42, num_faces=3, avg_confidence=0.92)
AttendanceLogger.log_recognition_event(student_id='S001', confidence=0.95)
AttendanceLogger.log_attendance_mark(student_id='S001', timestamp='2024-01-15 09:30:00')
```

## Usage Examples

### Example 1: Basic Face Detection in Video

```python
import cv2
from face_detection import FaceDetector

def detect_faces_in_video(video_path):
    detector = FaceDetector(confidence_threshold=0.6)
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        result = detector.detect(frame)
        
        # Process each face
        for face in result.detected_faces:
            x, y, w, h = face.bbox
            confidence = face.confidence
            
            # Draw box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{confidence:.3f}', (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Detections', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

detect_faces_in_video('student_video.mp4')
```

### Example 2: Batch Processing with Saved Crops

```python
import cv2
import os
from face_detection import FaceDetector

def save_face_dataset(video_path, output_dir='faces_dataset'):
    os.makedirs(output_dir, exist_ok=True)
    detector = FaceDetector(confidence_threshold=0.85)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    face_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        result = detector.detect(frame)
        
        for face in result.detected_faces:
            filename = os.path.join(
                output_dir,
                f'face_{face_index}_frame_{frame_count}_conf_{face.confidence:.2f}.jpg'
            )
            cv2.imwrite(filename, face.crop)
            face_index += 1
            
            print(f"Saved: {filename}")
    
    cap.release()
    print(f"Total faces extracted: {face_index}")

save_face_dataset('classroom_video.mp4')
```

### Example 3: Integration Point for Recognition

```python
import cv2
from face_detection import FaceDetector, FaceRecognizer

def process_frame_with_recognition(frame):
    """
    Demonstrates the data flow from detection to recognition.
    This shows where the recognition module will be integrated.
    """
    # Step 1: Detect faces
    detector = FaceDetector(confidence_threshold=0.5)
    detection_result = detector.detect(frame)
    
    print(f"Step 1 - Detection: {len(detection_result.detected_faces)} faces detected")
    
    # Step 2: [FUTURE] Pass to recognition module
    if detection_result.detected_faces:
        # Extract faces for recognition
        faces_for_recognition = detection_result.get_faces_for_recognition()
        
        print(f"Step 2 - Ready for Recognition:")
        for idx, face in enumerate(faces_for_recognition):
            print(f"  Face {idx}: confidence={face.confidence:.4f}, crop_shape={face.crop.shape}")
            
            # [PHASE 2] This is where recognition will happen:
            # recognizer = FaceRecognizer(model_path='...')
            # student_id, name, rec_conf = recognizer.recognize(face.crop)
            # face.recognized_student_id = student_id
            # face.recognized_name = name
            # face.recognition_confidence = rec_conf
    
    return detection_result

# Test with a frame
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    result = process_frame_with_recognition(frame)
cap.release()
```

## Logging

The system uses separate loggers for detection and recognition to enable independent monitoring:

```
2024-01-15 09:30:23,456 - [DETECTION] - INFO - FaceDetector initialized with confidence threshold: 0.5
2024-01-15 09:30:24,123 - [DETECTION] - INFO - Frame 1: 2 face(s) detected
2024-01-15 09:30:24,456 - [RECOGNITION] - INFO - FaceRecognizer initialized (placeholder - not yet implemented)
```

## Next Steps: Recognition Module Implementation

### Phase 2 Implementation Checklist

1. **Model Selection**: Choose a face recognition model
   - OpenFace, FaceNet, ArcFace, or VGGFace2
   - Transfer learning approaches

2. **Student Database**: Set up student profile storage
   - Face embeddings for each student
   - Student metadata (ID, name, class)
   - Embedding similarity thresholds

3. **Recognition Algorithm**:
   - Extract face embeddings from crops
   - Compare against student database
   - Implement confidence matching logic

4. **Implement FaceRecognizer class**:
   ```python
   def recognize(self, face_crop: np.ndarray) -> Tuple[str, str, float]:
       """
       Returns: (student_id, student_name, confidence)
       """
       # Extract embedding
       embedding = self.model.get_embedding(face_crop)
       
       # Find closest match in database
       student_id, confidence = self.db.find_closest_match(embedding)
       student_name = self.db.get_student_name(student_id)
       
       return student_id, student_name, confidence
   ```

5. **Attendance Recording**:
   - Timestamp marking
   - Database persistence
   - Entry/exit tracking

## Performance Tips

- **Resolution**: Higher frame resolution increases detection accuracy but reduces FPS
- **Confidence Threshold**: Adjust based on your use case
  - 0.7+: High precision, fewer false positives
  - 0.5-0.7: Balanced
  - <0.5: High recall, more false positives
- **Frame Skip**: Process every Nth frame for better performance
- **GPU Acceleration**: OpenCV with CUDA support for real-time processing

## Troubleshooting

### No faces detected
- Check lighting conditions
- Increase confidence threshold
- Ensure camera angle captures full faces

### Low FPS
- Reduce frame resolution
- Use cascade classifier instead of DNN (faster but less accurate)
- Skip frames: `if frame_count % 2 == 0: detect(frame)`

### Camera not opening
- Check camera permissions
- Try different camera index: `--camera 1` or `--camera 2`

## File Structure

```
attendance-system/
├── face_detection.py      # Core detection module
├── demo.py                # Live webcam demo
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── saved_faces/          # Directory for saved face crops
```

## License

This module is designed for educational and institutional use.

## Contributing

To extend this system:
1. Implement the `FaceRecognizer` class
2. Add attendance database module
3. Build web dashboard
4. Implement batch processing for large classes

---

**Status**: Phase 1 Complete ✅ - Detection Module
**Next Phase**: Face Recognition & Student Matching
