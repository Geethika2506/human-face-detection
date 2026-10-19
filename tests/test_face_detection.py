"""Unit tests for detection results and the YuNet / SFace wrappers.

Run from the repository root:

    python -m pytest tests/test_face_detection.py -v
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_detection import (
    Face, FaceDetector, DetectionResult,
    FaceRecognizer, AttendanceLogger
)


class TestFaceDataclass(unittest.TestCase):
    """Test Face dataclass."""
    
    def setUp(self):
        """Create test data."""
        self.test_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_face_creation_valid(self):
        """Test creating a valid Face object."""
        face = Face(
            bbox=(10, 20, 100, 120),
            crop=self.test_crop,
            confidence=0.95,
            frame_index=1
        )
        
        self.assertEqual(face.bbox, (10, 20, 100, 120))
        self.assertEqual(face.confidence, 0.95)
        self.assertEqual(face.frame_index, 1)
        self.assertIsNone(face.recognized_student_id)
    
    def test_face_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with self.assertRaises(ValueError):
            Face(
                bbox=(10, 20, 100, 120),
                crop=self.test_crop,
                confidence=1.5,  # Invalid
                frame_index=1
            )
    
    def test_face_empty_crop(self):
        """Test that empty crop raises error."""
        empty_crop = np.array([], dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            Face(
                bbox=(10, 20, 100, 120),
                crop=empty_crop,
                confidence=0.95,
                frame_index=1
            )


class TestDetectionResult(unittest.TestCase):
    """Test DetectionResult class."""
    
    def setUp(self):
        """Create test data."""
        self.test_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.frame_shape = (720, 1280, 3)
        self.result = DetectionResult(frame_index=1, frame_shape=self.frame_shape)
    
    def test_add_face(self):
        """Test adding faces to result."""
        face = Face(
            bbox=(10, 20, 100, 120),
            crop=self.test_crop,
            confidence=0.95,
            frame_index=1
        )
        
        self.result.add_face(face)
        
        self.assertEqual(len(self.result.detected_faces), 1)
        self.assertEqual(self.result.detected_faces[0], face)
    
    def test_get_faces_for_recognition(self):
        """Test retrieving faces for recognition."""
        faces = []
        for i in range(3):
            face = Face(
                bbox=(10 + i*50, 20, 100, 120),
                crop=self.test_crop,
                confidence=0.90 + i*0.03,
                frame_index=1
            )
            faces.append(face)
            self.result.add_face(face)
        
        recognition_faces = self.result.get_faces_for_recognition()
        self.assertEqual(len(recognition_faces), 3)
    
    def test_summary(self):
        """Test summary generation."""
        # No faces
        summary = self.result.summary()
        self.assertIn("No faces detected", summary)
        
        # Add faces
        for i in range(2):
            face = Face(
                bbox=(10 + i*50, 20, 100, 120),
                crop=self.test_crop,
                confidence=0.9,
                frame_index=1
            )
            self.result.add_face(face)
        
        summary = self.result.summary()
        self.assertIn("2 face", summary)


class TestFaceDetector(unittest.TestCase):
    """Test FaceDetector class."""
    
    def setUp(self):
        """Initialize detector."""
        self.detector = FaceDetector(confidence_threshold=0.5)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.confidence_threshold, 0.5)
        self.assertEqual(self.detector.frame_count, 0)
    
    def test_detector_with_invalid_threshold(self):
        """An out-of-range detector threshold is rejected."""
        with self.assertRaises(ValueError):
            FaceDetector(confidence_threshold=1.5)
    
    def test_detect_empty_frame(self):
        """Test detection on empty frame."""
        # Create blank frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = self.detector.detect(frame)
        
        self.assertEqual(len(result.detected_faces), 0)
        self.assertEqual(self.detector.frame_count, 1)
    
    def test_detect_multiple_frames(self):
        """Test detection on multiple frames."""
        for i in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = self.detector.detect(frame)
            self.assertEqual(result.frame_index, i + 1)
        
        self.assertEqual(self.detector.frame_count, 5)
    
    def test_detector_stats(self):
        """Test getting detector statistics."""
        stats = self.detector.get_stats()
        
        self.assertIn('total_frames_processed', stats)
        self.assertIn('confidence_threshold', stats)
        self.assertIn('detector_type', stats)
        self.assertEqual(stats['total_frames_processed'], 0)


class TestFaceRecognizer(unittest.TestCase):
    """Recognizer fields stay empty when a face cannot be aligned."""

    def setUp(self):
        self.recognizer = FaceRecognizer()
        self.test_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_unaligned_face_is_unknown(self):
        result = DetectionResult(1, (480, 640, 3))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(3):
            result.add_face(
                Face(
                    bbox=(10 + i * 50, 20, 100, 120),
                    crop=self.test_crop,
                    confidence=0.9,
                    frame_index=1,
                )
            )
        result = self.recognizer.batch_recognize(result, frame)
        for face in result.detected_faces:
            self.assertEqual(face.recognition_status, "unknown")
            self.assertIsNone(face.recognized_student_id)
            self.assertIsNone(face.recognized_name)
            self.assertIsNone(face.recognition_confidence)


class TestWithRealImages(unittest.TestCase):
    """Test with real images (if available)."""
    
    def test_detect_with_test_image(self):
        """Test detection with a test image."""
        # Create a synthetic test image with faces
        # (In real scenario, would use actual face images)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector = FaceDetector(confidence_threshold=0.5)
        result = detector.detect(frame)
        
        # Should return DetectionResult object
        self.assertIsInstance(result, DetectionResult)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_detection_to_recognition_pipeline(self):
        """Test data flow from detection to recognition."""
        detector = FaceDetector(confidence_threshold=0.5)
        recognizer = FaceRecognizer()
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Step 1: Detect
        detection_result = detector.detect(frame)
        self.assertIsInstance(detection_result, DetectionResult)
        
        # Step 2: Get faces for recognition
        faces_for_rec = detection_result.get_faces_for_recognition()
        self.assertIsInstance(faces_for_rec, list)
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        recognizer.batch_recognize(detection_result, frame)

        for face in detection_result.detected_faces:
            # All faces should have these attributes
            self.assertTrue(hasattr(face, 'recognized_student_id'))
            self.assertTrue(hasattr(face, 'recognized_name'))
            self.assertTrue(hasattr(face, 'recognition_confidence'))


class TestPerformance(unittest.TestCase):
    """Performance tests."""
    
    def test_detection_performance(self):
        """Test detection speed."""
        import time
        
        detector = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Warmup
        detector.detect(frame)
        
        # Time detection
        start = time.time()
        for _ in range(10):
            detector.detect(frame)
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        print(f"\nAverage detection time: {avg_time*1000:.2f} ms")
        
        # Should be reasonably fast
        self.assertLess(avg_time, 1.0)  # Less than 1 second per frame


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_detector_with_different_image_sizes(self):
        """Test detector with various image sizes."""
        detector = FaceDetector()
        
        sizes = [(240, 320), (480, 640), (1080, 1920)]
        
        for h, w in sizes:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            result = detector.detect(frame)
            self.assertIsInstance(result, DetectionResult)
    
    def test_detector_with_grayscale_frame(self):
        """Test detector with grayscale frame (should be BGR)."""
        detector = FaceDetector()
        
        # Create grayscale frame (incorrect input)
        frame = np.zeros((480, 640), dtype=np.uint8)
        
        # Should handle or raise appropriate error
        try:
            result = detector.detect(frame)
            self.assertIsInstance(result, DetectionResult)
        except (ValueError, cv2.error):
            # Expected error for incorrect input format
            pass
    
    def test_zero_confidence_threshold(self):
        """Test detector with zero confidence threshold."""
        detector = FaceDetector(confidence_threshold=0.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(frame)
        self.assertIsInstance(result, DetectionResult)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("="*70)
    print("FACE DETECTION MODULE - UNIT TESTS")
    print("="*70)
    run_tests()
