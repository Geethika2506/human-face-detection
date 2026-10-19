"""Enrollment checks with a fake detector and recognizer."""

import tempfile
import unittest

import numpy as np

from enroll import EnrollmentError, reference_from_frame, require_single_face
from face_detection import DetectionResult, Face, FaceDetector, FaceRecognizer
from gallery import Gallery
from model_assets import DETECTOR_SHA256, RECOGNIZER_SHA256
from quality import assess_diversity, assess_reference


FRONTAL = ((30.0, 40.0), (90.0, 40.0), (60.0, 60.0), (40.0, 90.0), (80.0, 90.0))


def sharp_crop() -> np.ndarray:
    crop = np.zeros((120, 120, 3), dtype=np.uint8)
    crop[::2] = 170
    crop[1::2] = 70
    return crop


class ScriptedDetector:
    def __init__(self, rows):
        self.rows = rows
        self.sizes = []

    def setInputSize(self, size):
        self.sizes.append((int(size[0]), int(size[1])))

    def detect(self, image):
        if self.rows is None:
            return 1, None
        return 1, np.asarray(self.rows, dtype=np.float32)


class ScriptedSFace:
    def __init__(self, vector):
        self.vector = vector

    def alignCrop(self, frame, row):
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def feature(self, aligned):
        return self.vector


def yunet_row(x=40, y=40, width=100, height=110, score=0.99) -> np.ndarray:
    return np.array(
        [x, y, width, height, 70, 70, 120, 70, 95, 100, 75, 120, 115, 120, score],
        dtype=np.float32,
    )


class TestQuality(unittest.TestCase):
    def test_accepts_a_clear_frontal_face(self):
        result = assess_reference(sharp_crop(), (0, 0, 120, 120), FRONTAL, 0.95)
        self.assertTrue(result.accepted, result.reasons)

    def test_rejects_blur_size_exposure_and_pose(self):
        blurry = assess_reference(np.full((120, 120, 3), 120, np.uint8), (0, 0, 120, 120), FRONTAL, 0.95)
        self.assertFalse(blurry.accepted)
        self.assertIn("blurry", blurry.reasons[0])

        small = assess_reference(sharp_crop(), (0, 0, 40, 40), FRONTAL, 0.95)
        self.assertIn("too small", small.reasons[0])

        dark = assess_reference(np.zeros((120, 120, 3), np.uint8), (0, 0, 120, 120), FRONTAL, 0.95)
        self.assertTrue(any("dark" in reason for reason in dark.reasons))

        bright = assess_reference(np.full((120, 120, 3), 255, np.uint8), (0, 0, 120, 120), FRONTAL, 0.95)
        self.assertTrue(any("bright" in reason for reason in bright.reasons))

        turned = ((30.0, 40.0), (90.0, 40.0), (20.0, 60.0), (40.0, 90.0), (80.0, 90.0))
        pose = assess_reference(sharp_crop(), (0, 0, 120, 120), turned, 0.95)
        self.assertTrue(any("turned" in reason for reason in pose.reasons))

    def test_diversity_skips_a_near_duplicate(self):
        first = np.zeros(128, dtype=np.float32)
        first[0] = 1.0
        same = assess_diversity(first, [first])
        self.assertFalse(same.accepted)
        other = np.zeros(128, dtype=np.float32)
        other[1] = 1.0
        self.assertTrue(assess_diversity(other, [first]).accepted)


class TestEnrollmentFaces(unittest.TestCase):
    def test_no_face_and_multiple_faces(self):
        with self.assertRaises(EnrollmentError):
            require_single_face(0)
        with self.assertRaises(EnrollmentError):
            require_single_face(2)

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        detector = FaceDetector(backend=ScriptedDetector(None), confidence_threshold=0.5)
        recognizer = FaceRecognizer(gallery=self._gallery(), backend=ScriptedSFace(np.ones((1, 128))))
        _embedding, _crop, reason = reference_from_frame(frame, detector, recognizer, [])
        self.assertIn("No face", reason)

        rows = np.stack([yunet_row(10, 10), yunet_row(120, 10)])
        detector.detector = ScriptedDetector(rows)
        _embedding, _crop, reason = reference_from_frame(frame, detector, recognizer, [])
        self.assertIn("2 faces", reason)

    def test_detector_scales_boxes_back_to_the_full_frame(self):
        backend = ScriptedDetector(yunet_row(10, 20, 30, 40, 0.95).reshape(1, 15))
        detector = FaceDetector(backend=backend, confidence_threshold=0.5, detect_max_side=500)
        frame = np.zeros((400, 1000, 3), dtype=np.uint8)
        result = detector.detect(frame)
        self.assertEqual(backend.sizes, [(500, 200)])
        self.assertEqual(len(result.detected_faces), 1)
        x, y, width, height = result.detected_faces[0].bbox
        self.assertEqual((x, y, width, height), (20, 40, 60, 80))
        self.assertEqual(result.detected_faces[0].landmarks[0], (140.0, 140.0))

    def _gallery(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        return Gallery(self.temporary.name, DETECTOR_SHA256, RECOGNIZER_SHA256)


class TestDetectionResultStillGroupsFaces(unittest.TestCase):
    def test_summary_and_empty_crop(self):
        result = DetectionResult(1, (40, 40, 3))
        self.assertIn("No faces", result.summary())
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            Face(bbox=(0, 0, 10, 10), crop=crop, confidence=1.5, frame_index=1)
        with self.assertRaises(ValueError):
            Face(bbox=(0, 0, 10, 10), crop=np.array([]), confidence=0.5, frame_index=1)


if __name__ == "__main__":
    unittest.main()
