"""Smoke check for the real ONNX files. Skipped until they have been downloaded."""

import unittest

import numpy as np

from face_detection import FaceDetector, FaceRecognizer
from gallery import Gallery
from model_assets import (
    DETECTOR_FILENAME,
    DETECTOR_SHA256,
    RECOGNIZER_FILENAME,
    RECOGNIZER_SHA256,
    ModelAssetError,
    model_path,
    verify_model,
)


def models_ready() -> bool:
    try:
        verify_model(model_path(DETECTOR_FILENAME), DETECTOR_SHA256)
        verify_model(model_path(RECOGNIZER_FILENAME), RECOGNIZER_SHA256)
        return True
    except ModelAssetError:
        return False


@unittest.skipUnless(models_ready(), "YuNet and SFace weights are not downloaded")
class TestRealModels(unittest.TestCase):
    def test_blank_frame_and_embedding_shape(self):
        detector = FaceDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(frame)
        self.assertEqual(len(result.detected_faces), 0)
        self.assertEqual(detector.get_stats()["detector_type"], "YuNet")

        recognizer = FaceRecognizer(gallery=Gallery(
            "data/gallery",
            DETECTOR_SHA256,
            RECOGNIZER_SHA256,
        ))
        row = np.array(
            [400, 200, 180, 220, 450, 270, 540, 270, 495, 330, 460, 380, 540, 380, 0.99],
            dtype=np.float32,
        )
        embedding = recognizer.embed(frame, row)
        self.assertEqual(embedding.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
