"""Gallery storage, model hashes, and recognizer wiring with a fake SFace."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from face_detection import Face, FaceRecognizer
from gallery import Gallery, GalleryError, ModelVersionError
from matching import l2_normalize
from model_assets import DETECTOR_SHA256, RECOGNIZER_SHA256, ModelAssetError, verify_model


def unit_matrix(rows: int = 2) -> np.ndarray:
    matrix = np.zeros((rows, 128), dtype=np.float32)
    for index in range(rows):
        matrix[index, index] = 1.0
    return matrix


class FakeSFace:
    def __init__(self, scale: float = 4.0):
        self.scale = scale

    def alignCrop(self, frame, row):
        return np.zeros((112, 112, 3), dtype=np.uint8)

    def feature(self, aligned):
        return np.ones((1, 128), dtype=np.float32) * self.scale


class FakeSFaceVector(FakeSFace):
    def __init__(self, vector):
        super().__init__()
        self.vector = np.asarray(vector, dtype=np.float32).reshape(1, 128)

    def feature(self, aligned):
        return self.vector


class TestGallery(unittest.TestCase):
    def test_save_list_and_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            gallery = Gallery(temporary, DETECTOR_SHA256, RECOGNIZER_SHA256)
            gallery.save("ada", "Ada Lovelace", unit_matrix(), opencv_version="4.14.0")
            listed = gallery.list_profiles()
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]["embedding_count"], 2)
            self.assertTrue(listed[0]["model_match"])
            self.assertFalse(listed[0]["images_retained"])
            self.assertFalse((Path(temporary) / "ada" / "images").exists())

            gallery.delete("ada")
            self.assertEqual(gallery.list_profiles(), [])
            with self.assertRaises(GalleryError):
                gallery.delete("ada")

    def test_replace_and_reject_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            gallery = Gallery(temporary, DETECTOR_SHA256, RECOGNIZER_SHA256)
            gallery.save("ada", "Ada", unit_matrix(1), opencv_version="4.14.0")
            with self.assertRaises(GalleryError):
                gallery.save("ada", "Ada", unit_matrix(1), opencv_version="4.14.0")
            gallery.save("ada", "Ada B", unit_matrix(3), opencv_version="4.14.0", replace=True)
            profiles = gallery.list_profiles()
            self.assertEqual(profiles[0]["display_name"], "Ada B")
            self.assertEqual(profiles[0]["embedding_count"], 3)

    def test_model_version_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            Gallery(temporary, DETECTOR_SHA256, RECOGNIZER_SHA256).save(
                "ada", "Ada", unit_matrix(1), opencv_version="4.14.0"
            )
            shifted = Gallery(temporary, DETECTOR_SHA256, "0" * 64)
            self.assertFalse(shifted.list_profiles()[0]["model_match"])
            with self.assertRaises(ModelVersionError):
                shifted.identity_records()

    def test_invalid_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            gallery = Gallery(temporary, DETECTOR_SHA256, RECOGNIZER_SHA256)
            with self.assertRaises(GalleryError):
                gallery.save("../ada", "Ada", unit_matrix(1), opencv_version="4.14.0")
            with self.assertRaises(GalleryError):
                gallery.save("ada", "   ", unit_matrix(1), opencv_version="4.14.0")


class TestModelFiles(unittest.TestCase):
    def test_missing_and_corrupt_files(self):
        with self.assertRaises(ModelAssetError):
            verify_model(Path("/tmp/does-not-exist-yunet.onnx"), DETECTOR_SHA256)
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "face.onnx"
            bad.write_bytes(b"not an onnx model")
            with self.assertRaises(ModelAssetError):
                verify_model(bad, DETECTOR_SHA256)

    def test_loader_wraps_create_failure(self):
        from face_detection import FaceDetector

        with patch("face_detection.verify_model", return_value=DETECTOR_SHA256), patch(
            "face_detection.cv2.FaceDetectorYN.create", side_effect=RuntimeError("corrupt onnx")
        ):
            with self.assertRaises(ModelAssetError):
                FaceDetector(model_path="unused.onnx")


class TestRecognizerEmbeddings(unittest.TestCase):
    def test_embed_normalizes_and_empty_gallery_is_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            gallery = Gallery(temporary, DETECTOR_SHA256, RECOGNIZER_SHA256)
            recognizer = FaceRecognizer(gallery=gallery, backend=FakeSFace())
            frame = np.zeros((80, 80, 3), dtype=np.uint8)
            row = np.zeros(15, dtype=np.float32)
            embedding = recognizer.embed(frame, row)
            self.assertEqual(embedding.shape, (128,))
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)
            self.assertTrue(np.allclose(embedding, l2_normalize(np.ones(128))))

            face = Face(bbox=(0, 0, 80, 80), crop=frame, confidence=0.99, frame_index=1, yunet_row=row)
            decision = recognizer.match_face(face, frame)
            self.assertEqual(decision.status, "unknown")
            self.assertIsNone(face.recognized_name)
            self.assertIsNone(face.recognized_student_id)
            self.assertIsNone(face.recognition_confidence)
            self.assertIsNone(face.match_score)

    def test_known_match_fills_name_and_leaves_probability_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            gallery = Gallery(temporary, DETECTOR_SHA256, RECOGNIZER_SHA256)
            ada = np.zeros(128, dtype=np.float32)
            ada[0] = 1.0
            bea = np.zeros(128, dtype=np.float32)
            bea[1] = 1.0
            gallery.save("ada", "Ada", ada.reshape(1, 128), opencv_version="4.14.0")
            gallery.save("bea", "Bea", bea.reshape(1, 128), opencv_version="4.14.0")
            recognizer = FaceRecognizer(
                gallery=gallery,
                backend=FakeSFaceVector(ada),
                accept_threshold=0.363,
                match_margin=0.08,
            )
            frame = np.zeros((80, 80, 3), dtype=np.uint8)
            face = Face(
                bbox=(0, 0, 80, 80),
                crop=frame,
                confidence=0.99,
                frame_index=1,
                yunet_row=np.zeros(15, dtype=np.float32),
            )
            decision = recognizer.match_face(face, frame)
            self.assertEqual(decision.status, "known")
            self.assertEqual(face.recognized_person_id, "ada")
            self.assertEqual(face.recognized_student_id, "ada")
            self.assertEqual(face.recognized_name, "Ada")
            self.assertIsNone(face.recognition_confidence)
            self.assertAlmostEqual(face.match_score, 1.0, places=5)

            close = FaceRecognizer(
                gallery=gallery,
                backend=FakeSFaceVector(l2_normalize(ada + bea)),
                accept_threshold=0.363,
                match_margin=0.08,
            )
            ambiguous = Face(
                bbox=(0, 0, 80, 80),
                crop=frame,
                confidence=0.99,
                frame_index=2,
                yunet_row=np.zeros(15, dtype=np.float32),
            )
            uncertain = close.match_face(ambiguous, frame)
            self.assertEqual(uncertain.status, "uncertain")
            self.assertIsNone(ambiguous.recognized_name)
            self.assertIsNone(ambiguous.recognized_student_id)


if __name__ == "__main__":
    unittest.main()
