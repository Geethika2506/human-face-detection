"""Local face detection and recognition.

YuNet finds each face and five landmarks. SFace aligns that face and produces a
128-d embedding. FaceRecognizer compares the embedding with the local gallery.

``confidence`` is the detector score. ``match_score`` is cosine similarity.
``recognition_confidence`` is left unset so it is not read as a probability.
``recognized_name`` and ``recognized_student_id`` are filled only for a known
match. ``recognized_student_id`` stores the enrolled person id.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from gallery import Gallery
from matching import IdentityRecord, MatchDecision, decide_identity, l2_normalize
from model_assets import (
    DETECTOR_FILENAME,
    DETECTOR_SHA256,
    RECOGNIZER_FILENAME,
    RECOGNIZER_SHA256,
    ModelAssetError,
    model_path,
    verify_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
)

detection_logger = logging.getLogger("DETECTION")
recognition_logger = logging.getLogger("RECOGNITION")


@dataclass
class Face:
    """A face found in one frame."""

    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    confidence: float
    frame_index: int
    recognized_student_id: Optional[str] = None
    recognized_name: Optional[str] = None
    recognition_confidence: Optional[float] = None
    recognized_person_id: Optional[str] = None
    recognition_status: Optional[str] = None
    match_score: Optional[float] = None
    match_margin: Optional[float] = None
    landmarks: Optional[tuple[tuple[float, float], ...]] = None
    yunet_row: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.crop is None or self.crop.size == 0:
            raise ValueError("Face crop cannot be empty")


class DetectionResult:
    """Faces found in a single frame."""

    def __init__(self, frame_index: int, frame_shape: tuple[int, ...]):
        self.frame_index = frame_index
        self.frame_shape = frame_shape
        self.detected_faces: list[Face] = []

    def add_face(self, face: Face) -> None:
        self.detected_faces.append(face)

    def get_faces_for_recognition(self) -> list[Face]:
        return self.detected_faces

    def summary(self) -> str:
        if not self.detected_faces:
            return "No faces detected"
        average = float(np.mean([face.confidence for face in self.detected_faces]))
        return (
            f"Frame {self.frame_index}: "
            f"{len(self.detected_faces)} face(s) detected. "
            f"Avg confidence: {average:.2f}"
        )


class FaceDetector:
    """YuNet detector. Boxes and landmarks are returned in full-frame coordinates."""

    def __init__(
        self,
        confidence_threshold: float = config.DETECTION_CONFIDENCE_DEFAULT,
        model_path: str | Path | None = None,
        detect_max_side: int = config.DETECT_MAX_SIDE,
        nms_threshold: float = config.DETECTION_NMS_THRESHOLD,
        top_k: int = config.DETECTION_TOP_K,
        backend=None,
    ):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = confidence_threshold
        self.detect_max_side = detect_max_side
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.frame_count = 0
        self.model_path = Path(model_path) if model_path else model_path_default(DETECTOR_FILENAME)
        self.model_sha256 = DETECTOR_SHA256
        if backend is not None:
            self.detector = backend
        else:
            self.model_sha256 = verify_model(self.model_path, DETECTOR_SHA256)
            self.detector = self._create_detector()
        detection_logger.info(
            "FaceDetector initialized with YuNet (confidence %.2f, max side %s)",
            confidence_threshold,
            detect_max_side,
        )

    def _create_detector(self):
        try:
            detector = cv2.FaceDetectorYN.create(
                str(self.model_path),
                "",
                (320, 320),
                self.confidence_threshold,
                self.nms_threshold,
                self.top_k,
            )
        except Exception as exc:
            raise ModelAssetError(f"Could not load YuNet model: {self.model_path}") from exc
        if detector is None:
            raise ModelAssetError(f"Could not load YuNet model: {self.model_path}")
        return detector

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Detect faces. The YuNet score is stored on Face.confidence."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("detect() expects a BGR image with shape (height, width, 3)")
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError("detect() received an empty image")

        self.frame_count += 1
        result = DetectionResult(self.frame_count, frame.shape)
        resized, scale = _resize_for_detector(frame, self.detect_max_side)
        self.detector.setInputSize((resized.shape[1], resized.shape[0]))
        faces = _unwrap_faces(self.detector.detect(resized))

        for row in faces:
            if row.shape[0] < 15:
                continue
            score = float(row[14])
            if score < self.confidence_threshold:
                continue
            full_row = row.astype(np.float32).copy()
            if scale != 1.0:
                full_row[:14] /= np.float32(scale)
            bbox = _clip_bbox(full_row[:4], frame.shape)
            if bbox is None:
                continue
            x, y, width, height = bbox
            face = Face(
                bbox=bbox,
                crop=frame[y : y + height, x : x + width].copy(),
                confidence=min(max(score, 0.0), 1.0),
                frame_index=self.frame_count,
                landmarks=_landmarks(full_row),
                yunet_row=full_row,
            )
            result.add_face(face)

        detection_logger.debug(
            "Frame %s: %s face(s) detected",
            self.frame_count,
            len(result.detected_faces),
        )
        return result

    def get_stats(self) -> dict:
        return {
            "total_frames_processed": self.frame_count,
            "confidence_threshold": self.confidence_threshold,
            "detector_type": "YuNet",
            "model": self.model_path.name,
        }


class FaceRecognizer:
    """SFace embeddings matched against the local gallery."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        gallery: Gallery | None = None,
        gallery_dir: str | Path | None = None,
        accept_threshold: float = config.ACCEPT_THRESHOLD,
        match_margin: float = config.MATCH_MARGIN,
        backend=None,
    ):
        self.accept_threshold = accept_threshold
        self.match_margin = match_margin
        self.model_path = Path(model_path) if model_path else model_path_default(RECOGNIZER_FILENAME)
        self.opencv_version = cv2.__version__
        if backend is not None:
            self.backend = backend
            self.model_sha256 = RECOGNIZER_SHA256
        else:
            self.model_sha256 = verify_model(self.model_path, RECOGNIZER_SHA256)
            self.backend = self._create_recognizer()
        self.gallery = gallery or Gallery(
            gallery_dir or config.GALLERY_DIR,
            detector_sha256=DETECTOR_SHA256,
            recognizer_sha256=self.model_sha256,
        )
        recognition_logger.info(
            "FaceRecognizer initialized (cosine threshold %.3f, margin %.3f)",
            accept_threshold,
            match_margin,
        )

    def _create_recognizer(self):
        try:
            recognizer = cv2.FaceRecognizerSF.create(str(self.model_path), "")
        except Exception as exc:
            raise ModelAssetError(f"Could not load SFace model: {self.model_path}") from exc
        if recognizer is None:
            raise ModelAssetError(f"Could not load SFace model: {self.model_path}")
        return recognizer

    def embed(self, frame: np.ndarray, yunet_row: np.ndarray) -> np.ndarray:
        """Align with the YuNet row and return an L2-normalized 128-d embedding."""
        if yunet_row is None:
            raise ValueError("A YuNet detection row is required to align the face")
        row = np.asarray(yunet_row, dtype=np.float32).reshape(-1)
        aligned = self.backend.alignCrop(frame, row)
        feature = self.backend.feature(aligned)
        return l2_normalize(feature)

    def match_face(
        self,
        face: Face,
        frame: np.ndarray,
        identities: list[IdentityRecord] | None = None,
    ) -> MatchDecision:
        """Fill recognition fields on ``face`` and return the decision."""
        if identities is None:
            identities = self.gallery.identity_records()
        try:
            embedding = self.embed(frame, face.yunet_row)
            decision = decide_identity(
                embedding,
                identities,
                self.accept_threshold,
                self.match_margin,
            )
        except (ValueError, cv2.error) as exc:
            recognition_logger.warning("Could not embed a face: %s", exc)
            decision = MatchDecision(status="unknown")
        _apply_decision(face, decision)
        return decision

    def batch_recognize(self, detection_result: DetectionResult, frame: np.ndarray) -> DetectionResult:
        """Recognize every face. Alignment uses the full frame, not only the crop."""
        identities = self.gallery.identity_records()
        for face in detection_result.get_faces_for_recognition():
            self.match_face(face, frame, identities=identities)
        return detection_result


class AttendanceLogger:
    """Kept so older examples still import. The webcam demo does not mark attendance."""

    @staticmethod
    def log_detection_event(frame_idx: int, num_faces: int, avg_confidence: float) -> None:
        detection_logger.info(
            "Frame %s: Detected %s face(s), avg detector score: %.4f",
            frame_idx,
            num_faces,
            avg_confidence,
        )

    @staticmethod
    def log_recognition_event(student_id: str, confidence: float) -> None:
        recognition_logger.info("Recognition event: %s, score: %.4f", student_id, confidence)

    @staticmethod
    def log_attendance_mark(student_id: str, timestamp: str) -> None:
        recognition_logger.info("Attendance mark requested for %s at %s", student_id, timestamp)


def model_path_default(filename: str) -> Path:
    return model_path(filename)


def _apply_decision(face: Face, decision: MatchDecision) -> None:
    face.recognition_status = decision.status
    face.match_score = decision.match_score
    face.match_margin = decision.match_margin
    face.recognition_confidence = None
    if decision.status == "known":
        face.recognized_person_id = decision.person_id
        face.recognized_student_id = decision.person_id
        face.recognized_name = decision.display_name
    else:
        face.recognized_person_id = None
        face.recognized_student_id = None
        face.recognized_name = None


def _resize_for_detector(frame: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink the longest side to ``max_side``. Scale is resized / original."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized, scale


def _unwrap_faces(raw) -> np.ndarray:
    faces = raw[1] if isinstance(raw, tuple) else raw
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    array = np.asarray(faces, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 15), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _clip_bbox(box: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    height, width = frame_shape[:2]
    x = max(0, int(np.floor(box[0])))
    y = max(0, int(np.floor(box[1])))
    right = min(width, int(np.ceil(box[0] + box[2])))
    bottom = min(height, int(np.ceil(box[1] + box[3])))
    box_width = right - x
    box_height = bottom - y
    if box_width <= 0 or box_height <= 0:
        return None
    return x, y, box_width, box_height


def _landmarks(row: np.ndarray) -> tuple[tuple[float, float], ...]:
    points = []
    for index in range(5):
        points.append((float(row[4 + 2 * index]), float(row[5 + 2 * index])))
    return tuple(points)
