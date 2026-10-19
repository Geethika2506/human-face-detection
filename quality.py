"""Quality gates for enrollment images.

A reference is kept when it contains a large enough, reasonably sharp and
exposed face that is not turned past the configured limits. YuNet landmarks are
ordered right eye, left eye, nose, right mouth corner, left mouth corner.
Roll is the angle of the line between the eyes. The yaw proxy is the nose's
horizontal offset from the eye midpoint, divided by the distance between the
eyes. A value near 0 is frontal.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config
from matching import l2_normalize


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    reasons: tuple[str, ...]


def laplacian_variance(crop: np.ndarray) -> float:
    gray = _gray(crop)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(crop: np.ndarray) -> float:
    return float(_gray(crop).mean())


def roll_degrees(right_eye: tuple[float, float], left_eye: tuple[float, float]) -> float:
    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def yaw_proxy(
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
    nose: tuple[float, float],
) -> float:
    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    interocular = float(np.hypot(dx, dy))
    if interocular < 1.0:
        return float("inf")
    eye_mid_x = (right_eye[0] + left_eye[0]) / 2.0
    return float((nose[0] - eye_mid_x) / interocular)


def assess_reference(
    crop: np.ndarray,
    bbox: tuple[int, int, int, int],
    landmarks: tuple[tuple[float, float], ...],
    detector_score: float,
    min_face_size: int = config.MIN_FACE_SIZE,
    min_detector_score: float = config.MIN_DETECTOR_SCORE,
    min_laplacian: float = config.MIN_LAPLACIAN_VARIANCE,
    min_brightness: float = config.MIN_BRIGHTNESS,
    max_brightness: float = config.MAX_BRIGHTNESS,
    max_roll_degrees: float = config.MAX_ROLL_DEGREES,
    max_yaw: float = config.MAX_YAW_PROXY,
) -> QualityResult:
    """Check one face crop. Diversity against earlier frames is separate."""
    reasons: list[str] = []
    _width = bbox[2]
    height = bbox[3]
    if min(_width, height) < min_face_size:
        reasons.append(f"face is too small ({_width}x{height})")
    if detector_score < min_detector_score:
        reasons.append(f"detector score {detector_score:.3f} is below {min_detector_score:.3f}")
    if crop is None or crop.size == 0:
        reasons.append("face crop is empty")
        return QualityResult(False, tuple(reasons))

    sharpness = laplacian_variance(crop)
    if sharpness < min_laplacian:
        reasons.append(f"face is too blurry (sharpness {sharpness:.1f})")

    brightness = mean_brightness(crop)
    if brightness < min_brightness:
        reasons.append(f"face is too dark (brightness {brightness:.1f})")
    elif brightness > max_brightness:
        reasons.append(f"face is too bright (brightness {brightness:.1f})")

    if landmarks is None or len(landmarks) != 5:
        reasons.append("facial landmarks are missing")
    else:
        right_eye, left_eye, nose, right_mouth, left_mouth = landmarks
        roll = roll_degrees(right_eye, left_eye)
        yaw = yaw_proxy(right_eye, left_eye, nose)
        if abs(roll) > max_roll_degrees:
            reasons.append(f"head is rolled too far ({roll:.1f} degrees)")
        if abs(yaw) > max_yaw:
            reasons.append(f"head is turned too far (yaw proxy {yaw:.2f})")
        eye_mid_y = (right_eye[1] + left_eye[1]) / 2.0
        mouth_mid_y = (right_mouth[1] + left_mouth[1]) / 2.0
        if mouth_mid_y <= eye_mid_y:
            reasons.append("landmarks are inconsistent")

    return QualityResult(accepted=not reasons, reasons=tuple(reasons))


def assess_diversity(
    embedding: np.ndarray,
    previous_embeddings: list[np.ndarray] | None,
    max_reference_cosine: float = config.MAX_REFERENCE_COSINE,
) -> QualityResult:
    """Reject a frame that repeats the last accepted reference."""
    if not previous_embeddings:
        return QualityResult(True, ())
    query = l2_normalize(embedding)
    previous = l2_normalize(previous_embeddings[-1])
    similarity = float(np.dot(query, previous))
    if similarity >= max_reference_cosine:
        return QualityResult(
            False,
            (f"too similar to the previous accepted frame (cosine {similarity:.3f})",),
        )
    return QualityResult(True, ())


def _gray(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 2:
        return crop
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
