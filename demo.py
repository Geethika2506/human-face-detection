"""Local webcam face-recognition demo.

Press q to quit. The demo does not save video or enroll new people.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

import config
from camera import Camera, CameraError, take_frame, use_camera
from face_detection import FaceDetector, FaceRecognizer
from gallery import Gallery, GalleryError
from model_assets import DETECTOR_SHA256, ModelAssetError, RECOGNIZER_SHA256
from tracking import Observation, Tracker


class FaceRecognitionDemo:
    def __init__(
        self,
        camera_index: int = config.DEFAULT_CAMERA_INDEX,
        confidence_threshold: float = config.DETECTION_CONFIDENCE_DEFAULT,
        accept_threshold: float = config.ACCEPT_THRESHOLD,
        match_margin: float = config.MATCH_MARGIN,
        width: int = config.CAMERA_FRAME_WIDTH,
        height: int = config.CAMERA_FRAME_HEIGHT,
        process_every: int = config.PROCESS_EVERY_N_FRAMES,
        gallery_dir: str = config.GALLERY_DIR,
        detect_max_side: int = config.DETECT_MAX_SIDE,
    ):
        if process_every < 1:
            raise ValueError("process_every must be at least 1")
        self.process_every = process_every
        self.accept_threshold = accept_threshold
        self.match_margin = match_margin
        self.detector = FaceDetector(
            confidence_threshold=confidence_threshold,
            detect_max_side=detect_max_side,
        )
        self.recognizer = FaceRecognizer(
            gallery=Gallery(gallery_dir, DETECTOR_SHA256, RECOGNIZER_SHA256),
            accept_threshold=accept_threshold,
            match_margin=match_margin,
        )
        self.tracker = Tracker(
            iou_threshold=config.TRACK_IOU_THRESHOLD,
            smooth_window=config.TRACK_SMOOTH_WINDOW,
            max_missed=config.TRACK_MAX_MISSED,
        )
        self.camera = Camera(camera_index, width, height)
        self.fps = 0.0
        self.process_ms = 0.0

    def run(self) -> None:
        print("Face recognition demo")
        print("Press q to quit. Nothing is recorded.")
        print(
            f"Cosine accept threshold {self.accept_threshold:.3f}, "
            f"margin {self.match_margin:.3f}."
        )

        def body(camera: Camera) -> None:
            failures = 0
            frame_index = 0
            fps_count = 0
            fps_started = time.perf_counter()
            tracks = []
            while True:
                frame, failures = take_frame(
                    camera, failures, config.MAX_CONSECUTIVE_READ_FAILURES
                )
                if frame is None:
                    continue
                frame_index += 1
                fps_count += 1
                elapsed = time.perf_counter() - fps_started
                if elapsed >= 1.0:
                    self.fps = fps_count / elapsed
                    fps_count = 0
                    fps_started = time.perf_counter()
                if frame_index % self.process_every == 0:
                    started = time.perf_counter()
                    detection = self.detector.detect(frame)
                    self.recognizer.batch_recognize(detection, frame)
                    observations = [_observation(face) for face in detection.detected_faces]
                    tracks = self.tracker.update(observations)
                    self.process_ms = (time.perf_counter() - started) * 1000.0
                annotated = draw_frame(
                    frame,
                    tracks,
                    self.fps,
                    self.process_ms,
                    self.accept_threshold,
                    self.match_margin,
                )
                cv2.imshow("Face recognition", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit.")
                    break

        use_camera(self.camera, body)
        print(f"Frames processed by the detector: {self.detector.frame_count}")


def draw_frame(frame, tracks, fps: float, process_ms: float, accept_threshold: float, match_margin: float):
    annotated = frame.copy()
    for track in tracks:
        x, y, width, height = track.bbox
        color = _color(track.label)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        text = track.label if not track.score_text else f"{track.label}  {track.score_text}"
        _draw_label(annotated, text, x, y, color)
    lines = [
        f"FPS {fps:.1f}   recognition {process_ms:.0f} ms",
        f"accept {accept_threshold:.3f}   margin {match_margin:.3f}",
        "q quit",
    ]
    panel_height = 18 * len(lines) + 12
    cv2.rectangle(annotated, (8, 8), (360, 8 + panel_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (16, 28 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _observation(face) -> Observation:
    return Observation(
        bbox=face.bbox,
        status=face.recognition_status or "unknown",
        person_id=face.recognized_person_id,
        display_name=face.recognized_name,
        match_score=face.match_score,
    )


def _color(label: str) -> tuple[int, int, int]:
    if label == "Uncertain":
        return (0, 255, 255)
    if label == "Unknown":
        return (0, 165, 255)
    return (0, 200, 0)


def _draw_label(frame, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_width, text_height), _baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_height - 8)
    cv2.rectangle(frame, (x, top), (x + text_width + 8, top + text_height + 8), color, -1)
    cv2.putText(frame, text, (x + 4, top + text_height + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local webcam face-recognition demo")
    parser.add_argument("--camera", type=int, default=config.DEFAULT_CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=config.CAMERA_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=config.CAMERA_FRAME_HEIGHT)
    parser.add_argument("--confidence", type=float, default=config.DETECTION_CONFIDENCE_DEFAULT)
    parser.add_argument("--accept-threshold", type=float, default=config.ACCEPT_THRESHOLD)
    parser.add_argument("--margin", type=float, default=config.MATCH_MARGIN)
    parser.add_argument("--process-every", type=int, default=config.PROCESS_EVERY_N_FRAMES)
    parser.add_argument("--detect-max-side", type=int, default=config.DETECT_MAX_SIDE)
    parser.add_argument("--gallery", default=config.GALLERY_DIR)
    args = parser.parse_args(argv)
    if not 0.0 <= args.confidence <= 1.0:
        print("Error: --confidence must be between 0 and 1.")
        return 1
    if not 0.0 <= args.accept_threshold <= 1.0:
        print("Error: --accept-threshold must be between 0 and 1.")
        return 1
    if args.margin < 0:
        print("Error: --margin must be zero or positive.")
        return 1
    try:
        FaceRecognitionDemo(
            camera_index=args.camera,
            confidence_threshold=args.confidence,
            accept_threshold=args.accept_threshold,
            match_margin=args.margin,
            width=args.width,
            height=args.height,
            process_every=args.process_every,
            gallery_dir=args.gallery,
            detect_max_side=args.detect_max_side,
        ).run()
    except (CameraError, GalleryError, ModelAssetError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Quit.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
