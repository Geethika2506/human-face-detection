"""Laptop webcam helper.

The capture buffer is set to one frame and each loop grabs once, so recognition
always uses the newest frame instead of a growing queue. ``close`` releases the
device and closes HighGUI windows, including after an error.
"""

from __future__ import annotations

import cv2

CAMERA_HELP = (
    "Could not open the camera. On macOS, allow Camera access for this Terminal "
    "or IDE in System Settings > Privacy & Security > Camera, then run the command "
    "again. Another camera can be selected with --camera 1."
)


class CameraError(RuntimeError):
    """The webcam could not be opened or a frame could not be read."""


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index = index
        self.width = width
        self.height = height
        self.cap = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.index)
        if self.cap is None or not self.cap.isOpened():
            self.close()
            raise CameraError(CAMERA_HELP)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read_latest(self):
        """Grab and retrieve one frame. Return None when the read fails."""
        if self.cap is None:
            raise CameraError("Camera is not open.")
        if not self.cap.grab():
            return None
        ok, frame = self.cap.retrieve()
        if not ok or frame is None or frame.size == 0:
            return None
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()


def use_camera(camera: Camera, body):
    """Open the camera, run ``body``, and always release it."""
    camera.open()
    try:
        return body(camera)
    finally:
        camera.close()


def take_frame(camera: Camera, failures: int, max_failures: int):
    """Return (frame or None, updated failure count). Raise after repeated misses."""
    frame = camera.read_latest()
    if frame is None:
        failures += 1
        if failures >= max_failures:
            raise CameraError(
                "Stopped after repeated failed frame reads. Check that the camera "
                "is connected and not in use by another application."
            )
        return None, failures
    return frame, 0
