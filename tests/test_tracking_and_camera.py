"""Temporal smoothing and camera cleanup. The webcam is mocked."""

import unittest
from unittest.mock import MagicMock, patch

from camera import CAMERA_HELP, Camera, CameraError, take_frame, use_camera
from tracking import Observation, Tracker


def known(person_id: str, name: str, bbox=(0, 0, 40, 40), score=0.8) -> Observation:
    return Observation(bbox, "known", person_id, name, score)


class TestSmoothing(unittest.TestCase):
    def setUp(self):
        self.tracker = Tracker(iou_threshold=0.3, smooth_window=3, max_missed=1)

    def test_name_waits_for_a_stable_match_and_clears(self):
        first = self.tracker.update([known("ada", "Ada")])
        self.assertEqual(first[0].label, "Unknown")
        self.tracker.update([known("ada", "Ada")])
        named = self.tracker.update([known("ada", "Ada")])
        self.assertEqual(named[0].label, "Ada")

        cleared = self.tracker.update([Observation((0, 0, 40, 40), "unknown", match_score=0.2)])
        self.assertEqual(cleared[0].label, "Unknown")
        self.assertNotIn("Ada", cleared[0].label)

    def test_uncertain_and_identity_change_reset_the_name(self):
        for _ in range(3):
            self.tracker.update([known("ada", "Ada")])
        uncertain = self.tracker.update(
            [Observation((0, 0, 40, 40), "uncertain", match_score=0.7)]
        )
        self.assertEqual(uncertain[0].label, "Uncertain")

        switched = self.tracker.update([known("bea", "Bea", score=0.9)])
        self.assertEqual(switched[0].label, "Unknown")
        self.assertNotEqual(switched[0].label, "Ada")
        self.tracker.update([known("bea", "Bea", score=0.9)])
        confirmed = self.tracker.update([known("bea", "Bea", score=0.9)])
        self.assertEqual(confirmed[0].label, "Bea")

    def test_lost_track_drops_its_label(self):
        self.tracker.update([known("ada", "Ada")])
        self.tracker.update([])
        self.assertEqual(len(self.tracker.update([])), 0)

    def test_low_overlap_does_not_move_a_name_onto_another_face(self):
        stable = Tracker(iou_threshold=0.3, smooth_window=1, max_missed=2)
        for _ in range(1):
            stable.update([known("ada", "Ada", bbox=(0, 0, 30, 30))])
        moved = stable.update([known("bea", "Bea", bbox=(200, 200, 30, 30))])
        labels = {track.label for track in moved}
        bea_track = next(track for track in moved if track.bbox[0] == 200)
        self.assertEqual(bea_track.label, "Bea")
        self.assertNotEqual(bea_track.label, "Ada")
        self.assertIn("Ada", labels)

    def test_reset_clears_tracks(self):
        self.tracker.update([known("ada", "Ada")])
        self.tracker.reset()
        self.assertEqual(self.tracker.tracks, [])


class TestCameraCleanup(unittest.TestCase):
    def test_close_releases_the_device_and_windows(self):
        camera = Camera(0)
        fake = MagicMock()
        fake.isOpened.return_value = True
        with patch("camera.cv2.VideoCapture", return_value=fake), patch(
            "camera.cv2.destroyAllWindows"
        ) as destroy:
            camera.open()
            camera.close()
        fake.release.assert_called_once()
        destroy.assert_called_once()

    def test_open_failure_explains_permission_and_still_releases(self):
        camera = Camera(0)
        fake = MagicMock()
        fake.isOpened.return_value = False
        with patch("camera.cv2.VideoCapture", return_value=fake), patch("camera.cv2.destroyAllWindows"):
            with self.assertRaises(CameraError) as caught:
                camera.open()
        self.assertIn("Privacy", str(caught.exception))
        self.assertIn(CAMERA_HELP, str(caught.exception))
        fake.release.assert_called_once()

    def test_body_error_still_closes_the_camera(self):
        camera = Camera(0)
        fake = MagicMock()
        fake.isOpened.return_value = True

        def boom(_camera):
            raise RuntimeError("preview failed")

        with patch("camera.cv2.VideoCapture", return_value=fake), patch("camera.cv2.destroyAllWindows"):
            with self.assertRaises(RuntimeError):
                use_camera(camera, boom)
        fake.release.assert_called_once()
        self.assertIsNone(camera.cap)

    def test_repeated_read_failures_stop(self):
        camera = MagicMock()
        camera.read_latest.return_value = None
        failures = 0
        _frame, failures = take_frame(camera, failures, 3)
        self.assertEqual(failures, 1)
        with self.assertRaises(CameraError):
            take_frame(camera, 2, 3)


if __name__ == "__main__":
    unittest.main()
