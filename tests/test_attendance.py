"""Attendance storage, check-in rules, reports, and the report CLI. No camera."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

import attend
from attendance import (
    AttendanceError,
    AttendanceMarker,
    AttendanceStore,
    Position,
    Station,
    build_report,
    frame_position,
    write_csv,
)
from camera import parse_source
from gallery import Gallery
from model_assets import DETECTOR_SHA256, RECOGNIZER_SHA256
from tracking import Observation, Tracker

START = datetime(2026, 11, 19, 9, 0, 0, tzinfo=timezone(timedelta(hours=1)))
ROOM = Station("front-cam", "Room 101", 52.37, 4.89)
CENTER = Position("middle-center", 0.5, 0.5)
FRAME = np.zeros((300, 300, 3), dtype=np.uint8)


def known(person_id, name, bbox=(100, 100, 60, 60), score=0.7):
    return Observation(bbox, "known", person_id, name, score)


class Clock:
    def __init__(self, start=START):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class TestPositionAndStation(unittest.TestCase):
    def test_zones_follow_the_camera_view(self):
        shape = (300, 300, 3)
        self.assertEqual(frame_position((0, 0, 30, 30), shape).zone, "top-left")
        self.assertEqual(frame_position((120, 120, 60, 60), shape).zone, "middle-center")
        self.assertEqual(frame_position((270, 270, 30, 30), shape).zone, "bottom-right")
        position = frame_position((120, 120, 60, 60), shape)
        self.assertEqual((position.center_x, position.center_y), (0.5, 0.5))

    def test_station_validation(self):
        with self.assertRaises(AttendanceError):
            Station("bad id!", "Room 101")
        with self.assertRaises(AttendanceError):
            Station("cam", "  ")
        with self.assertRaises(AttendanceError):
            Station("cam", "Room", latitude=10.0)
        with self.assertRaises(AttendanceError):
            Station("cam", "Room", latitude=95.0, longitude=0.0)

    def test_parse_source_accepts_index_or_stream(self):
        self.assertEqual(parse_source("0"), 0)
        self.assertEqual(parse_source("rtsp://cam/stream"), "rtsp://cam/stream")


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = AttendanceStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_check_in_once_then_sightings_update_last_seen(self):
        session = self.store.start_session("CS101", ROOM, START)
        status = self.store.check_in(session, "ada", "Ada", START, ROOM, CENTER, 0.6)
        self.assertEqual(status, "present")
        self.assertIsNone(self.store.check_in(session, "ada", "Ada", START, ROOM, CENTER, 0.6))

        hall = Station("hall-cam", "Hallway")
        later = START + timedelta(minutes=5)
        self.store.record_sighting(session, "ada", later, hall, Position("top-left", 0.1, 0.1), 0.8)
        row = self.store.attendance(session)[0]
        self.assertEqual(row["first_seen"], START.isoformat())
        self.assertEqual(row["location"], "Room 101")
        self.assertEqual(row["last_seen"], later.isoformat())
        self.assertEqual(row["last_location"], "Hallway")
        self.assertEqual(row["last_zone"], "top-left")
        self.assertEqual(row["sightings"], 2)
        self.assertAlmostEqual(row["best_score"], 0.8)

        trail = self.store.sightings(session, "ada")
        self.assertEqual([item["location"] for item in trail], ["Room 101", "Hallway"])
        self.assertAlmostEqual(trail[0]["latitude"], 52.37)

    def test_late_after_threshold(self):
        session = self.store.start_session("CS101", ROOM, START, late_after_minutes=10)
        on_time = self.store.check_in(
            session, "ada", "Ada", START + timedelta(minutes=10), ROOM, CENTER, 0.6
        )
        late = self.store.check_in(
            session, "bea", "Bea", START + timedelta(minutes=11), ROOM, CENTER, 0.6
        )
        self.assertEqual((on_time, late), ("present", "late"))

    def test_delete_session_removes_records(self):
        session = self.store.start_session("CS101", ROOM, START)
        self.store.check_in(session, "ada", "Ada", START, ROOM, CENTER, 0.6)
        self.store.delete_session(session)
        self.assertEqual(self.store.attendance(session), [])
        self.assertEqual(self.store.sightings(session), [])
        with self.assertRaises(AttendanceError):
            self.store.get_session(session)

    def test_report_lists_absent_students_and_writes_csv(self):
        session = self.store.start_session("CS101", ROOM, START)
        self.store.check_in(session, "ada", "Ada", START, ROOM, CENTER, 0.6)
        enrolled = [
            {"person_id": "ada", "display_name": "Ada"},
            {"person_id": "bea", "display_name": "Bea"},
        ]
        rows = build_report(self.store, session, enrolled)
        self.assertEqual([(row["person_id"], row["status"]) for row in rows], [("ada", "present"), ("bea", "absent")])
        with tempfile.TemporaryDirectory() as temporary:
            path = write_csv(rows, Path(temporary) / "out" / "report.csv")
            with path.open(encoding="utf-8") as handle:
                saved = list(csv.DictReader(handle))
        self.assertEqual(saved[1]["status"], "absent")
        self.assertEqual(saved[0]["location"], "Room 101")


class TestMarker(unittest.TestCase):
    def setUp(self):
        self.store = AttendanceStore(":memory:")
        self.session = self.store.start_session("CS101", ROOM, START)
        self.clock = Clock()
        self.tracker = Tracker(iou_threshold=0.3, smooth_window=1, max_missed=2)

    def tearDown(self):
        self.store.close()

    def marker(self, **kwargs):
        options = dict(confirm_steps=3, sighting_interval_seconds=10, now_fn=self.clock)
        options.update(kwargs)
        return AttendanceMarker(self.store, self.session, ROOM, **options)

    def step(self, marker, observations):
        tracks = self.tracker.update(observations)
        self.clock.advance(1)
        return marker.update(FRAME, tracks), tracks

    def test_checks_in_only_after_a_stable_match(self):
        marker = self.marker()
        for _ in range(2):
            newly, tracks = self.step(marker, [known("ada", "Ada")])
            self.assertEqual(newly, [])
            self.assertEqual(marker.tag_for(tracks[0]), "checking in...")
        newly, tracks = self.step(marker, [known("ada", "Ada")])
        self.assertEqual(newly, ["ada"])
        self.assertTrue(marker.tag_for(tracks[0]).startswith("present 09:00:03"))
        row = self.store.attendance(self.session)[0]
        self.assertEqual(row["zone"], "middle-center")
        self.assertEqual(row["location"], "Room 101")

    def test_unknown_result_resets_the_streak(self):
        marker = self.marker()
        self.step(marker, [known("ada", "Ada")])
        self.step(marker, [known("ada", "Ada")])
        self.step(marker, [Observation((100, 100, 60, 60), "unknown", match_score=0.1)])
        newly, _ = self.step(marker, [known("ada", "Ada")])
        self.assertEqual(newly, [])
        self.assertEqual(self.store.attendance(self.session), [])

    def test_same_person_on_two_faces_is_not_marked(self):
        marker = self.marker(confirm_steps=1)
        pair = [known("ada", "Ada", bbox=(0, 0, 50, 50)), known("ada", "Ada", bbox=(200, 200, 50, 50))]
        newly, tracks = self.step(marker, pair)
        self.assertEqual(newly, [])
        self.assertEqual(marker.tag_for(tracks[0]), "duplicate face, not marked")
        self.assertEqual(self.store.attendance(self.session), [])

    def test_sightings_are_throttled(self):
        marker = self.marker(confirm_steps=1)
        for _ in range(25):
            self.step(marker, [known("ada", "Ada")])
        # Check-in at t=1, then one sighting every 10 seconds up to t=25.
        self.assertEqual(len(self.store.sightings(self.session, "ada")), 3)

    def test_existing_check_ins_are_loaded_when_joining(self):
        self.store.check_in(self.session, "ada", "Ada", START, Station("other", "Room 102"), CENTER, 0.5)
        marker = self.marker(confirm_steps=1)
        newly, _ = self.step(marker, [known("ada", "Ada")])
        self.assertEqual(newly, [])
        self.assertIn("ada", marker.checked_in)
        self.assertEqual(self.store.attendance(self.session)[0]["last_location"], "Room 101")

    def test_snapshot_saved_at_check_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = self.marker(confirm_steps=1, snapshot_dir=temporary)
            self.step(marker, [known("ada", "Ada")])
            snapshot = self.store.attendance(self.session)[0]["snapshot"]
            self.assertTrue(Path(snapshot).is_file())
            self.assertIn("session_", snapshot)


class TestTrackPersonId(unittest.TestCase):
    def test_person_id_set_only_when_confirmed(self):
        tracker = Tracker(smooth_window=2)
        self.assertIsNone(tracker.update([known("ada", "Ada")])[0].person_id)
        self.assertEqual(tracker.update([known("ada", "Ada")])[0].person_id, "ada")
        cleared = tracker.update([Observation((100, 100, 60, 60), "uncertain", match_score=0.5)])
        self.assertIsNone(cleared[0].person_id)


class TestGalleryCache(unittest.TestCase):
    def test_records_reload_only_when_files_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            gallery = Gallery(temporary, DETECTOR_SHA256, RECOGNIZER_SHA256)
            embeddings = np.eye(2, 128, dtype=np.float32)
            gallery.save("ada", "Ada", embeddings, opencv_version="4.14.0")
            with patch.object(gallery, "_load_records", wraps=gallery._load_records) as load:
                gallery.identity_records()
                gallery.identity_records()
                self.assertEqual(load.call_count, 1)
                other = Gallery(temporary, DETECTOR_SHA256, RECOGNIZER_SHA256)
                other.save("bea", "Bea", embeddings, opencv_version="4.14.0")
                records = gallery.identity_records()
                self.assertEqual(load.call_count, 2)
            self.assertEqual({record.person_id for record in records}, {"ada", "bea"})


class TestReportCli(unittest.TestCase):
    def test_sessions_and_report_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            db = str(Path(temporary) / "attendance.db")
            gallery_dir = str(Path(temporary) / "gallery")
            Gallery(gallery_dir, DETECTOR_SHA256, RECOGNIZER_SHA256).save(
                "bea", "Bea", np.eye(1, 128, dtype=np.float32), opencv_version="4.14.0"
            )
            store = AttendanceStore(db)
            session = store.start_session("CS101", ROOM, START)
            store.check_in(session, "ada", "Ada", START, ROOM, CENTER, 0.6)
            store.close()

            output = io.StringIO()
            csv_path = Path(temporary) / "report.csv"
            with redirect_stdout(output):
                self.assertEqual(attend.main(["sessions", "--db", db]), 0)
                code = attend.main(
                    [
                        "report",
                        "--session-id",
                        str(session),
                        "--db",
                        db,
                        "--gallery",
                        gallery_dir,
                        "--trail",
                        "--csv",
                        str(csv_path),
                    ]
                )
            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("CS101", text)
            self.assertIn("Present 1  Late 0  Absent 1", text)
            self.assertIn("Room 101 (middle-center)", text)
            self.assertTrue(csv_path.is_file())

    def test_run_requires_session_or_join(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(attend.main(["run", "--db", ":memory:"]), 1)
        self.assertIn("--session", output.getvalue())


if __name__ == "__main__":
    unittest.main()
