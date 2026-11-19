"""Camera-based attendance.

``run`` opens a camera as an attendance station, checks in enrolled students
whose face is recognized, and records when and where they were seen.
``sessions``, ``report`` and ``delete-session`` read and manage the records.

Press q in the camera window to end the session.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from attendance import (
    AttendanceError,
    AttendanceMarker,
    AttendanceStore,
    Station,
    build_report,
    now_local,
    write_csv,
)
from camera import CameraError, parse_source
from demo import FaceRecognitionDemo
from gallery import Gallery, GalleryError
from model_assets import DETECTOR_SHA256, ModelAssetError, RECOGNIZER_SHA256


class AttendanceSession(FaceRecognitionDemo):
    window_title = "Attendance"

    def __init__(
        self,
        store: AttendanceStore,
        session_id: int,
        session_name: str,
        station: Station,
        confirm_steps: int = config.ATTENDANCE_CONFIRM_STEPS,
        sighting_interval_seconds: float = config.ATTENDANCE_SIGHTING_INTERVAL_SECONDS,
        snapshot_dir: str | None = None,
        **demo_kwargs,
    ):
        super().__init__(**demo_kwargs)
        self.session_id = session_id
        self.session_name = session_name
        self.station = station
        self.enrolled = len(self.recognizer.gallery.list_profiles())
        self.marker = AttendanceMarker(
            store,
            session_id,
            station,
            confirm_steps=confirm_steps,
            sighting_interval_seconds=sighting_interval_seconds,
            snapshot_dir=snapshot_dir,
        )
        self._tracks = []

    def announce(self) -> None:
        print(f"Attendance session {self.session_id}: {self.session_name}")
        print(f"Camera {self.station.camera_id} at {self.station.location}")
        print("Press q in the camera window to end the session.")

    def on_tracks(self, frame, tracks) -> None:
        self._tracks = tracks
        for person_id in self.marker.update(frame, tracks):
            check_in = self.marker.checked_in[person_id]
            print(
                f"{check_in.first_seen:%H:%M:%S}  {check_in.status.upper():8} "
                f"{check_in.display_name} ({person_id}) at {self.station.location}"
            )

    def track_tags(self) -> dict[int, str]:
        tags = {}
        for track in self._tracks:
            tag = self.marker.tag_for(track)
            if tag:
                tags[track.track_id] = tag
        return tags

    def panel_lines(self) -> list[str]:
        lines = [
            f"{self.session_name}  (session {self.session_id})",
            f"{self.station.location}  [{self.station.camera_id}]  {now_local():%Y-%m-%d %H:%M:%S}",
            f"Checked in {len(self.marker.checked_in)} of {self.enrolled} enrolled",
        ]
        if self.marker.recent:
            _person_id, latest = self.marker.recent[0]
            lines.append(
                f"Last: {latest.display_name} {latest.status} {latest.first_seen:%H:%M:%S}"
            )
        return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Face-recognition attendance from a camera.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Take attendance from a camera")
    run.add_argument("--session", help="Session name, for example 'CS101 Lecture'")
    run.add_argument("--join", type=int, help="Add this camera to an existing session id")
    run.add_argument("--location", default=config.DEFAULT_LOCATION, help="Where the camera is")
    run.add_argument("--camera-id", default=config.DEFAULT_CAMERA_ID, help="Name of this camera")
    run.add_argument("--lat", type=float, help="Latitude of the camera location")
    run.add_argument("--lon", type=float, help="Longitude of the camera location")
    run.add_argument(
        "--late-after",
        type=float,
        help="Minutes after the session starts when a check-in counts as late",
    )
    run.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Save the face crop at check-in as evidence",
    )
    run.add_argument(
        "--camera",
        type=parse_source,
        default=config.DEFAULT_CAMERA_INDEX,
        help="Webcam index, or an RTSP/HTTP stream URL or video file",
    )
    run.add_argument("--width", type=int, default=config.CAMERA_FRAME_WIDTH)
    run.add_argument("--height", type=int, default=config.CAMERA_FRAME_HEIGHT)
    run.add_argument("--confidence", type=float, default=config.DETECTION_CONFIDENCE_DEFAULT)
    run.add_argument("--accept-threshold", type=float, default=config.ACCEPT_THRESHOLD)
    run.add_argument("--margin", type=float, default=config.MATCH_MARGIN)
    run.add_argument("--process-every", type=int, default=config.PROCESS_EVERY_N_FRAMES)
    run.add_argument("--confirm-steps", type=int, default=config.ATTENDANCE_CONFIRM_STEPS)
    run.add_argument("--gallery", default=config.GALLERY_DIR)
    run.add_argument("--db", default=config.ATTENDANCE_DB)

    sessions = sub.add_parser("sessions", help="List attendance sessions")
    sessions.add_argument("--db", default=config.ATTENDANCE_DB)

    report = sub.add_parser("report", help="Show who was present, late, or absent")
    report.add_argument("--session-id", type=int, required=True)
    report.add_argument("--csv", help="Also write the report to this CSV file")
    report.add_argument(
        "--trail", action="store_true", help="Show each student's timestamped locations"
    )
    report.add_argument("--gallery", default=config.GALLERY_DIR)
    report.add_argument("--db", default=config.ATTENDANCE_DB)

    delete = sub.add_parser("delete-session", help="Delete a session and its records")
    delete.add_argument("--session-id", type=int, required=True)
    delete.add_argument("--yes", action="store_true")
    delete.add_argument("--db", default=config.ATTENDANCE_DB)

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "sessions":
            return _sessions(args)
        if args.command == "report":
            return _report(args)
        return _delete(args)
    except (AttendanceError, CameraError, GalleryError, ModelAssetError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Stopped.")
        return 0


def _run(args) -> int:
    if bool(args.session) == bool(args.join):
        raise AttendanceError("Pass either --session NAME to start a session or --join ID.")
    if not 0.0 <= args.accept_threshold <= 1.0:
        raise AttendanceError("--accept-threshold must be between 0 and 1.")
    station = Station(args.camera_id, " ".join(args.location.split()), args.lat, args.lon)
    store = AttendanceStore(args.db)
    try:
        if args.join:
            session = store.get_session(args.join)
            if session["ended_at"]:
                raise AttendanceError(f"Session {args.join} has already ended.")
            session_id, session_name, started_here = args.join, session["name"], False
        else:
            session_id = store.start_session(args.session, station, now_local(), args.late_after)
            session_name, started_here = store.get_session(session_id)["name"], True
        try:
            AttendanceSession(
                store,
                session_id,
                session_name,
                station,
                confirm_steps=args.confirm_steps,
                snapshot_dir=config.ATTENDANCE_SNAPSHOT_DIR if args.save_snapshots else None,
                camera_index=args.camera,
                confidence_threshold=args.confidence,
                accept_threshold=args.accept_threshold,
                match_margin=args.margin,
                width=args.width,
                height=args.height,
                process_every=args.process_every,
                gallery_dir=args.gallery,
            ).run()
        finally:
            if started_here:
                store.end_session(session_id, now_local())
        print()
        _print_report(store, session_id, _enrolled(args.gallery))
    finally:
        store.close()
    return 0


def _sessions(args) -> int:
    store = AttendanceStore(args.db)
    try:
        rows = store.list_sessions()
        if not rows:
            print("No attendance sessions.")
            return 0
        print("ID\tPRESENT\tSTARTED\t\t\t\tENDED\t\t\t\tLOCATION\tNAME")
        for row in rows:
            print(
                f"{row['id']}\t{row['present']}\t{row['started_at']}\t"
                f"{row['ended_at'] or 'running':25}\t{row['location']}\t{row['name']}"
            )
    finally:
        store.close()
    return 0


def _report(args) -> int:
    store = AttendanceStore(args.db)
    try:
        rows = _print_report(store, args.session_id, _enrolled(args.gallery))
        if args.trail:
            _print_trail(store, args.session_id)
        if args.csv:
            print(f"Wrote {write_csv(rows, args.csv)}")
    finally:
        store.close()
    return 0


def _delete(args) -> int:
    store = AttendanceStore(args.db)
    try:
        store.get_session(args.session_id)
        if not args.yes:
            answer = input(f"Type {args.session_id} to delete that session: ").strip()
            if answer != str(args.session_id):
                print("Deletion cancelled.")
                return 1
        store.delete_session(args.session_id)
        snapshots = Path(config.ATTENDANCE_SNAPSHOT_DIR) / f"session_{args.session_id}"
        if snapshots.is_dir():
            for path in snapshots.iterdir():
                path.unlink()
            snapshots.rmdir()
        print(f"Deleted session {args.session_id}.")
    finally:
        store.close()
    return 0


def _enrolled(gallery_dir: str) -> list[dict]:
    return Gallery(gallery_dir, DETECTOR_SHA256, RECOGNIZER_SHA256).list_profiles()


def _print_report(store: AttendanceStore, session_id: int, enrolled: list[dict]) -> list[dict]:
    session = store.get_session(session_id)
    rows = build_report(store, session_id, enrolled)
    print(f"Session {session_id}: {session['name']}")
    print(f"Location {session['location']}  started {session['started_at']}", end="")
    print(f"  ended {session['ended_at']}" if session["ended_at"] else "  (running)")
    counts = {status: sum(row["status"] == status for row in rows) for status in ("present", "late", "absent")}
    print(f"Present {counts['present']}  Late {counts['late']}  Absent {counts['absent']}")
    if not rows:
        print("No students enrolled or seen.")
        return rows
    print()
    print(f"{'STATUS':8} {'ID':16} {'NAME':24} {'CHECK-IN':26} {'LAST SEEN':26} WHERE")
    for row in rows:
        where = ""
        if row["status"] != "absent":
            where = f"{row['last_location']} ({row['last_zone']})"
        print(
            f"{row['status']:8} {row['person_id']:16} {row['display_name'][:24]:24} "
            f"{row['first_seen'] or '-':26} {row['last_seen'] or '-':26} {where}"
        )
    return rows


def _print_trail(store: AttendanceStore, session_id: int) -> None:
    sightings = store.sightings(session_id)
    print()
    if not sightings:
        print("No sightings recorded.")
        return
    print(f"{'SEEN AT':26} {'ID':16} {'CAMERA':14} {'LOCATION':24} {'ZONE':14} SCORE")
    for row in sightings:
        score = "" if row["score"] is None else f"{row['score']:.3f}"
        print(
            f"{row['seen_at']:26} {row['person_id']:16} {row['camera_id']:14} "
            f"{row['location'][:24]:24} {row['zone']:14} {score}"
        )


if __name__ == "__main__":
    sys.exit(main())
