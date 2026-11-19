"""Attendance records from confirmed face matches.

Each camera is a *station* with a fixed place, the way a CCTV camera is mounted
in one room. A student's location is that place plus where the face appeared
in the camera's view (a 3x3 zone and the normalized face center). Records are
kept in a local SQLite file so several camera processes can write to the same
session.

Timestamps are local time with the UTC offset, for example
``2026-11-19T09:02:13+01:00``.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import cv2

CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    location TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    late_after_minutes REAL,
    started_at TEXT NOT NULL,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS attendance (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    location TEXT NOT NULL,
    zone TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    last_camera_id TEXT NOT NULL,
    last_location TEXT NOT NULL,
    last_zone TEXT NOT NULL,
    sightings INTEGER NOT NULL DEFAULT 1,
    best_score REAL,
    snapshot TEXT,
    PRIMARY KEY (session_id, person_id)
);
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL,
    seen_at TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    location TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    zone TEXT NOT NULL,
    center_x REAL NOT NULL,
    center_y REAL NOT NULL,
    score REAL
);
CREATE INDEX IF NOT EXISTS sightings_by_person ON sightings(session_id, person_id, seen_at);
"""

REPORT_FIELDS = [
    "person_id",
    "display_name",
    "status",
    "first_seen",
    "last_seen",
    "location",
    "zone",
    "last_location",
    "last_zone",
    "sightings",
    "best_score",
    "snapshot",
]


class AttendanceError(RuntimeError):
    """The attendance database is missing data or the request is invalid."""


@dataclass(frozen=True)
class Station:
    """A camera and the place it watches."""

    camera_id: str
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def __post_init__(self) -> None:
        if not CAMERA_ID_RE.fullmatch(self.camera_id or ""):
            raise AttendanceError(
                "Camera id must be 1-64 characters and use only letters, numbers, '_' and '-'."
            )
        if not self.location.strip():
            raise AttendanceError("Location cannot be empty.")
        if (self.latitude is None) != (self.longitude is None):
            raise AttendanceError("Pass both latitude and longitude, or neither.")
        if self.latitude is not None and not -90.0 <= self.latitude <= 90.0:
            raise AttendanceError("Latitude must be between -90 and 90.")
        if self.longitude is not None and not -180.0 <= self.longitude <= 180.0:
            raise AttendanceError("Longitude must be between -180 and 180.")


@dataclass(frozen=True)
class Position:
    """Where a face sits in the camera view. Coordinates run from 0 to 1."""

    zone: str
    center_x: float
    center_y: float


def frame_position(bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> Position:
    """Map a face box to a 3x3 zone such as ``middle-left`` in camera view."""
    x, y, width, height = bbox
    frame_height, frame_width = frame_shape[:2]
    center_x = min(max((x + width / 2.0) / frame_width, 0.0), 1.0)
    center_y = min(max((y + height / 2.0) / frame_height, 0.0), 1.0)
    column = ("left", "center", "right")[min(2, int(center_x * 3))]
    row = ("top", "middle", "bottom")[min(2, int(center_y * 3))]
    return Position(f"{row}-{column}", round(center_x, 3), round(center_y, 3))


def now_local() -> datetime:
    return datetime.now().astimezone().replace(microsecond=0)


def _iso(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat()


class AttendanceStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def start_session(
        self,
        name: str,
        station: Station,
        started_at: datetime,
        late_after_minutes: Optional[float] = None,
    ) -> int:
        name = " ".join((name or "").split())
        if not name:
            raise AttendanceError("Session name cannot be empty.")
        if late_after_minutes is not None and late_after_minutes < 0:
            raise AttendanceError("Late-after minutes must be zero or positive.")
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO sessions (name, camera_id, location, latitude, longitude, "
                "late_after_minutes, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    name,
                    station.camera_id,
                    station.location,
                    station.latitude,
                    station.longitude,
                    late_after_minutes,
                    _iso(started_at),
                ),
            )
        return int(cursor.lastrowid)

    def end_session(self, session_id: int, ended_at: datetime) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?", (_iso(ended_at), session_id)
            )

    def get_session(self, session_id: int) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise AttendanceError(f"No attendance session with id {session_id}.")
        return row

    def list_sessions(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT s.*, COUNT(a.person_id) AS present FROM sessions s "
            "LEFT JOIN attendance a ON a.session_id = s.id "
            "GROUP BY s.id ORDER BY s.id DESC"
        ).fetchall()

    def delete_session(self, session_id: int) -> None:
        self.get_session(session_id)
        with self.conn:
            self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    def check_in(
        self,
        session_id: int,
        person_id: str,
        display_name: str,
        seen_at: datetime,
        station: Station,
        position: Position,
        score: Optional[float],
        snapshot: Optional[str] = None,
    ) -> Optional[str]:
        """Mark a student present. Return the status, or None if already marked."""
        session = self.get_session(session_id)
        status = attendance_status(session, seen_at)
        with self.conn:
            cursor = self.conn.execute(
                "INSERT OR IGNORE INTO attendance (session_id, person_id, display_name, status, "
                "first_seen, camera_id, location, zone, last_seen, last_camera_id, "
                "last_location, last_zone, sightings, best_score, snapshot) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (
                    session_id,
                    person_id,
                    display_name,
                    status,
                    _iso(seen_at),
                    station.camera_id,
                    station.location,
                    position.zone,
                    _iso(seen_at),
                    station.camera_id,
                    station.location,
                    position.zone,
                    score,
                    snapshot,
                ),
            )
            if cursor.rowcount == 0:
                return None
            self._insert_sighting(session_id, person_id, seen_at, station, position, score)
        return status

    def record_sighting(
        self,
        session_id: int,
        person_id: str,
        seen_at: datetime,
        station: Station,
        position: Position,
        score: Optional[float],
    ) -> None:
        """Update last-seen time and place for a student who is already present."""
        with self.conn:
            self.conn.execute(
                "UPDATE attendance SET last_seen = ?, last_camera_id = ?, last_location = ?, "
                "last_zone = ?, sightings = sightings + 1, "
                "best_score = MAX(COALESCE(best_score, ?), COALESCE(?, best_score)) "
                "WHERE session_id = ? AND person_id = ?",
                (
                    _iso(seen_at),
                    station.camera_id,
                    station.location,
                    position.zone,
                    score,
                    score,
                    session_id,
                    person_id,
                ),
            )
            self._insert_sighting(session_id, person_id, seen_at, station, position, score)

    def _insert_sighting(self, session_id, person_id, seen_at, station, position, score) -> None:
        self.conn.execute(
            "INSERT INTO sightings (session_id, person_id, seen_at, camera_id, location, "
            "latitude, longitude, zone, center_x, center_y, score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                person_id,
                _iso(seen_at),
                station.camera_id,
                station.location,
                station.latitude,
                station.longitude,
                position.zone,
                position.center_x,
                position.center_y,
                score,
            ),
        )

    def attendance(self, session_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM attendance WHERE session_id = ? ORDER BY first_seen, person_id",
            (session_id,),
        ).fetchall()

    def sightings(self, session_id: int, person_id: Optional[str] = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM sightings WHERE session_id = ?"
        params: list = [session_id]
        if person_id:
            query += " AND person_id = ?"
            params.append(person_id)
        return self.conn.execute(query + " ORDER BY seen_at, id", params).fetchall()


def attendance_status(session: sqlite3.Row, seen_at: datetime) -> str:
    late_after = session["late_after_minutes"]
    if late_after is None:
        return "present"
    started = datetime.fromisoformat(session["started_at"])
    return "late" if seen_at > started + timedelta(minutes=late_after) else "present"


@dataclass
class CheckIn:
    display_name: str
    status: str
    first_seen: datetime


class AttendanceMarker:
    """Turn confirmed tracks into check-ins and periodic location updates.

    A track counts once its confirmed person id has held for ``confirm_steps``
    consecutive recognition steps. When two faces in the same frame resolve to
    the same person, neither is marked, since one of them is likely a photo
    or a mismatch.
    """

    def __init__(
        self,
        store: AttendanceStore,
        session_id: int,
        station: Station,
        confirm_steps: int = 5,
        sighting_interval_seconds: float = 10.0,
        snapshot_dir: str | Path | None = None,
        now_fn: Callable[[], datetime] = now_local,
    ):
        if confirm_steps < 1:
            raise AttendanceError("confirm_steps must be at least 1.")
        self.store = store
        self.session_id = session_id
        self.station = station
        self.confirm_steps = confirm_steps
        self.sighting_interval = timedelta(seconds=sighting_interval_seconds)
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.now_fn = now_fn
        self._streaks: dict[int, tuple[str, int]] = {}
        self._last_written: dict[str, datetime] = {}
        self.duplicates: set[str] = set()
        self.checked_in: dict[str, CheckIn] = {
            row["person_id"]: CheckIn(
                row["display_name"], row["status"], datetime.fromisoformat(row["first_seen"])
            )
            for row in store.attendance(session_id)
        }
        self.recent: list[tuple[str, CheckIn]] = []

    def update(self, frame, tracks) -> list[str]:
        """Process one recognition step. Return person ids checked in just now."""
        now = self.now_fn()
        visible = [track for track in tracks if track.missed == 0]
        visible_ids = {track.track_id for track in visible}
        self._streaks = {
            track_id: streak for track_id, streak in self._streaks.items() if track_id in visible_ids
        }

        claims = Counter(track.person_id for track in visible if track.person_id)
        self.duplicates = {person_id for person_id, count in claims.items() if count > 1}

        newly: list[str] = []
        for track in visible:
            if track.person_id is None:
                self._streaks.pop(track.track_id, None)
                continue
            previous_id, count = self._streaks.get(track.track_id, (None, 0))
            count = count + 1 if previous_id == track.person_id else 1
            self._streaks[track.track_id] = (track.person_id, count)
            if count < self.confirm_steps or track.person_id in self.duplicates:
                continue
            if self._mark(frame, track, now):
                newly.append(track.person_id)
        return newly

    def _mark(self, frame, track, now: datetime) -> bool:
        person_id = track.person_id
        position = frame_position(track.bbox, frame.shape)
        score = track.history[-1].match_score if track.history else None
        if person_id not in self.checked_in:
            snapshot = self._save_snapshot(frame, track)
            status = self.store.check_in(
                self.session_id,
                person_id,
                track.label,
                now,
                self.station,
                position,
                score,
                snapshot,
            )
            if status is not None:
                self._last_written[person_id] = now
                check_in = CheckIn(track.label, status, now)
                self.checked_in[person_id] = check_in
                self.recent.insert(0, (person_id, check_in))
                del self.recent[5:]
                return True
            # Another camera on the same session marked this student first.
            self.checked_in[person_id] = CheckIn(track.label, "present", now)
        last = self._last_written.get(person_id)
        if last is None or now - last >= self.sighting_interval:
            self.store.record_sighting(
                self.session_id, person_id, now, self.station, position, score
            )
            self._last_written[person_id] = now
        return False

    def _save_snapshot(self, frame, track) -> Optional[str]:
        if self.snapshot_dir is None:
            return None
        x, y, width, height = track.bbox
        crop = frame[y : y + height, x : x + width]
        if crop.size == 0:
            return None
        folder = self.snapshot_dir / f"session_{self.session_id}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{track.person_id}_{self.station.camera_id}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path)

    def tag_for(self, track) -> Optional[str]:
        if track.person_id is None:
            return None
        if track.person_id in self.duplicates:
            return "duplicate face, not marked"
        check_in = self.checked_in.get(track.person_id)
        if check_in is None:
            return "checking in..."
        return f"{check_in.status} {check_in.first_seen:%H:%M:%S}"


def build_report(store: AttendanceStore, session_id: int, enrolled: list[dict]) -> list[dict]:
    """Present and late students first, then enrolled students who were not seen."""
    store.get_session(session_id)
    rows = []
    seen = set()
    for record in store.attendance(session_id):
        seen.add(record["person_id"])
        rows.append({field: record[field] for field in REPORT_FIELDS})
    for profile in sorted(enrolled, key=lambda item: item["person_id"]):
        if profile["person_id"] in seen:
            continue
        row = {field: "" for field in REPORT_FIELDS}
        row.update(
            person_id=profile["person_id"],
            display_name=profile.get("display_name", ""),
            status="absent",
            sightings=0,
        )
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REPORT_FIELDS})
    return destination
