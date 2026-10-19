"""Local enrollment gallery.

Each person lives in ``data/gallery/<person_id>/`` as ``profile.json`` plus
``embeddings.npy``. Original photos are omitted unless enrollment was asked to
keep them. Matching refuses a profile whose detector or recognizer hash differs
from the models loaded in this process.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from matching import EMBEDDING_DIM, IdentityRecord, l2_normalize

PERSON_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
AGGREGATION_METHOD = "max_cosine"


class GalleryError(RuntimeError):
    """The gallery is missing data or the request is invalid."""


class ModelVersionError(GalleryError):
    """Stored embeddings were built with a different model file."""


@dataclass
class StoredProfile:
    person_id: str
    display_name: str
    embeddings: np.ndarray
    detector_sha256: str
    recognizer_sha256: str
    embedding_dim: int
    opencv_version: str
    created_at: str
    updated_at: str
    images_retained: bool


def validate_person_id(person_id: str) -> str:
    if not PERSON_ID_RE.fullmatch(person_id or ""):
        raise GalleryError(
            "Person id must be 1-64 characters and use only letters, numbers, '_' and '-'."
        )
    return person_id


def validate_display_name(display_name: str) -> str:
    name = " ".join((display_name or "").split())
    if not name or len(name) > 80:
        raise GalleryError("Display name must contain 1-80 characters.")
    return name


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != EMBEDDING_DIM or matrix.shape[0] < 1:
        raise GalleryError(
            f"Embeddings must have shape (N, {EMBEDDING_DIM}) with at least one row."
        )
    return np.stack([l2_normalize(row) for row in matrix]).astype(np.float32)


class Gallery:
    def __init__(
        self,
        root: str | Path,
        detector_sha256: str,
        recognizer_sha256: str,
    ):
        self.root = Path(root)
        self.detector_sha256 = detector_sha256
        self.recognizer_sha256 = recognizer_sha256

    def person_dir(self, person_id: str) -> Path:
        return self.root / validate_person_id(person_id)

    def list_profiles(self) -> list[dict]:
        """Return profile metadata. A hash mismatch is reported, not raised."""
        if not self.root.exists():
            return []
        profiles = []
        for path in sorted(self.root.iterdir()):
            profile_path = path / "profile.json"
            if not path.is_dir() or not profile_path.is_file():
                continue
            metadata = json.loads(profile_path.read_text(encoding="utf-8"))
            metadata["model_match"] = (
                metadata.get("detector_sha256") == self.detector_sha256
                and metadata.get("recognizer_sha256") == self.recognizer_sha256
                and metadata.get("embedding_dim") == EMBEDDING_DIM
            )
            profiles.append(metadata)
        return profiles

    def identity_records(self) -> list[IdentityRecord]:
        """Load embeddings for matching. Raise if any profile uses other weights."""
        records = []
        for metadata in self.list_profiles():
            if not metadata["model_match"]:
                raise ModelVersionError(
                    f"Profile {metadata.get('person_id')} was enrolled with a different model. "
                    "Delete that profile and enroll it again with the current models."
                )
            stored = self._read_profile(metadata["person_id"])
            records.append(
                IdentityRecord(
                    person_id=stored.person_id,
                    display_name=stored.display_name,
                    embeddings=stored.embeddings,
                )
            )
        return records

    def save(
        self,
        person_id: str,
        display_name: str,
        embeddings: np.ndarray,
        opencv_version: str,
        replace: bool = False,
        images: Optional[list[np.ndarray]] = None,
    ) -> StoredProfile:
        person_id = validate_person_id(person_id)
        display_name = validate_display_name(display_name)
        matrix = normalize_embeddings(embeddings)
        destination = self.person_dir(person_id)
        created_at = _now()
        if destination.exists():
            if not replace:
                raise GalleryError(
                    f"{person_id} is already enrolled. Pass --replace to overwrite that profile."
                )
            existing = destination / "profile.json"
            if existing.is_file():
                previous = json.loads(existing.read_text(encoding="utf-8"))
                created_at = previous.get("created_at", created_at)
            shutil.rmtree(destination)

        destination.mkdir(parents=True, exist_ok=False)
        images_retained = bool(images)
        if images_retained:
            image_dir = destination / "images"
            image_dir.mkdir()
            import cv2

            for index, image in enumerate(images or []):
                cv2.imwrite(str(image_dir / f"{index:03d}.jpg"), image)

        profile = StoredProfile(
            person_id=person_id,
            display_name=display_name,
            embeddings=matrix,
            detector_sha256=self.detector_sha256,
            recognizer_sha256=self.recognizer_sha256,
            embedding_dim=EMBEDDING_DIM,
            opencv_version=opencv_version,
            created_at=created_at,
            updated_at=_now(),
            images_retained=images_retained,
        )
        self._write_profile(profile)
        return profile

    def delete(self, person_id: str) -> None:
        destination = self.person_dir(person_id)
        if not destination.is_dir():
            raise GalleryError(f"No enrolled profile for {person_id}.")
        shutil.rmtree(destination)

    def _read_profile(self, person_id: str) -> StoredProfile:
        destination = self.person_dir(person_id)
        profile_path = destination / "profile.json"
        embedding_path = destination / "embeddings.npy"
        if not profile_path.is_file() or not embedding_path.is_file():
            raise GalleryError(f"Profile {person_id} is incomplete.")
        metadata = json.loads(profile_path.read_text(encoding="utf-8"))
        embeddings = np.load(embedding_path)
        return StoredProfile(
            person_id=metadata["person_id"],
            display_name=metadata["display_name"],
            embeddings=normalize_embeddings(embeddings),
            detector_sha256=metadata["detector_sha256"],
            recognizer_sha256=metadata["recognizer_sha256"],
            embedding_dim=int(metadata["embedding_dim"]),
            opencv_version=metadata.get("opencv_version", ""),
            created_at=metadata.get("created_at", ""),
            updated_at=metadata.get("updated_at", ""),
            images_retained=bool(metadata.get("images_retained", False)),
        )

    def _write_profile(self, profile: StoredProfile) -> None:
        destination = self.person_dir(profile.person_id)
        metadata = {
            "person_id": profile.person_id,
            "display_name": profile.display_name,
            "embedding_count": int(profile.embeddings.shape[0]),
            "embedding_dim": profile.embedding_dim,
            "detector_model": "face_detection_yunet_2023mar.onnx",
            "detector_sha256": profile.detector_sha256,
            "recognizer_model": "face_recognition_sface_2021dec.onnx",
            "recognizer_sha256": profile.recognizer_sha256,
            "aggregation": AGGREGATION_METHOD,
            "opencv_version": profile.opencv_version,
            "created_at": profile.created_at,
            "updated_at": profile.updated_at,
            "images_retained": profile.images_retained,
        }
        _atomic_text(destination / "profile.json", json.dumps(metadata, indent=2) + "\n")
        _atomic_npy(destination / "embeddings.npy", profile.embeddings)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_npy(path: Path, embeddings: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, embeddings)
    temporary.replace(path)
