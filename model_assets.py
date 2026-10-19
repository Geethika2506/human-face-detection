"""Pinned YuNet and SFace files.

Weights stay outside git. ``scripts/download_models.py`` checks the hashes below.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

DETECTOR_FILENAME = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILENAME = "face_recognition_sface_2021dec.onnx"

DETECTOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
RECOGNIZER_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"

MODELS = {
    DETECTOR_FILENAME: {
        "sha256": DETECTOR_SHA256,
        "urls": [
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
    },
    RECOGNIZER_FILENAME: {
        "sha256": RECOGNIZER_SHA256,
        "urls": [
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ],
    },
}


class ModelAssetError(RuntimeError):
    """A model file is missing or does not match the pinned hash."""


def model_path(filename: str, models_dir: Path | None = None) -> Path:
    return Path(models_dir or MODELS_DIR) / filename


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, expected_sha256: str) -> str:
    """Return the hash when the file matches. Raise ModelAssetError otherwise."""
    path = Path(path)
    if not path.is_file():
        raise ModelAssetError(
            f"Missing model file: {path}. Download it with: python scripts/download_models.py"
        )
    try:
        digest = file_sha256(path)
    except OSError as exc:
        raise ModelAssetError(f"Could not read model file: {path}") from exc
    if digest != expected_sha256:
        raise ModelAssetError(
            f"Model file does not match the pinned hash: {path}. "
            f"Expected {expected_sha256}, got {digest}. "
            "Download a fresh copy with: python scripts/download_models.py"
        )
    return digest


def download_models(models_dir: Path | None = None) -> None:
    """Download any missing model and reject a file whose hash does not match."""
    destination = Path(models_dir or MODELS_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, spec in MODELS.items():
        target = destination / filename
        if target.is_file():
            try:
                verify_model(target, spec["sha256"])
                print(f"Already present: {target}")
                continue
            except ModelAssetError as exc:
                print(exc)
                target.unlink()
        _download_one(target, spec["urls"], spec["sha256"])


def _download_one(target: Path, urls: list[str], expected_sha256: str) -> None:
    errors: list[str] = []
    for url in urls:
        partial = target.with_suffix(target.suffix + ".partial")
        try:
            print(f"Downloading {target.name}")
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "human-face-detection-local-demo"},
            )
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            partial.replace(target)
            verify_model(target, expected_sha256)
            print(f"Saved {target}")
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            partial.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
    raise ModelAssetError(
        f"Could not download {target.name}. " + " | ".join(errors)
    )
