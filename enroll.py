"""Enroll a consenting person into the local gallery.

Webcam enrollment keeps about 10-20 varied frames. A folder of authorized
photos is the other path. Original images are discarded unless --keep-images
is set. The live recognition demo never calls this module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from camera import Camera, CameraError, take_frame, use_camera
from face_detection import FaceDetector, FaceRecognizer
from gallery import Gallery, GalleryError, validate_display_name, validate_person_id
from model_assets import DETECTOR_SHA256, RECOGNIZER_SHA256
from quality import assess_diversity, assess_reference

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CONSENT_TEXT = """
These images are used only to build a face embedding for this local
recognition demo on this computer. They are not uploaded. The original photos
are discarded unless you pass --keep-images. Enroll only yourself or a person
who has agreed to be enrolled.
""".strip()


class EnrollmentError(RuntimeError):
    """The reference image or the enrollment session cannot be saved."""


def confirm_consent(input_fn=input) -> bool:
    print(CONSENT_TEXT)
    answer = input_fn("Type yes to continue: ")
    return answer.strip().lower() == "yes"


def require_single_face(face_count: int) -> None:
    if face_count == 0:
        raise EnrollmentError("No face in the reference image. Exactly one face is required.")
    if face_count != 1:
        raise EnrollmentError(
            f"Found {face_count} faces in the reference image. Exactly one face is required."
        )


def reference_from_frame(frame: np.ndarray, detector: FaceDetector, recognizer: FaceRecognizer, previous):
    """Return (embedding, crop, reason). Reason is empty when the frame is kept."""
    result = detector.detect(frame)
    try:
        require_single_face(len(result.detected_faces))
    except EnrollmentError as exc:
        return None, None, str(exc)
    face = result.detected_faces[0]
    quality = assess_reference(face.crop, face.bbox, face.landmarks, face.confidence)
    if not quality.accepted:
        return None, None, quality.reasons[0]
    try:
        embedding = recognizer.embed(frame, face.yunet_row)
    except (ValueError, cv2.error) as exc:
        return None, None, f"could not build an embedding ({exc})"
    diversity = assess_diversity(embedding, previous)
    if not diversity.accepted:
        return None, None, diversity.reasons[0]
    return embedding, face.crop, ""


def enroll_folder(
    image_dir: str | Path,
    person_id: str,
    display_name: str,
    gallery: Gallery,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    keep_images: bool = False,
    replace: bool = False,
) -> dict:
    """Enroll every acceptable image in a folder. Fail when none pass."""
    person_id = validate_person_id(person_id)
    display_name = validate_display_name(display_name)
    _ensure_replace(gallery, person_id, replace)
    paths = list_images(image_dir)
    if not paths:
        raise EnrollmentError(f"No images found in {image_dir}.")

    accepted = []
    crops = []
    rejected = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            rejected.append((path.name, "could not read image"))
            continue
        embedding, crop, reason = reference_from_frame(image, detector, recognizer, accepted)
        if embedding is None:
            rejected.append((path.name, reason))
            continue
        accepted.append(embedding)
        if keep_images:
            crops.append(crop)

    if not accepted:
        detail = "; ".join(f"{name}: {reason}" for name, reason in rejected)
        raise EnrollmentError(f"No reference image passed the enrollment checks. {detail}")

    gallery.save(
        person_id,
        display_name,
        np.stack(accepted),
        recognizer.opencv_version,
        replace=replace,
        images=crops if keep_images else None,
    )
    return {"accepted": len(accepted), "rejected": rejected}


def run_webcam_enrollment(
    person_id: str,
    display_name: str,
    gallery: Gallery,
    detector: FaceDetector,
    recognizer: FaceRecognizer,
    camera: Camera,
    keep_images: bool = False,
    replace: bool = False,
    target: int = config.ENROLL_TARGET_FRAMES,
    minimum: int = config.ENROLL_MIN_FRAMES,
    maximum: int = config.ENROLL_MAX_FRAMES,
) -> dict:
    """Capture varied webcam frames and store their embeddings."""
    person_id = validate_person_id(person_id)
    display_name = validate_display_name(display_name)
    _ensure_replace(gallery, person_id, replace)
    accepted: list[np.ndarray] = []
    crops: list[np.ndarray] = []
    state = {"cancelled": False}

    def body(open_camera: Camera):
        failures = 0
        reason = "Look at the camera"
        while len(accepted) < maximum:
            frame, failures = take_frame(open_camera, failures, config.MAX_CONSECUTIVE_READ_FAILURES)
            if frame is None:
                continue
            if len(accepted) < maximum:
                embedding, crop, frame_reason = reference_from_frame(
                    frame, detector, recognizer, accepted
                )
                reason = frame_reason or "Good frame captured"
                if embedding is not None:
                    accepted.append(embedding)
                    if keep_images:
                        crops.append(crop)
                    if len(accepted) >= target:
                        reason = "Press space to save, or keep moving to add more"
            _draw_enrollment(frame, len(accepted), target, maximum, reason)
            cv2.imshow("Enroll", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                state["cancelled"] = True
                break
            if key == ord(" ") and len(accepted) >= minimum:
                break
            if len(accepted) >= maximum:
                break
        return len(accepted)

    use_camera(camera, body)
    if state["cancelled"]:
        raise EnrollmentError("Enrollment cancelled. Nothing was saved.")
    if len(accepted) < minimum:
        raise EnrollmentError(
            f"Captured {len(accepted)} frame(s). At least {minimum} are required, so nothing was saved."
        )
    gallery.save(
        person_id,
        display_name,
        np.stack(accepted),
        recognizer.opencv_version,
        replace=replace,
        images=crops if keep_images else None,
    )
    return {"accepted": len(accepted), "rejected": []}


def list_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    if not root.is_dir():
        raise EnrollmentError(f"Image folder does not exist: {root}")
    return sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _ensure_replace(gallery: Gallery, person_id: str, replace: bool) -> None:
    if gallery.person_dir(person_id).exists() and not replace:
        raise GalleryError(
            f"{person_id} is already enrolled. Pass --replace to overwrite that profile."
        )


def _draw_enrollment(frame, count: int, target: int, maximum: int, reason: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Captured {count}/{target} (max {maximum})  q cancel  space save",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        reason[:90],
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _runtime(gallery_dir: str):
    detector = FaceDetector()
    recognizer = FaceRecognizer(
        gallery=Gallery(gallery_dir, DETECTOR_SHA256, RECOGNIZER_SHA256)
    )
    return detector, recognizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enroll a person for the local recognition demo.")
    sub = parser.add_subparsers(dest="command", required=True)

    webcam = sub.add_parser("webcam", help="Capture reference frames from the laptop webcam")
    _add_identity_args(webcam)
    webcam.add_argument("--camera", type=int, default=config.DEFAULT_CAMERA_INDEX)
    webcam.add_argument("--width", type=int, default=config.CAMERA_FRAME_WIDTH)
    webcam.add_argument("--height", type=int, default=config.CAMERA_FRAME_HEIGHT)
    webcam.add_argument("--count", type=int, default=config.ENROLL_TARGET_FRAMES)

    folder = sub.add_parser("folder", help="Enroll authorized images from a folder")
    _add_identity_args(folder)
    folder.add_argument("--images", required=True)

    listing = sub.add_parser("list", help="List enrolled profiles")
    listing.add_argument("--gallery", default=config.GALLERY_DIR)

    delete = sub.add_parser("delete", help="Delete an enrolled profile")
    delete.add_argument("--person-id", required=True)
    delete.add_argument("--gallery", default=config.GALLERY_DIR)
    delete.add_argument("--yes", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            _print_profiles(Gallery(args.gallery, DETECTOR_SHA256, RECOGNIZER_SHA256))
            return 0
        if args.command == "delete":
            return _delete(args)
        if not confirm_consent():
            print("Enrollment cancelled.")
            return 1
        if args.command == "webcam":
            return _webcam(args)
        return _folder(args)
    except (EnrollmentError, GalleryError, CameraError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


def _add_identity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--person-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--gallery", default=config.GALLERY_DIR)


def _print_profiles(gallery: Gallery) -> None:
    profiles = gallery.list_profiles()
    if not profiles:
        print("No enrolled profiles.")
        return
    for profile in profiles:
        match = "current models" if profile["model_match"] else "MODEL MISMATCH"
        print(
            f"{profile['person_id']}\t{profile['display_name']}\t"
            f"{profile['embedding_count']} embeddings\t{match}"
        )


def _delete(args) -> int:
    gallery = Gallery(args.gallery, DETECTOR_SHA256, RECOGNIZER_SHA256)
    if not args.yes:
        answer = input(f"Type {args.person_id} to delete that profile: ").strip()
        if answer != args.person_id:
            print("Deletion cancelled.")
            return 1
    gallery.delete(args.person_id)
    print(f"Deleted {args.person_id}.")
    return 0


def _webcam(args) -> int:
    if not config.ENROLL_MIN_FRAMES <= args.count <= config.ENROLL_MAX_FRAMES:
        raise EnrollmentError(
            f"--count must be between {config.ENROLL_MIN_FRAMES} and {config.ENROLL_MAX_FRAMES}."
        )
    detector, recognizer = _runtime(args.gallery)
    camera = Camera(args.camera, args.width, args.height)
    result = run_webcam_enrollment(
        args.person_id,
        args.name,
        recognizer.gallery,
        detector,
        recognizer,
        camera,
        keep_images=args.keep_images,
        replace=args.replace,
        target=args.count,
    )
    print(f"Enrolled {args.person_id} with {result['accepted']} embeddings.")
    return 0


def _folder(args) -> int:
    detector, recognizer = _runtime(args.gallery)
    result = enroll_folder(
        args.images,
        args.person_id,
        args.name,
        recognizer.gallery,
        detector,
        recognizer,
        keep_images=args.keep_images,
        replace=args.replace,
    )
    print(f"Enrolled {args.person_id} with {result['accepted']} embeddings.")
    if result["accepted"] < config.ENROLL_MIN_FRAMES:
        print(
            f"Only {result['accepted']} image(s) passed. "
            f"{config.ENROLL_MIN_FRAMES} or more varied references are recommended."
        )
    for name, reason in result["rejected"]:
        print(f"Skipped {name}: {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
