"""Offline evaluation on separately captured images.

Layout, with calibration and final kept as different capture sessions::

    eval_data/calibration/enrolled/<person_id>/*.jpg
    eval_data/calibration/unenrolled/<label>/*.jpg
    eval_data/final/enrolled/<person_id>/*.jpg
    eval_data/final/unenrolled/<label>/*.jpg

Rates below use images that contain exactly one detected face. Images with no
face or more than one face are detection failures and stay out of those rates.
A small volunteer set does not establish production reliability.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import config
from face_detection import FaceDetector, FaceRecognizer
from gallery import Gallery, GalleryError
from model_assets import DETECTOR_SHA256, ModelAssetError, RECOGNIZER_SHA256

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
ACCEPT_SWEEP = (0.25, 0.30, 0.35, 0.363, 0.40, 0.45, 0.50, 0.55, 0.60)
MARGIN_SWEEP = (0.02, 0.05, 0.08, 0.12)
DISCLAIMER = (
    "These figures describe this capture only. A small volunteer test does not "
    "establish production reliability. Thresholds can need another calibration "
    "as the gallery grows. This matcher does not check liveness."
)


@dataclass
class Sample:
    path: Path
    group: str
    true_id: str


@dataclass
class EvalRecord:
    path: str
    group: str
    true_id: str
    face_count: int
    detection_failure: bool
    status: str | None
    predicted_id: str | None
    match_score: float | None
    match_margin: float | None
    best_person_id: str | None
    second_score: float | None
    latency_ms: float | None


def iter_samples(split_dir: Path) -> list[Sample]:
    samples = []
    samples.extend(_samples_in(split_dir / "enrolled", "enrolled"))
    samples.extend(_samples_in(split_dir / "unenrolled", "unenrolled"))
    return samples


def overlapping_paths(data_dir: Path) -> list[str]:
    calibration = {sample.path.resolve() for sample in iter_samples(data_dir / "calibration")}
    final = {sample.path.resolve() for sample in iter_samples(data_dir / "final")}
    return sorted(str(path) for path in calibration & final)


def evaluate_samples(samples: list[Sample], detector: FaceDetector, recognizer: FaceRecognizer) -> list[EvalRecord]:
    identities = recognizer.gallery.identity_records()
    records = []
    for sample in samples:
        image = cv2.imread(str(sample.path))
        if image is None:
            records.append(
                EvalRecord(
                    path=str(sample.path),
                    group=sample.group,
                    true_id=sample.true_id,
                    face_count=0,
                    detection_failure=True,
                    status=None,
                    predicted_id=None,
                    match_score=None,
                    match_margin=None,
                    best_person_id=None,
                    second_score=None,
                    latency_ms=None,
                )
            )
            continue
        started = time.perf_counter()
        detection = detector.detect(image)
        face_count = len(detection.detected_faces)
        if face_count != 1:
            records.append(
                EvalRecord(
                    path=str(sample.path),
                    group=sample.group,
                    true_id=sample.true_id,
                    face_count=face_count,
                    detection_failure=True,
                    status=None,
                    predicted_id=None,
                    match_score=None,
                    match_margin=None,
                    best_person_id=None,
                    second_score=None,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            continue
        decision = recognizer.match_face(detection.detected_faces[0], image, identities=identities)
        records.append(
            EvalRecord(
                path=str(sample.path),
                group=sample.group,
                true_id=sample.true_id,
                face_count=1,
                detection_failure=False,
                status=decision.status,
                predicted_id=decision.person_id,
                match_score=decision.match_score,
                match_margin=decision.match_margin,
                best_person_id=decision.best_person_id,
                second_score=decision.second_score,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        )
    return records


def summarize(records: list[EvalRecord], gallery_profiles: list[dict], hardware: dict, thresholds: dict, split: str) -> dict:
    scored = [record for record in records if not record.detection_failure]
    enrolled = [record for record in scored if record.group == "enrolled"]
    unenrolled = [record for record in scored if record.group == "unenrolled"]
    latencies = [record.latency_ms for record in records if record.latency_ms is not None]
    correct = [
        record for record in enrolled if record.status == "known" and record.predicted_id == record.true_id
    ]
    wrong = [
        record for record in enrolled if record.status == "known" and record.predicted_id != record.true_id
    ]
    enrolled_unknown = [record for record in enrolled if record.status == "unknown"]
    enrolled_uncertain = [record for record in enrolled if record.status == "uncertain"]
    false_ids = [record for record in unenrolled if record.status == "known"]
    confusion: dict[str, dict[str, int]] = {}
    for record in enrolled:
        predicted = record.predicted_id if record.status == "known" else record.status
        confusion.setdefault(record.true_id, {})
        confusion[record.true_id][predicted or "unknown"] = confusion[record.true_id].get(predicted or "unknown", 0) + 1

    per_identity: dict[str, int] = {}
    for record in records:
        if record.group == "enrolled":
            per_identity[record.true_id] = per_identity.get(record.true_id, 0) + 1

    return {
        "split": split,
        "disclaimer": DISCLAIMER,
        "thresholds": thresholds,
        "hardware": hardware,
        "gallery_identities": len(gallery_profiles),
        "gallery_embeddings": int(sum(profile.get("embedding_count", 0) for profile in gallery_profiles)),
        "sample_counts": {
            "images": len(records),
            "enrolled_images": sum(1 for record in records if record.group == "enrolled"),
            "unenrolled_images": sum(1 for record in records if record.group == "unenrolled"),
            "enrolled_with_one_face": len(enrolled),
            "unenrolled_with_one_face": len(unenrolled),
            "per_enrolled_identity": per_identity,
        },
        "correct_identification_rate": _rate(len(correct), len(enrolled)),
        "false_identification_rate": _rate(len(false_ids), len(unenrolled)),
        "enrolled_unknown_rate": _rate(len(enrolled_unknown), len(enrolled)),
        "enrolled_uncertain_rate": _rate(len(enrolled_uncertain), len(enrolled)),
        "enrolled_confusion_count": len(wrong),
        "detection_failures": sum(1 for record in records if record.detection_failure),
        "detection_failure_rate": _rate(
            sum(1 for record in records if record.detection_failure),
            len(records),
        ),
        "confusion": confusion,
        "latency_ms": {
            "median": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "records": [asdict(record) for record in records],
    }


def calibration_sweep(records: list[EvalRecord]) -> list[dict]:
    """Re-score stored similarities. This does not change the saved report thresholds."""
    rows = []
    for accept in ACCEPT_SWEEP:
        for margin in MARGIN_SWEEP:
            rescored = [_with_threshold(record, accept, margin) for record in records]
            summary = summarize(rescored, [], {}, {"accept_threshold": accept, "match_margin": margin}, "sweep")
            rows.append(
                {
                    "accept_threshold": accept,
                    "match_margin": margin,
                    "correct_identification_rate": summary["correct_identification_rate"],
                    "false_identification_rate": summary["false_identification_rate"],
                    "enrolled_unknown_rate": summary["enrolled_unknown_rate"],
                    "enrolled_uncertain_rate": summary["enrolled_uncertain_rate"],
                }
            )
    return rows


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 1:
        return float(ordered[0])
    position = (ordered.size - 1) * (percent / 100.0)
    low = int(np.floor(position))
    high = int(np.ceil(position))
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def hardware_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
    }


def _with_threshold(record: EvalRecord, accept: float, margin: float) -> EvalRecord:
    if record.detection_failure or record.match_score is None:
        return record
    gap = None if record.second_score is None else record.match_score - record.second_score
    if record.match_score < accept:
        status, predicted = "unknown", None
    elif gap is not None and gap < margin:
        status, predicted = "uncertain", None
    else:
        status, predicted = "known", record.best_person_id
    return EvalRecord(
        path=record.path,
        group=record.group,
        true_id=record.true_id,
        face_count=record.face_count,
        detection_failure=False,
        status=status,
        predicted_id=predicted,
        match_score=record.match_score,
        match_margin=gap,
        best_person_id=record.best_person_id,
        second_score=record.second_score,
        latency_ms=record.latency_ms,
    )


def _samples_in(root: Path, group: str) -> list[Sample]:
    if not root.is_dir():
        return []
    samples = []
    for person_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image in sorted(person_dir.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(Sample(image, group, person_dir.name))
    return samples


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_report(report: dict) -> str:
    counts = report["sample_counts"]
    lines = [
        f"Split: {report['split']}",
        DISCLAIMER,
        f"Accept threshold: {report['thresholds']['accept_threshold']}",
        f"Margin: {report['thresholds']['match_margin']}",
        f"Hardware: {report['hardware'].get('platform', '')} ({report['hardware'].get('machine', '')})",
        f"Python {report['hardware'].get('python', '')}, OpenCV {report['hardware'].get('opencv', '')}",
        f"Gallery identities: {report['gallery_identities']}",
        f"Gallery embeddings: {report['gallery_embeddings']}",
        f"Images: {counts['images']} (enrolled {counts['enrolled_images']}, unenrolled {counts['unenrolled_images']})",
        f"Single-face enrolled: {counts['enrolled_with_one_face']}",
        f"Single-face unenrolled: {counts['unenrolled_with_one_face']}",
        f"Correct identification rate: {_format_rate(report['correct_identification_rate'])}",
        f"False identification rate: {_format_rate(report['false_identification_rate'])}",
        f"Enrolled unknown rate: {_format_rate(report['enrolled_unknown_rate'])}",
        f"Enrolled uncertain rate: {_format_rate(report['enrolled_uncertain_rate'])}",
        f"Enrolled confusion count: {report['enrolled_confusion_count']}",
        f"Detection failures: {report['detection_failures']} ({_format_rate(report['detection_failure_rate'])})",
        f"Latency median ms: {_format_rate(report['latency_ms']['median'])}",
        f"Latency p95 ms: {_format_rate(report['latency_ms']['p95'])}",
        "Confusion (true id -> predicted):",
    ]
    if report["confusion"]:
        for true_id, predictions in sorted(report["confusion"].items()):
            rendered = ", ".join(f"{label}={count}" for label, count in sorted(predictions.items()))
            lines.append(f"  {true_id}: {rendered}")
    else:
        lines.append("  none")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the local gallery on captured images")
    parser.add_argument("--split", required=True, choices=("calibration", "final"))
    parser.add_argument("--data-dir", default="eval_data")
    parser.add_argument("--gallery", default=config.GALLERY_DIR)
    parser.add_argument("--accept-threshold", type=float, default=config.ACCEPT_THRESHOLD)
    parser.add_argument("--margin", type=float, default=config.MATCH_MARGIN)
    parser.add_argument("--confidence", type=float, default=config.DETECTION_CONFIDENCE_DEFAULT)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    split_dir = data_dir / args.split
    if not split_dir.is_dir():
        print(
            f"Missing {split_dir}. Create enrolled/<person_id>/ and unenrolled/<label>/ "
            "from a capture session that is separate from enrollment."
        )
        return 1
    shared = overlapping_paths(data_dir)
    if shared:
        print(f"Warning: {len(shared)} image path(s) appear in both calibration and final.")

    gallery = Gallery(args.gallery, DETECTOR_SHA256, RECOGNIZER_SHA256)
    try:
        detector = FaceDetector(confidence_threshold=args.confidence)
        recognizer = FaceRecognizer(
            gallery=gallery,
            accept_threshold=args.accept_threshold,
            match_margin=args.margin,
        )
    except (GalleryError, ModelAssetError) as exc:
        print(f"Error: {exc}")
        return 1
    samples = iter_samples(split_dir)
    if not samples:
        print(f"No images found under {split_dir}.")
        return 1
    try:
        records = evaluate_samples(samples, detector, recognizer)
    except GalleryError as exc:
        print(f"Error: {exc}")
        return 1
    report = summarize(
        records,
        gallery.list_profiles(),
        hardware_info(),
        {"accept_threshold": args.accept_threshold, "match_margin": args.margin},
        args.split,
    )
    print(format_report(report))
    if args.split == "calibration":
        print("\nCalibration sweep (not applied to this report):")
        print("accept  margin  correct  false_id  unknown  uncertain")
        for row in calibration_sweep(records):
            print(
                f"{row['accept_threshold']:.3f}   {row['match_margin']:.2f}    "
                f"{_format_rate(row['correct_identification_rate']):>7}  "
                f"{_format_rate(row['false_identification_rate']):>8}  "
                f"{_format_rate(row['enrolled_unknown_rate']):>7}  "
                f"{_format_rate(row['enrolled_uncertain_rate']):>9}"
            )
        print("Copy a chosen pair into config.py and evaluate --split final on a different capture.")

    report_dir = Path(config.REPORTS_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    destination = report_dir / f"eval_{args.split}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
