"""Metric math and the separation between calibration and final captures."""

import tempfile
import unittest
from pathlib import Path

from evaluate import EvalRecord, calibration_sweep, iter_samples, overlapping_paths, percentile, summarize


def record(group, true_id, status, predicted, score=0.8, second=0.2, failure=False, latency=10.0):
    return EvalRecord(
        path=f"{group}/{true_id}.jpg",
        group=group,
        true_id=true_id,
        face_count=0 if failure else 1,
        detection_failure=failure,
        status=None if failure else status,
        predicted_id=None if failure else predicted,
        match_score=None if failure else score,
        match_margin=None if failure or second is None else score - second,
        best_person_id=None if failure else true_id if group == "enrolled" else "ada",
        second_score=None if failure else second,
        latency_ms=None if latency is None else latency,
    )


class TestMetrics(unittest.TestCase):
    def test_rates_confusion_and_latency(self):
        records = [
            record("enrolled", "ada", "known", "ada", latency=10),
            record("enrolled", "ada", "known", "bea", score=0.7, second=0.6, latency=30),
            record("enrolled", "bea", "unknown", None, score=0.2, second=0.1, latency=20),
            record("enrolled", "bea", "uncertain", None, score=0.6, second=0.55, latency=40),
            record("unenrolled", "guest", "known", "ada", latency=15),
            record("unenrolled", "guest", "unknown", None, score=0.1, second=0.05, latency=25),
            record("enrolled", "ada", "known", "ada", failure=True, latency=5),
        ]
        # The confused enrolled row stored best_person_id as true_id via the helper.
        # Override it so a wrong known id stays wrong under the reported status.
        records[1] = EvalRecord(
            path="enrolled/ada-wrong.jpg",
            group="enrolled",
            true_id="ada",
            face_count=1,
            detection_failure=False,
            status="known",
            predicted_id="bea",
            match_score=0.7,
            match_margin=0.1,
            best_person_id="bea",
            second_score=0.6,
            latency_ms=30,
        )
        report = summarize(
            records,
            [{"embedding_count": 4}, {"embedding_count": 2}],
            {"platform": "test", "machine": "arm64", "python": "3.14", "opencv": "4.14.0"},
            {"accept_threshold": 0.363, "match_margin": 0.08},
            "final",
        )
        self.assertEqual(report["gallery_identities"], 2)
        self.assertEqual(report["gallery_embeddings"], 6)
        self.assertEqual(report["sample_counts"]["images"], 7)
        self.assertEqual(report["detection_failures"], 1)
        self.assertAlmostEqual(report["correct_identification_rate"], 0.25)
        self.assertAlmostEqual(report["false_identification_rate"], 0.5)
        self.assertAlmostEqual(report["enrolled_unknown_rate"], 0.25)
        self.assertAlmostEqual(report["enrolled_uncertain_rate"], 0.25)
        self.assertEqual(report["enrolled_confusion_count"], 1)
        self.assertEqual(report["confusion"]["ada"]["bea"], 1)
        self.assertEqual(report["latency_ms"]["median"], 20.0)
        self.assertAlmostEqual(report["latency_ms"]["p95"], 37.0)

    def test_percentile_and_empty(self):
        self.assertIsNone(percentile([], 95))
        self.assertEqual(percentile([4.0], 95), 4.0)

    def test_sweep_can_change_a_decision_without_rewriting_inputs(self):
        records = [
            record("enrolled", "ada", "unknown", None, score=0.40, second=0.10),
        ]
        loose = calibration_sweep(records)
        matched = next(
            row for row in loose if row["accept_threshold"] == 0.30 and row["match_margin"] == 0.08
        )
        strict = next(
            row for row in loose if row["accept_threshold"] == 0.55 and row["match_margin"] == 0.08
        )
        self.assertEqual(matched["correct_identification_rate"], 1.0)
        self.assertEqual(strict["correct_identification_rate"], 0.0)
        self.assertEqual(records[0].status, "unknown")

    def test_splits_stay_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, person in (("calibration", "ada"), ("final", "bea")):
                folder = root / split / "enrolled" / person
                folder.mkdir(parents=True)
                (folder / "frame.jpg").write_bytes(b"jpeg")
                other = root / split / "unenrolled" / "guest"
                other.mkdir(parents=True)
                (other / "frame.jpg").write_bytes(b"jpeg")
            calibration = iter_samples(root / "calibration")
            final = iter_samples(root / "final")
            self.assertEqual(sorted(sample.true_id for sample in calibration), ["ada", "guest"])
            self.assertEqual(sorted(sample.true_id for sample in final), ["bea", "guest"])
            self.assertEqual(overlapping_paths(root), [])


if __name__ == "__main__":
    unittest.main()
