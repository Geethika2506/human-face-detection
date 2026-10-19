"""Identity decisions and embedding normalization. No camera and no model files."""

import math
import unittest

import numpy as np

from matching import IdentityRecord, decide_identity, l2_normalize


def basis(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


def angled(cosine: float, axis: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[0] = cosine
    vector[axis] = math.sqrt(1.0 - cosine * cosine)
    return vector


class TestNormalization(unittest.TestCase):
    def test_shape_and_unit_length(self):
        normalized = l2_normalize(np.ones((1, 128), dtype=np.float32) * 5)
        self.assertEqual(normalized.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=5)

    def test_rejects_wrong_shape_and_zero(self):
        with self.assertRaises(ValueError):
            l2_normalize(np.ones(127, dtype=np.float32))
        with self.assertRaises(ValueError):
            l2_normalize(np.zeros(128, dtype=np.float32))


class TestDecisions(unittest.TestCase):
    def test_empty_gallery(self):
        decision = decide_identity(basis(0), [], 0.363, 0.08)
        self.assertEqual(decision.status, "unknown")
        self.assertIsNone(decision.match_score)
        self.assertIsNone(decision.person_id)
        self.assertIsNone(decision.display_name)

    def test_known_unknown_and_uncertain(self):
        query = basis(0)
        identities = [
            IdentityRecord("ada", "Ada", angled(0.80, 2).reshape(1, 128)),
            IdentityRecord("bea", "Bea", angled(0.20, 3).reshape(1, 128)),
        ]
        known = decide_identity(query, identities, 0.363, 0.08)
        self.assertEqual(known.status, "known")
        self.assertEqual(known.person_id, "ada")
        self.assertEqual(known.display_name, "Ada")
        self.assertAlmostEqual(known.match_score, 0.80, places=5)
        self.assertGreater(known.match_margin, 0.08)

        unknown = decide_identity(query, identities, 0.90, 0.08)
        self.assertEqual(unknown.status, "unknown")
        self.assertIsNone(unknown.person_id)
        self.assertIsNone(unknown.display_name)
        self.assertAlmostEqual(unknown.match_score, 0.80, places=5)

        close = [
            IdentityRecord("ada", "Ada", angled(0.80, 2).reshape(1, 128)),
            IdentityRecord("bea", "Bea", angled(0.75, 3).reshape(1, 128)),
        ]
        uncertain = decide_identity(query, close, 0.363, 0.08)
        self.assertEqual(uncertain.status, "uncertain")
        self.assertIsNone(uncertain.person_id)
        self.assertAlmostEqual(uncertain.match_margin, 0.05, places=5)

    def test_margin_accepts_a_wider_gap(self):
        query = basis(0)
        identities = [
            IdentityRecord("ada", "Ada", angled(0.80, 2).reshape(1, 128)),
            IdentityRecord("bea", "Bea", angled(0.75, 3).reshape(1, 128)),
        ]
        decision = decide_identity(query, identities, 0.363, 0.04)
        self.assertEqual(decision.status, "known")
        self.assertEqual(decision.person_id, "ada")

    def test_single_identity_has_no_margin(self):
        query = basis(0)
        identities = [IdentityRecord("ada", "Ada", angled(0.50, 2).reshape(1, 128))]
        accepted = decide_identity(query, identities, 0.363, 0.08)
        self.assertEqual(accepted.status, "known")
        self.assertIsNone(accepted.match_margin)
        rejected = decide_identity(query, identities, 0.70, 0.08)
        self.assertEqual(rejected.status, "unknown")
        self.assertIsNone(rejected.person_id)

    def test_max_cosine_uses_the_nearest_reference(self):
        query = basis(0)
        references = np.stack([angled(0.40, 2), angled(0.85, 3)])
        identities = [
            IdentityRecord("ada", "Ada", references),
            IdentityRecord("bea", "Bea", angled(0.10, 4).reshape(1, 128)),
        ]
        decision = decide_identity(query, identities, 0.363, 0.08)
        self.assertEqual(decision.status, "known")
        self.assertAlmostEqual(decision.match_score, 0.85, places=5)


if __name__ == "__main__":
    unittest.main()
