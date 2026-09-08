import unittest

import numpy as np

from live_recognition import resolve_identity


class DummyANN:
    classes_ = np.array([2, 0, 1])


class TestLabelMapping(unittest.TestCase):
    def test_resolve_identity_uses_ann_class_ids(self):
        probabilities = np.array([0.1, 0.8, 0.1])
        label_names = {0: "Alok", 1: "Nitin", 2: "Ritesh"}

        name, confidence = resolve_identity(probabilities, DummyANN(), label_names, 0.55)

        self.assertEqual(name, "Alok")
        self.assertAlmostEqual(confidence, 0.8)

    def test_low_confidence_is_unknown(self):
        probabilities = np.array([0.55, 0.25, 0.20])
        label_names = {0: "Alok", 1: "Nitin", 2: "Ritesh"}

        name, confidence = resolve_identity(probabilities, DummyANN(), label_names, 0.6)

        self.assertEqual(name, "Unknown")
        self.assertAlmostEqual(confidence, 0.55)


if __name__ == "__main__":
    unittest.main()
