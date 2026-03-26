import numpy as np

from engine.monitoring.drift_detection import DriftDetector


def test_no_drift():

    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0, 1, 1000)

    detector = DriftDetector()

    result = detector.ks_test(ref, cur)

    assert result["drift"] is False


def test_drift_detected():

    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(3, 1, 1000)

    detector = DriftDetector()

    result = detector.ks_test(ref, cur)

    assert result["drift"] is True