import numpy as np

from src import evaluate


def test_threshold_flags_the_requested_share():
    probabilities = np.linspace(0, 1, 1000)
    threshold = evaluate.choose_threshold(probabilities, share=0.2)
    assert abs((probabilities >= threshold).mean() - 0.2) < 0.01


def test_results_at_threshold():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    r = evaluate.results_at_threshold(y, p, threshold=0.5)
    assert r["true_positives"] == 1 and r["false_positives"] == 1
    assert r["false_negatives"] == 1 and r["true_negatives"] == 2
    assert r["recall"] == 0.5 and r["precision"] == 0.5
