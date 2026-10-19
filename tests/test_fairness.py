import numpy as np
import pandas as pd

from src import fairness


def test_age_band():
    assert fairness.age_band(45) == "under 50"
    assert fairness.age_band(55) == "50-69"
    assert fairness.age_band(75) == "70+"


def test_recall_and_false_positive_rate():
    y = np.array([1, 1, 0, 0])
    flagged = np.array([1.0, 0.0, 1.0, 0.0])
    assert fairness.recall(y, flagged) == 0.5
    assert fairness.false_positive_rate(y, flagged) == 0.5


def test_group_report_marks_small_groups():
    rng = np.random.default_rng(0)
    big = pd.DataFrame({"g": "big", "y": rng.random(2000) < 0.1})
    small = pd.DataFrame({"g": "small", "y": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]})
    df = pd.concat([big, small], ignore_index=True)
    df["y"] = df["y"].astype(int)
    df["p"] = rng.random(len(df))
    df["flagged"] = (df["p"] > 0.8).astype(float)

    report = fairness.group_report(df, "g").set_index("group")

    assert report.loc["big", "reliable"]
    assert not report.loc["small", "reliable"]
    assert report.loc["small", "visits"] == 10
