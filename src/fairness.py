"""Step 8: checking for bias.

The same model and the same threshold are applied to everyone. Here we check
whether it works equally well for different groups of patients:

- ROC-AUC: does the model rank patients equally well inside each group?
- recall: what share of readmitted patients in each group does it catch?
- false positive rate: how often are patients who were NOT readmitted flagged?
- predicted vs observed rate: does the model over- or under-estimate risk for a group?

Small groups give unreliable numbers, so every group gets a 95% range, and groups
with too few readmitted patients are marked.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import baseline, config, evaluate, preprocess

MIN_POSITIVES = 50   # below this, a group's numbers are too uncertain to compare


def age_band(age):
    if age < 50:
        return "under 50"
    if age < 70:
        return "50-69"
    return "70+"


def recall(y, flagged):
    return flagged[y == 1].mean() if (y == 1).any() else np.nan


def false_positive_rate(y, flagged):
    return flagged[y == 0].mean() if (y == 0).any() else np.nan


def bootstrap_range(y, values, metric, n=300, seed=0):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():          # need both classes for AUC
            results.append(metric(y[i], values[i]))
    low, high = np.percentile(results, [2.5, 97.5])
    return f"{low:.2f}-{high:.2f}"


def group_report(df, column):
    rows = []
    for group, part in df.groupby(column):
        y = part["y"].to_numpy()
        p = part["p"].to_numpy()
        flagged = part["flagged"].to_numpy()
        positives = int(y.sum())
        rows.append({
            "attribute": column,
            "group": group,
            "visits": len(part),
            "readmitted": positives,
            "observed_rate": y.mean(),
            "predicted_rate": p.mean(),
            "roc_auc": roc_auc_score(y, p) if 0 < positives < len(y) else np.nan,
            "roc_auc_95": bootstrap_range(y, p, roc_auc_score) if positives >= 10 else "",
            "recall": recall(y, flagged),
            "recall_95": bootstrap_range(y, flagged, recall) if positives >= 10 else "",
            "false_positive_rate": false_positive_rate(y, flagged),
            "reliable": positives >= MIN_POSITIVES,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    model = joblib.load(config.MODEL_PATH)
    _, val, test = baseline.load_splits()
    X_val, _ = preprocess.split_xy(val)
    X_test, y_test = preprocess.split_xy(test)

    threshold = evaluate.choose_threshold(model.predict_proba(X_val)[:, 1])

    df = test[["gender", "race", "age"]].copy()
    df["age_band"] = df["age"].map(age_band)
    df["y"] = y_test.to_numpy()
    df["p"] = model.predict_proba(X_test)[:, 1]
    df["flagged"] = (df["p"] >= threshold).astype(float)

    report = pd.concat([group_report(df, c) for c in ["gender", "age_band", "race"]])

    pd.set_option("display.width", 160)
    shown = report.copy()
    for col in ["observed_rate", "predicted_rate", "roc_auc", "recall", "false_positive_rate"]:
        shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    print(f"Test set, threshold {threshold:.3f}\n")
    print(shown.to_string(index=False))
    print(f"\nreliable = at least {MIN_POSITIVES} readmitted patients in the group")

    config.REPORTS_DIR.mkdir(exist_ok=True)
    report.round(4).to_csv(config.REPORTS_DIR / "fairness.csv", index=False)
