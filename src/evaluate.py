"""Step 7: the final evaluation. This is the only file that uses the test set.

1. Choose the decision threshold on the validation set.
2. Score the saved model once on the test set.
3. Report ranking quality, results at the threshold, calibration, and how much
   each number could move by chance (bootstrap).
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)

from src import baseline, config, preprocess

# The hospital can give extra follow-up to about 1 in 5 patients.
# The threshold is set so that the 20% highest-risk patients are flagged.
SHARE_FLAGGED = 0.20


def choose_threshold(val_probabilities, share=SHARE_FLAGGED):
    return float(np.quantile(val_probabilities, 1 - share))


def bootstrap_interval(y_true, probabilities, metric, n=500, seed=0):
    """Resample the test set many times to see how much a score moves by chance."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    values = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        values.append(metric(y_true[i], probabilities[i]))
    return np.percentile(values, [2.5, 97.5])


def results_at_threshold(y_true, probabilities, threshold):
    flagged = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, flagged).ravel()
    return {
        "flagged_share": float(flagged.mean()),
        "recall": recall_score(y_true, flagged),
        "precision": precision_score(y_true, flagged),
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn), "true_negatives": int(tn),
    }


def plot_calibration(y_true, probabilities, path):
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10,
                                            strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 0.5], [0, 0.5], "--", color="grey", label="perfect calibration")
    ax.plot(predicted, observed, "o-", color="#c0392b", label="model")
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed readmission rate")
    ax.set_title("Calibration on the test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    model = joblib.load(config.MODEL_PATH)
    _, val, test = baseline.load_splits()
    X_val, _ = preprocess.split_xy(val)
    X_test, y_test = preprocess.split_xy(test)

    threshold = choose_threshold(model.predict_proba(X_val)[:, 1])
    p_test = model.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, p_test)
    pr = average_precision_score(y_test, p_test)
    roc_ci = bootstrap_interval(y_test, p_test, roc_auc_score)
    pr_ci = bootstrap_interval(y_test, p_test, average_precision_score)
    at = results_at_threshold(y_test, p_test, threshold)
    brier = brier_score_loss(y_test, p_test)
    base_rate = float(y_test.mean())

    print(f"Test set: {len(y_test):,} visits, {base_rate:.1%} readmitted within 30 days\n")
    print("Ranking quality (threshold-free)")
    print(f"  ROC-AUC  {roc:.3f}   95% range {roc_ci[0]:.3f}-{roc_ci[1]:.3f}   (random = 0.5)")
    print(f"  PR-AUC   {pr:.3f}   95% range {pr_ci[0]:.3f}-{pr_ci[1]:.3f}   (random = {base_rate:.3f})\n")
    print(f"At threshold {threshold:.3f} (chosen on validation to flag ~{SHARE_FLAGGED:.0%})")
    print(f"  flagged    {at['flagged_share']:.1%} of patients")
    print(f"  recall     {at['recall']:.1%} of readmitted patients are flagged")
    print(f"  precision  {at['precision']:.1%} of flagged patients are readmitted "
          f"({at['precision'] / base_rate:.1f}x the base rate)")
    print(f"  confusion  TP {at['true_positives']}  FP {at['false_positives']}  "
          f"FN {at['false_negatives']}  TN {at['true_negatives']}\n")
    print(f"Calibration: Brier score {brier:.4f} "
          f"(always predicting the base rate gives {brier_score_loss(y_test, np.full_like(p_test, base_rate)):.4f})")

    config.REPORTS_DIR.mkdir(exist_ok=True)
    plot_calibration(y_test, p_test, config.REPORTS_DIR / "calibration.png")
    metrics = {
        "test_visits": len(y_test), "base_rate": round(base_rate, 4),
        "roc_auc": round(roc, 4), "roc_auc_95": [round(x, 4) for x in roc_ci],
        "pr_auc": round(pr, 4), "pr_auc_95": [round(x, 4) for x in pr_ci],
        "threshold": round(threshold, 4),
        "at_threshold": {k: round(v, 4) if isinstance(v, float) else v for k, v in at.items()},
        "brier": round(brier, 4),
    }
    (config.REPORTS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
