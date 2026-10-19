"""Step 6: compare models on the validation set, keep the best one, save it.

The test set is not used here. It is used once, in evaluate.py, for the final scores.
"""
import json
from datetime import date

import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src import baseline, config, preprocess


def make_boosting(n_trees, learning_rate=0.1):
    """Gradient boosted trees: many small decision trees, each one fixing the
    mistakes of the trees before it. Same idea as XGBoost, built into scikit-learn.

    Trees do not need scaled numbers, so numeric columns pass through unchanged.
    """
    columns = ColumnTransformer([
        ("numeric", "passthrough", preprocess.NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         preprocess.CATEGORICAL),
    ])
    model = HistGradientBoostingClassifier(
        max_iter=n_trees,
        learning_rate=learning_rate,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=0,
    )
    return make_pipeline(columns, model)


CANDIDATES = {
    "logistic_regression": baseline.make_model,
    "boosting_50_trees": lambda: make_boosting(50),
    "boosting_100_trees": lambda: make_boosting(100),
    "boosting_200_trees": lambda: make_boosting(200),
}


def compare(train, val):
    X_train, y_train = preprocess.split_xy(train)
    X_val, y_val = preprocess.split_xy(val)

    results = {}
    fitted = {}
    for name, make in CANDIDATES.items():
        model = make().fit(X_train, y_train)
        results[name] = baseline.scores(y_val, model.predict_proba(X_val)[:, 1])
        fitted[name] = model
        baseline.print_scores(name, results[name])
    return results, fitted


if __name__ == "__main__":
    train, val, _ = baseline.load_splits()
    print("Validation scores:\n")
    results, fitted = compare(train, val)

    # PR-AUC is the main metric: it focuses on the rare readmitted patients.
    best = max(results, key=lambda name: results[name]["pr_auc"])
    print(f"\nBest on validation: {best}")

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[best], config.MODEL_PATH)
    metadata = {
        "model": best,
        "trained_on": str(date.today()),
        "scikit_learn_version": sklearn.__version__,   # needed to load the file again
        "train_visits": len(train),
        "validation_scores": {k: round(v, 4) for k, v in results[best].items()},
        "all_validation_scores": {
            name: {k: round(v, 4) for k, v in r.items()} for name, r in results.items()
        },
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Saved to {config.MODEL_PATH.relative_to(config.ROOT)}")
