"""Step 5b: the baseline model, logistic regression.

Every later model has to beat this one to be worth its extra complexity.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from src import clean, features, preprocess, split


def load_splits():
    data = features.add_diagnosis_groups(clean.clean(clean.load_raw()))
    data = preprocess.prepare(data)
    train, val, test = split.split_by_patient(data)

    grouper = preprocess.SpecialtyGrouper().fit(train)
    return grouper.transform(train), grouper.transform(val), grouper.transform(test)


def make_model():
    return make_pipeline(
        preprocess.make_column_transformer(),
        LogisticRegression(max_iter=2000),
    )


def scores(y_true, probabilities):
    return {
        "accuracy": accuracy_score(y_true, probabilities >= 0.5),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def print_scores(name, result):
    print(f"{name:28} accuracy {result['accuracy']:.3f}   "
          f"ROC-AUC {result['roc_auc']:.3f}   PR-AUC {result['pr_auc']:.3f}")


if __name__ == "__main__":
    train, val, _ = load_splits()      # the test set stays untouched until the end
    X_train, y_train = preprocess.split_xy(train)
    X_val, y_val = preprocess.split_xy(val)

    print(f"validation set: {len(y_val):,} visits, {y_val.mean():.1%} readmitted\n")

    always_no = y_val * 0.0
    print_scores("Always predict 'no'", scores(y_val, always_no))

    model = make_model().fit(X_train, y_train)
    print_scores("Logistic regression", scores(y_val, model.predict_proba(X_val)[:, 1]))
