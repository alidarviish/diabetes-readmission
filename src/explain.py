"""Step 9: making predictions understandable for clinicians.

Two levels:

1. Whole model: which pieces of patient information matter most overall?
   Permutation importance: shuffle one column so it no longer matches its patient,
   and measure how much worse the model gets. A big drop means the model relies on it.

2. One patient: why did the model give *this* patient a high risk?
   For each piece of information, replace this patient's value with a typical value
   (the most common one in training) and see how the predicted risk changes.
   If the risk drops a lot when "4 previous inpatient visits" becomes "0", that
   history is a main reason for the high risk.

Both work on the original columns (age, number_inpatient, ...) rather than on the
model's internal one-hot columns, so the explanations use words clinicians know.
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import baseline, config, evaluate, preprocess

# Clinician-friendly names for the columns
LABELS = {
    "number_inpatient": "Inpatient stays in previous year",
    "number_emergency": "Emergency visits in previous year",
    "number_outpatient": "Outpatient visits in previous year",
    "discharge_disposition_id": "Where the patient was discharged to",
    "admission_source_id": "How the patient was admitted",
    "admission_type_id": "Type of admission",
    "time_in_hospital": "Days in hospital",
    "num_medications": "Number of medications",
    "num_lab_procedures": "Number of lab tests",
    "num_procedures": "Number of procedures",
    "number_diagnoses": "Number of diagnoses",
    "diag_1_group": "Main diagnosis group",
    "diag_2_group": "Second diagnosis group",
    "diag_3_group": "Third diagnosis group",
    "medical_specialty": "Admitting specialty",
    "age": "Age",
    "A1Cresult": "HbA1c test result",
    "max_glu_serum": "Glucose test result",
    "diabetesMed": "On diabetes medication",
    "change": "Diabetes medication changed",
    "insulin": "Insulin",
}


def label(column):
    return LABELS.get(column, column)


def read_code_descriptions(path=config.ROOT / "data" / "IDS_mapping.csv"):
    """IDS_mapping.csv holds three small tables (admission type, discharge,
    admission source) one after another. Return {column: {code: description}}."""
    descriptions = {}
    with open(path, encoding="utf-8") as file:
        tables = file.read().strip().split("\n,\n")
    for table in tables:
        lines = table.strip().splitlines()
        column = lines[0].split(",")[0]
        descriptions[column] = {}
        for line in lines[1:]:
            code, text = line.split(",", 1)
            if code.strip():
                descriptions[column][code.strip()] = text.strip('"')
    return descriptions


def readable(column, value, descriptions):
    """Turn a code like discharge_disposition_id '6' into its meaning."""
    return descriptions.get(column, {}).get(str(value), value)


def global_importance(model, X, y, n_repeats=5):
    result = permutation_importance(model, X, y, scoring="average_precision",
                                    n_repeats=n_repeats, random_state=0, n_jobs=-1)
    return (pd.Series(result.importances_mean, index=X.columns)
            .sort_values(ascending=False))


def typical_values(X_train):
    """Median for numbers, most common value for categories."""
    typical = {}
    for column in X_train.columns:
        if column in preprocess.NUMERIC:
            typical[column] = X_train[column].median()
        else:
            typical[column] = X_train[column].mode()[0]
    return typical


def explain_patient(model, patient, typical):
    """How much does the risk change if each value is replaced by a typical one?

    Positive = this value raises the patient's risk.
    """
    base_risk = model.predict_proba(patient)[0, 1]
    variants = []
    for column in patient.columns:
        changed = patient.copy()
        changed[column] = typical[column]
        variants.append(changed)
    risks = model.predict_proba(pd.concat(variants))[:, 1]
    effect = pd.Series(base_risk - risks, index=patient.columns)
    return base_risk, effect.sort_values(ascending=False)


def patient_card(patient, risk, effect, base_rate, descriptions, top=4):
    lines = [f"Predicted 30-day readmission risk: {risk:.0%} "
             f"({risk / base_rate:.1f}x the average of {base_rate:.0%})",
             "Main reasons for higher risk (risk change if only this one value were typical;"
             " factors overlap, so the numbers do not add up):"]
    for column, change in effect.head(top).items():
        if change <= 0.005:
            break
        value = readable(column, patient[column].iloc[0], descriptions)
        lines.append(f"  + {label(column)}: {value}  "
                     f"(adds about {change * 100:.0f} percentage points)")
    lowering = effect[effect < -0.005].sort_values().head(2)
    if len(lowering):
        lines.append("Factors lowering risk:")
        for column, change in lowering.items():
            value = readable(column, patient[column].iloc[0], descriptions)
            lines.append(f"  - {label(column)}: {value}  "
                         f"(lowers by about {-change * 100:.0f} percentage points)")
    return "\n".join(lines)


def plot_importance(importance, path, top=12):
    top_items = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([label(c) for c in top_items.index], top_items.values, color="#2c7fb8")
    ax.set_xlabel("Drop in PR-AUC when the column is shuffled")
    ax.set_title("What the model relies on most")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    model = joblib.load(config.MODEL_PATH)
    train, val, test = baseline.load_splits()
    X_train, _ = preprocess.split_xy(train)
    X_val, y_val = preprocess.split_xy(val)
    X_test, y_test = preprocess.split_xy(test)

    # Global importance is measured on validation, not test
    importance = global_importance(model, X_val, y_val)
    print("Most important information (drop in PR-AUC when shuffled):\n")
    for column, value in importance.head(10).items():
        print(f"  {label(column):42} {value:.4f}")
    config.REPORTS_DIR.mkdir(exist_ok=True)
    plot_importance(importance, config.REPORTS_DIR / "feature_importance.png")

    # Explain three high-risk test patients
    typical = typical_values(X_train)
    base_rate = float(train[config.TARGET].mean())
    descriptions = read_code_descriptions()
    risks = model.predict_proba(X_test)[:, 1]
    readmitted = y_test.to_numpy() == 1
    threshold = evaluate.choose_threshold(model.predict_proba(X_val)[:, 1])

    # Three kinds of example: a correct high-risk call, a high-risk patient who was
    # not readmitted (a false alarm), and a patient just above the threshold.
    examples = [
        np.flatnonzero(readmitted)[np.argmax(risks[readmitted])],
        np.flatnonzero(~readmitted)[np.argmax(risks[~readmitted])],
        np.argmin(np.where(risks >= threshold, risks, np.inf)),
    ]

    cards = []
    for i in examples:
        patient = X_test.iloc[[i]]
        risk, effect = explain_patient(model, patient, typical)
        outcome = "readmitted" if y_test.iloc[i] else "not readmitted"
        card = patient_card(patient, risk, effect, base_rate, descriptions)
        cards.append(f"{card}\n(Actual outcome: {outcome})")
    text = "\n\n".join(cards)
    print("\n\nExample explanations for high-risk test patients:\n")
    print(text)
    (config.REPORTS_DIR / "example_explanations.txt").write_text(text + "\n", encoding="utf-8")
