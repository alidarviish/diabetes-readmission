"""Step 3: turning ICD-9 diagnosis codes into a few medical groups.

There are around 700 different codes in each diagnosis column. Most appear only a
handful of times, which is too little for a model to learn from. Grouping them by
body system, the way ICD-9 itself is organised, gives 9 groups with enough data each.
The groups follow Strack et al. (2014), the paper that published this dataset.
"""
import pandas as pd

DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]

# (group name, list of (first code, last code) ranges, inclusive)
ICD9_RANGES = [
    ("Circulatory",     [(390, 459), (785, 785)]),
    ("Respiratory",     [(460, 519), (786, 786)]),
    ("Digestive",       [(520, 579), (787, 787)]),
    ("Injury",          [(800, 999)]),
    ("Musculoskeletal", [(710, 739)]),
    ("Genitourinary",   [(580, 629), (788, 788)]),
    ("Neoplasms",       [(140, 239)]),
]


def icd9_group(code):
    """'428' -> 'Circulatory', '250.83' -> 'Diabetes', 'V57' -> 'Other', NaN -> 'Missing'."""
    if pd.isna(code):
        return "Missing"

    code = str(code)
    # V codes (reasons for a visit, like rehabilitation) and E codes (external
    # causes, like a fall) are not diseases of a body system.
    if code.startswith(("V", "E")):
        return "Other"

    number = float(code)
    if int(number) == 250:          # 250.xx: diabetes and its complications
        return "Diabetes"
    for group, ranges in ICD9_RANGES:
        for first, last in ranges:
            if first <= int(number) <= last:
                return group
    return "Other"


def add_diagnosis_groups(df):
    """Replace diag_1, diag_2, diag_3 with diag_1_group, diag_2_group, diag_3_group."""
    df = df.copy()
    for column in DIAGNOSIS_COLUMNS:
        df[f"{column}_group"] = df[column].map(icd9_group)
    return df.drop(columns=DIAGNOSIS_COLUMNS)


if __name__ == "__main__":
    from src import clean, config

    data = add_diagnosis_groups(clean.clean(clean.load_raw()))
    summary = data.groupby("diag_1_group")[config.TARGET].agg(["size", "mean"])
    summary.columns = ["visits", "readmitted_30"]
    summary["readmitted_30"] = (summary["readmitted_30"] * 100).round(1)
    print("Main diagnosis (diag_1) groups:\n")
    print(summary.sort_values("visits", ascending=False))
