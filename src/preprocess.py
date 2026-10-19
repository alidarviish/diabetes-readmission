"""Step 5a: turning the cleaned table into numbers a model can use."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

NUMERIC = [
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]

# These look like numbers but are codes: admission type 3 is not "three times"
# admission type 1. So they are treated as categories, not as numeric values.
CODES = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "acarbose", "insulin",
    "glyburide-metformin",
]

CATEGORICAL = (
    ["race", "gender", "medical_specialty", "max_glu_serum", "A1Cresult",
     "change", "diabetesMed", "diag_1_group", "diag_2_group", "diag_3_group"]
    + CODES + MEDICATIONS
)

# medical_specialty has about 70 values; most are rare. Keep the most common ones.
TOP_SPECIALTIES = 10


def age_to_number(age):
    """'[70-80)' -> 75, the middle of the range."""
    low, high = age.strip("[)").split("-")
    return (int(low) + int(high)) / 2


class SpecialtyGrouper:
    """Keeps the most common specialties seen in training, everything else becomes 'Other'.

    It learns the list from the training set only (fit), then applies the same list
    to validation and test (transform), so no information leaks from them.
    """

    def fit(self, df):
        counts = df["medical_specialty"].value_counts()
        self.keep = set(counts.drop("Unknown", errors="ignore").index[:TOP_SPECIALTIES])
        self.keep.add("Unknown")
        return self

    def transform(self, df):
        df = df.copy()
        df["medical_specialty"] = df["medical_specialty"].where(
            df["medical_specialty"].isin(self.keep), "Other")
        return df


def prepare(df):
    """Steps that need nothing learned from data."""
    df = df.copy()
    df["age"] = df["age"].map(age_to_number)
    df[CODES] = df[CODES].astype(str)
    return df


def make_column_transformer():
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def split_xy(df):
    return df[NUMERIC + CATEGORICAL], df[config.TARGET]
