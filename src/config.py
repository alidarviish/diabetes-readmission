from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = ROOT / "data" / "diabetic_data.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = ROOT / "reports"

TARGET = "readmitted_30"

# Mostly empty (weight 97%, payer_code 40%) or just a row number.
DROP_COLUMNS = ["encounter_id", "weight", "payer_code"]

# Medication columns where more than 99.9% of rows have the same value.
# Found in notebooks/01_explore.ipynb, section 7.
CONSTANT_COLUMNS = [
    "examide", "citoglipton", "acetohexamide", "glimepiride-pioglitazone",
    "metformin-pioglitazone", "metformin-rosiglitazone", "troglitazone",
    "glipizide-metformin", "tolbutamide", "miglitol", "tolazamide", "chlorpropamide",
]

# discharge_disposition_id values meaning the patient died (11) or went to
# hospice care (13, 14, 19, 20, 21). These patients cannot be readmitted.
NO_READMISSION_POSSIBLE = [11, 13, 14, 19, 20, 21]

# Missing values in these columns become their own category.
FILL_UNKNOWN = ["race", "medical_specialty"]
