import pandas as pd

from src import explain


class FakeModel:
    """Risk = 0.1 + 0.1 per previous inpatient stay. Ignores everything else."""

    def predict_proba(self, X):
        risk = (0.1 + 0.1 * X["number_inpatient"]).clip(0, 1).to_numpy()
        return pd.DataFrame({0: 1 - risk, 1: risk}).to_numpy()


def test_typical_values():
    X = pd.DataFrame({"number_inpatient": [0, 0, 1, 5], "gender": ["F", "M", "F", "F"]})
    typical = explain.typical_values(X)
    assert typical["number_inpatient"] == 0.5     # median
    assert typical["gender"] == "F"               # most common


def test_explain_patient_finds_the_reason():
    patient = pd.DataFrame({"number_inpatient": [4], "gender": ["M"]})
    typical = {"number_inpatient": 0, "gender": "F"}

    risk, effect = explain.explain_patient(FakeModel(), patient, typical)

    assert round(risk, 2) == 0.5
    assert effect.index[0] == "number_inpatient"
    assert round(effect["number_inpatient"], 2) == 0.4   # 0.5 -> 0.1 when typical
    assert effect["gender"] == 0                          # the fake model ignores it


def test_readable_code():
    descriptions = {"discharge_disposition_id": {"1": "Discharged to home"}}
    assert explain.readable("discharge_disposition_id", "1", descriptions) == "Discharged to home"
    assert explain.readable("age", 75, descriptions) == 75
