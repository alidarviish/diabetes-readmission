import numpy as np
import pandas as pd
import pytest

from src import features


@pytest.mark.parametrize("code, group", [
    ("428", "Circulatory"),      # heart failure
    ("414", "Circulatory"),
    ("785", "Circulatory"),      # single extra code outside the main range
    ("486", "Respiratory"),      # pneumonia
    ("786", "Respiratory"),
    ("250", "Diabetes"),
    ("250.83", "Diabetes"),      # decimals belong to the same group
    ("820", "Injury"),
    ("715", "Musculoskeletal"),
    ("599", "Genitourinary"),
    ("162", "Neoplasms"),
    ("V57", "Other"),            # V and E codes are not body systems
    ("E909", "Other"),
    ("296", "Other"),            # mental disorders: not one of the main groups
    (np.nan, "Missing"),
])
def test_icd9_group(code, group):
    assert features.icd9_group(code) == group


def test_add_diagnosis_groups():
    df = pd.DataFrame({
        "diag_1": ["428", "250.01"],
        "diag_2": ["V58", np.nan],
        "diag_3": ["486", "820"],
        "age": ["[70-80)", "[50-60)"],
    })

    result = features.add_diagnosis_groups(df)

    assert "diag_1" not in result.columns
    assert result["diag_1_group"].tolist() == ["Circulatory", "Diabetes"]
    assert result["diag_2_group"].tolist() == ["Other", "Missing"]
    assert result["diag_3_group"].tolist() == ["Respiratory", "Injury"]
    assert result["age"].tolist() == ["[70-80)", "[50-60)"]   # other columns untouched
