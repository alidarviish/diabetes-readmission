import numpy as np
import pandas as pd
import pytest

from src import clean, config


@pytest.fixture
def raw():
    """A tiny table with the same columns as the real data and one problem per row."""
    rows = [
        # normal visit, readmitted within 30 days
        {"discharge_disposition_id": 1, "gender": "Female", "race": "Caucasian",
         "medical_specialty": "Cardiology", "readmitted": "<30"},
        # missing race and specialty written as '?', readmitted after 30 days
        {"discharge_disposition_id": 1, "gender": "Male", "race": "?",
         "medical_specialty": "?", "readmitted": ">30"},
        # patient died: must be removed
        {"discharge_disposition_id": 11, "gender": "Male", "race": "Caucasian",
         "medical_specialty": "Surgery", "readmitted": "NO"},
        # hospice: must be removed
        {"discharge_disposition_id": 13, "gender": "Female", "race": "Asian",
         "medical_specialty": "?", "readmitted": "NO"},
        # invalid gender: must be removed
        {"discharge_disposition_id": 1, "gender": "Unknown/Invalid", "race": "Other",
         "medical_specialty": "?", "readmitted": "NO"},
        # normal visit, not readmitted
        {"discharge_disposition_id": 6, "gender": "Female", "race": "AfricanAmerican",
         "medical_specialty": "Nephrology", "readmitted": "NO"},
    ]
    df = pd.DataFrame(rows)
    for col in config.DROP_COLUMNS + config.CONSTANT_COLUMNS:
        df[col] = "?" if col == "weight" else "No"
    df["encounter_id"] = range(len(df))
    return df


def test_question_marks_become_nan(raw):
    df = clean.replace_question_marks(raw)
    assert df.loc[1, "race"] is np.nan or pd.isna(df.loc[1, "race"])
    assert (df == "?").sum().sum() == 0


def test_drop_columns(raw):
    df = clean.drop_columns(raw)
    for col in config.DROP_COLUMNS + config.CONSTANT_COLUMNS:
        assert col not in df.columns


def test_dead_and_hospice_patients_removed(raw):
    df = clean.remove_impossible_readmissions(raw)
    assert not df["discharge_disposition_id"].isin([11, 13]).any()
    assert len(df) == 4


def test_target_is_only_within_30_days(raw):
    df = clean.make_target(raw)
    assert "readmitted" not in df.columns
    assert df[config.TARGET].tolist() == [1, 0, 0, 0, 0, 0]


def test_full_clean(raw):
    df = clean.clean(raw)
    assert len(df) == 3
    assert df["race"].tolist() == ["Caucasian", "Unknown", "AfricanAmerican"]
    assert df["medical_specialty"].tolist() == ["Cardiology", "Unknown", "Nephrology"]
    assert df[config.TARGET].tolist() == [1, 0, 0]
    assert df.isna().sum().sum() == 0


def test_load_raw_keeps_none_as_text(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("A1Cresult,race\nNone,?\n>7,Caucasian\n", encoding="utf-8")

    df = clean.replace_question_marks(clean.load_raw(path))

    assert df["A1Cresult"].tolist() == ["None", ">7"]   # a real category, not missing
    assert pd.isna(df.loc[0, "race"])                   # '?' is missing
