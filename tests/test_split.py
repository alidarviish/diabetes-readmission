import pandas as pd
import pytest

from src import split


@pytest.fixture
def visits():
    """100 patients; patient i has (i % 4) + 1 visits, so some have many."""
    rows = []
    for patient in range(100):
        for _ in range(patient % 4 + 1):
            rows.append({"patient_nbr": patient, "readmitted_30": patient % 9 == 0})
    return pd.DataFrame(rows)


def test_no_patient_in_two_sets(visits):
    train, val, test = split.split_by_patient(visits)
    assert split.shared_patients(train, test) == 0
    assert split.shared_patients(train, val) == 0
    assert split.shared_patients(val, test) == 0


def test_every_visit_is_used_once(visits):
    train, val, test = split.split_by_patient(visits)
    assert len(train) + len(val) + len(test) == len(visits)


def test_sizes_follow_the_requested_share(visits):
    train, val, test = split.split_by_patient(visits, val_size=0.15, test_size=0.15)
    assert test["patient_nbr"].nunique() == 15
    assert val["patient_nbr"].nunique() == 15
    assert train["patient_nbr"].nunique() == 70


def test_same_seed_gives_same_split(visits):
    first = split.split_by_patient(visits, seed=1)[2]
    second = split.split_by_patient(visits, seed=1)[2]
    assert first["patient_nbr"].tolist() == second["patient_nbr"].tolist()


def test_random_row_split_leaks_patients(visits):
    train, test = split.random_row_split(visits, test_size=0.3)
    assert split.shared_patients(train, test) > 0
