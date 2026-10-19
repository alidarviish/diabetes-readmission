import pandas as pd

from src import preprocess


def test_age_to_number():
    assert preprocess.age_to_number("[70-80)") == 75
    assert preprocess.age_to_number("[0-10)") == 5


def test_codes_become_text():
    df = pd.DataFrame({"age": ["[50-60)"], "admission_type_id": [1],
                       "discharge_disposition_id": [3], "admission_source_id": [7]})
    result = preprocess.prepare(df)
    assert result.loc[0, "admission_type_id"] == "1"
    assert result.loc[0, "age"] == 55


def test_specialty_grouper_learns_from_training_only():
    train = pd.DataFrame({"medical_specialty":
                          ["Cardiology"] * 5 + ["Surgery"] * 3 + ["Unknown"] * 4})
    other = pd.DataFrame({"medical_specialty": ["Cardiology", "Dentistry", "Unknown"]})

    grouper = preprocess.SpecialtyGrouper().fit(train)
    result = grouper.transform(other)

    # Dentistry never appeared in training, so it becomes 'Other'
    assert result["medical_specialty"].tolist() == ["Cardiology", "Other", "Unknown"]
