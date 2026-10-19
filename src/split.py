"""Step 4: splitting the data into train, validation and test sets by patient.

Some patients have many visits. If one visit of a patient is in the training set
and another in the test set, the model has already seen that patient: the same age,
the same medications, often the same diagnoses, so the test score no longer measures
what we care about: how the model does on a patient it has never seen. So we split
the *patients*, and every visit follows its patient.
"""
import numpy as np

from src import config


def split_by_patient(df, val_size=0.15, test_size=0.15, seed=42):
    """Return (train, val, test). All visits of one patient go to the same set."""
    patients = df["patient_nbr"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n_test = int(len(patients) * test_size)
    n_val = int(len(patients) * val_size)
    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test:n_test + n_val])

    in_test = df["patient_nbr"].isin(test_patients)
    in_val = df["patient_nbr"].isin(val_patients)
    train = df[~in_test & ~in_val]
    val = df[in_val]
    test = df[in_test]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def random_row_split(df, test_size=0.15, seed=42):
    """The usual random split by rows. Only used to show why it is wrong here."""
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(df)) < test_size
    return df[~is_test], df[is_test]


def shared_patients(a, b):
    return len(set(a["patient_nbr"]) & set(b["patient_nbr"]))


if __name__ == "__main__":
    from src import clean, features

    data = features.add_diagnosis_groups(clean.clean(clean.load_raw()))

    train, test = random_row_split(data)
    print("Random split by rows:")
    print(f"  patients in both train and test: {shared_patients(train, test):,}")
    print()

    train, val, test = split_by_patient(data)
    print("Split by patient:")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5}  {len(part):6,} visits  "
              f"{part['patient_nbr'].nunique():6,} patients  "
              f"{part[config.TARGET].mean():.1%} readmitted")
    print(f"  patients shared by train and test: {shared_patients(train, test)}")
    print(f"  patients shared by train and val:  {shared_patients(train, val)}")
