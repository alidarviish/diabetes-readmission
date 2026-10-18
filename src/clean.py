"""Step 2: cleaning. Each function does one thing, so each decision is easy to test."""
import numpy as np
import pandas as pd

from src import config


def load_raw(path=config.RAW_DATA):
    """Read the CSV without letting pandas guess which values are missing.

    By default pandas turns text like 'None' into NaN. In this dataset 'None' in
    max_glu_serum and A1Cresult means 'the test was not done', which is real
    information, so we keep every value as written and handle '?' ourselves.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[])


def replace_question_marks(df):
    """The dataset writes missing values as '?'. Turn them into real NaN."""
    return df.replace("?", np.nan)


def drop_columns(df):
    return df.drop(columns=config.DROP_COLUMNS + config.CONSTANT_COLUMNS)


def remove_impossible_readmissions(df):
    """Remove visits where the patient died or went to hospice."""
    keep = ~df["discharge_disposition_id"].isin(config.NO_READMISSION_POSSIBLE)
    return df[keep]


def remove_invalid_gender(df):
    """Three rows have gender 'Unknown/Invalid'."""
    return df[df["gender"].isin(["Female", "Male"])]


def fill_unknown(df):
    df = df.copy()
    df[config.FILL_UNKNOWN] = df[config.FILL_UNKNOWN].fillna("Unknown")
    return df


def make_target(df):
    """1 if readmitted within 30 days, otherwise 0. The original column is removed."""
    df = df.copy()
    df[config.TARGET] = (df["readmitted"] == "<30").astype(int)
    return df.drop(columns="readmitted")


STEPS = [
    replace_question_marks,
    drop_columns,
    remove_impossible_readmissions,
    remove_invalid_gender,
    fill_unknown,
    make_target,
]


def clean(df, verbose=False):
    for step in STEPS:
        before = df.shape
        df = step(df)
        if verbose:
            print(f"{step.__name__:32} {before} -> {df.shape}")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    data = clean(load_raw(), verbose=True)
    print()
    print("readmitted within 30 days:", f"{data[config.TARGET].mean():.1%}")
    print("missing values left:")
    print(data.isna().sum()[lambda s: s > 0])
