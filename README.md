# Predicting 30-Day Readmission in Diabetic Patients

[![Tests](https://github.com/alidarviish/diabetes-readmission/actions/workflows/tests.yml/badge.svg)](https://github.com/alidarviish/diabetes-readmission/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)

A model that estimates, at discharge, how likely a diabetic patient is to be
readmitted to hospital within 30 days, so that follow-up care can go to the patients
who need it most.

Data: [Diabetes 130-US Hospitals, 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(UCI), about 100,000 hospital visits.

## Results at a glance

On a held-out test set of 15,108 visits from patients the model never saw:

| | Model | Random guess |
|---|---|---|
| ROC-AUC | **0.687** (95% range 0.674-0.700) | 0.500 |
| PR-AUC | **0.257** (95% range 0.239-0.277) | 0.120 |

Used to pick the 20% highest-risk patients for follow-up, the model finds **42%** of
the patients who are readmitted, and **24%** of the patients it flags are readmitted:
**twice** the rate of picking patients at random.

## The data and how it was prepared

Details in [`notebooks/01_explore.ipynb`](notebooks/01_explore.ipynb).

- **One row is one visit, not one patient.** 101,766 visits come from 71,518 patients.
- **The target is rare:** 11% of visits end in a readmission within 30 days.
- **Missing values are written as `?`** and converted to real missing values.
  `weight` (97% missing) and `payer_code` (40%) were dropped; `race` and
  `medical_specialty` keep missing values as an `Unknown` category.
- **`None` is not missing.** In the HbA1c and glucose columns, `None` means the test was
  not done. pandas silently turns this text into missing values by default, so the
  file is read with `keep_default_na=False`.
- **Patients who died or went to hospice were removed** (2,423 visits): they cannot
  be readmitted.
- **12 medication columns with the same value in over 99.9% of rows were dropped.**
- **ICD-9 codes:** about 700 diagnosis codes per column were grouped into 9 body-system
  groups (circulatory, respiratory, diabetes, ...), following Strack et al. (2014),
  the paper that published this dataset.
- **Split by patient, not by row** (70% / 15% / 15%), so that no patient appears in
  both training and test.

## Model

| Model | Validation ROC-AUC | Validation PR-AUC |
|---|---|---|
| Logistic regression (baseline) | 0.666 | 0.221 |
| **Gradient boosting, 50 trees** | **0.680** | **0.239** |
| Gradient boosting, 100 trees | 0.674 | 0.234 |
| Gradient boosting, 200 trees | 0.665 | 0.227 |

Gradient boosting (scikit-learn's `HistGradientBoostingClassifier`, the same method
as XGBoost) was chosen on the validation set. More trees made it worse: the extra
trees start fitting noise in the training data.

I also tried averaging logistic regression, gradient boosting and a random forest.
The difference from gradient boosting alone was within the noise of the validation
set, so the simpler model was kept.

The trained pipeline (preprocessing + model) is saved in `models/model.joblib`, with
its scores and scikit-learn version in `models/metadata.json`.

## 1. How is accuracy measured?

Accuracy is not useful here: a model that always predicts "no readmission" is 89%
accurate and finds no one. Instead:

- **ROC-AUC and PR-AUC** measure how well the model ranks patients by risk, without
  choosing a threshold. PR-AUC focuses on the rare readmitted patients, so it is the
  main metric.
- **Recall and precision at a threshold** show what happens in practice. The threshold
  is set on the validation set so that the 20% highest-risk patients are flagged,
  assuming the hospital can give extra follow-up to about 1 in 5 patients.
  Confusion matrix on test: 755 true positives, 2,375 false alarms, 1,053 missed,
  10,925 true negatives.
- **Calibration:** when the model says "15% risk", about 15% of those patients are
  readmitted (see [`reports/calibration.png`](reports/calibration.png)), so the risk
  can be shown to clinicians as a real probability.
- **95% ranges** from bootstrapping the test set show how much each score could move
  by chance.
- The test set was used **once**, after every choice (model, threshold) was made on
  the validation set.

## 2. How is bias checked?

The same model and threshold were applied to every patient, and performance was
compared across gender, age and race ([`reports/fairness.csv`](reports/fairness.csv)).

- **Gender:** no difference (ROC-AUC 0.68 for women, 0.69 for men; same recall).
- **Age: the model is clearly weaker for older patients.** ROC-AUC is 0.77 for patients
  under 50 and 0.64 for patients 70 and over, and it finds 56% vs 37% of the readmitted
  patients in these groups. The 95% ranges do not overlap, so this is not chance.
  Older patients are also the largest group.
- **Race:** African American and Caucasian patients get very similar results. The
  smaller groups (Asian, Hispanic, Other, Unknown) show lower recall, but each has
  fewer than 50 readmitted patients in the test set, which is too few to draw a
  conclusion. These groups need to be checked again with more data.

What could be done: tell clinicians the model is less reliable for patients over 70,
consider a separate threshold for them together with the clinical team, collect more
data for the small groups, and repeat this check regularly once the model is in use.

## 3. How are predictions explained to clinicians?

**For the model as a whole,** permutation importance shows which information the
model relies on ([`reports/feature_importance.png`](reports/feature_importance.png)).
Two factors matter far more than the rest: the **number of inpatient stays in the
previous year** and **where the patient was discharged to**.

**For one patient,** each value is replaced by a typical one to see how much it
changes the risk. The result is a short card in clinical terms
([`reports/example_explanations.txt`](reports/example_explanations.txt)):

```
Predicted 30-day readmission risk: 65% (5.8x the average of 11%)
Main reasons for higher risk:
  + Inpatient stays in previous year: 9   (adds about 49 percentage points)
  + Second diagnosis group: Neoplasms     (adds about 11 percentage points)
  + Emergency visits in previous year: 3  (adds about 10 percentage points)
```

Two limits are stated with every explanation:

- Factors overlap, so the numbers do not add up to the total risk. SHAP values would
  split the risk between factors exactly; this simpler method was used so each step
  can be explained.
- **The model finds patients at risk; it does not say what causes the risk.** Many
  previous stays are a sign of a more complex condition, not a cause: admitting the
  patient less often would not lower the risk. Which intervention helps is a clinical
  decision.

## Limitations

- Prediction is hard with this data: ROC-AUC around 0.68-0.70 is typical for this
  dataset. Much of what drives readmission (support at home, following the treatment)
  is not recorded.
- The data is from US hospitals in 1999-2008; treatment and coding have changed since.
- The dataset has no visit dates, so a patient's earlier visits cannot be used safely
  as history.

## Project structure

```
src/
  config.py       paths and every cleaning decision in one place
  clean.py        step 2: '?', dropped columns, impossible readmissions, target
  features.py     step 3: ICD-9 codes to body-system groups
  split.py        step 4: train / validation / test by patient
  preprocess.py   step 5: age, codes as categories, scaling, one-hot encoding
  baseline.py     step 5: logistic regression baseline
  train.py        step 6: compare models on validation, save the best
  evaluate.py     step 7: threshold from validation, final scores on test
  fairness.py     step 8: performance by gender, age and race
  explain.py      step 9: feature importance and per-patient explanations
notebooks/01_explore.ipynb   step 1: exploring the data
models/                      saved model and metadata
reports/                     results, charts and explanations
tests/                       38 tests, no data needed
```

## Running it

Download the dataset from UCI and put `diabetic_data.csv` and `IDS_mapping.csv` in
`data/`. Then:

```bash
pip install -r requirements-dev.txt
python -m src.train       # compare models, save the best (about 30 seconds)
python -m src.evaluate    # final scores on the test set
python -m src.fairness    # bias check
python -m src.explain     # feature importance and example explanations
pytest
```

To load the saved model, use the scikit-learn version listed in `models/metadata.json`.
