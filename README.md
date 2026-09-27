# VitalAlert

An undergraduate graduation project using MIMIC-IV 2.2 to predict
future vital signs from past observations.

The current implementation focuses on predicting the mean heart rate
over the next hour using six hours of historical heart-rate measurements.

## Current progress

- Inspected and screened 13 measurement items covering heart rate,
  oxygen saturation, respiratory rate, and blood pressure.
- Built hourly heart-rate inputs and next-hour prediction targets.
- Separated patients into training, validation, and test groups.
- Compared three baselines: the last hourly value, the six-hour mean,
  and the mean of training targets.
- Checked processing rules with synthetic tests and independently
  recalculated 75 prediction windows.

LSTM training and clinical alert evaluation have not yet been completed.
Prediction inputs for the other vital signs have not yet been constructed.

## Preliminary results

The three baselines were evaluated on the same 610,450 test windows.

| Baseline | MAE (bpm) |
| --- | ---: |
| Last hourly value | 6.446 |
| Six-hour mean | 6.283 |
| Mean of training targets | 14.033 |

These are initial results. Clinical usefulness and statistical significance
have not been established.

## Main files

- `scripts/pipeline.py`: data screening, hourly input construction,
  patient splitting, and baseline evaluation.
- `scripts/validate_local.py`: independent checks of local processing outputs.
- `tests/test_pipeline.py`: synthetic tests without patient data.
- `scripts/run_tests.py`: runs the tests and writes their actual result to
  `reports/synthetic_test_summary.json`.
- `configs/pipeline.json`: local data paths and processing settings.
- `environment.yml`: Python environment specification.

- `docs/METHODS.md`: inclusion rules, input construction and evaluation details.

## Running locally

Access to MIMIC-IV 2.2 must be obtained separately.
This repository does not distribute the dataset.

Create and activate the environment:

```powershell
conda env create -f environment.yml
conda activate vitalalert
```

If the environment already exists, activate it without creating it again.

Run the synthetic tests from the repository root:

```powershell
python -B scripts/run_tests.py
```

Before processing real data, edit `configs/pipeline.json`:
set `raw_root` to the directory containing the `hosp` and `icu` folders,
and set `private_root` to a local output directory outside this repository.

```powershell
python -B -u scripts/pipeline.py
```

Use a new `run_id` when code or processing settings change.
A repeated run with the same settings may reuse an existing scan.

## Code version used for these results

The reported metrics were calculated with the code at
[commit d6f3622](https://github.com/hsang325/VitalAlert/tree/d6f36228860ff6620d91eabba5f1fdca950112c2).
The subsequent review clarified comments and spacing and removed an unused
variable in the pipeline. Its calculation statements are unchanged, as checked
by comparing Python syntax trees after excluding docstrings and that variable.
Real patient data were not processed again during this review.

`reports/aggregate_summary.json` and `reports/validation_summary.json` retain
the original run results. The latter contains a historical synthetic test count;
current `validate_local.py` reports only the window checks it actually performs.
New synthetic test results are recorded separately by `run_tests.py`.
`reports/code_review_provenance.json` links the original run to this review.

The existing run folder checks the full source-file fingerprint, including
comments and formatting. Use a new `run_id` for a processing run with the current
source; do not overwrite the old run's fingerprint.

## Data handling

Raw data, patient-level outputs, and detailed processing logs remain local
and must not be uploaded to this repository.
The pipeline does not modify the source data.
