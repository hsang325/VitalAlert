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
- `configs/pipeline.json`: local data paths and processing settings.
- `environment.yml`: Python environment specification.

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
python -B -m unittest discover -s tests -v
```

Before processing real data, edit `configs/pipeline.json`:
set `raw_root` to the directory containing the `hosp` and `icu` folders,
and set `private_root` to a local output directory outside this repository.

```powershell
python -B -u scripts/pipeline.py
```

Use a new `run_id` when code or processing settings change.
A repeated run with the same settings may reuse an existing scan.

## Data handling

Raw data, patient-level outputs, and detailed processing logs remain local
and must not be uploaded to this repository.
The pipeline does not modify the source data.