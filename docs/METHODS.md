# VitalAlert: initial heart-rate preprocessing protocol

Protocol date: 2026-09-26. Implemented in `scripts/pipeline.py`.

## Scope and rationale

The current prediction task uses six hours of heart-rate history to predict the
mean heart rate over the following hour. Other candidate vital signs have been
screened at record level; their forecasting inputs are not yet constructed.
LSTM training and clinical alert evaluation remain future work.

The reported results use commit d6f3622. A subsequent code review clarified
comments and formatting and removed an unused variable without changing the
pipeline calculations. See `reports/code_review_provenance.json` for the link
between the original results and the reviewed source. Original aggregate and
window-check reports are retained; newly run synthetic tests have a separate
report in `reports/synthetic_test_summary.json`.

## Data and integrity

- Existing local MIMIC-IV 2.2 `hosp` and `icu` modules; no redownload.
- Four compressed input files are hashed using SHA-256 and compared with the
  accompanying local SHA256SUMS.txt. This detects corruption relative to that
  manifest; it does not independently authenticate the manifest against a remote source.
- The entire compressed chartevents file is scanned for 13 predefined item IDs.
  CSV parsing errors are not ignored. This is not a head-of-file sample.
- Source files are read-only. Detailed artifacts remain outside the code workspace.
- Public reference: https://physionet.org/content/mimiciv/2.2/

## Cohort and split

- Join ICU stays to patients. Require positive stay duration and approximate
  age at ICU admission >=18, computed as anchor_age + year(intime) - anchor_year.
  Approximation and top-coded elderly ages limit age interpretation.
- Match events on subject_id, hadm_id and stay_id, then require charttime inside
  the ICU stay. Do not join on stay_id alone when filtering source events.
- Split subjects using SHA256(str(subject_id) + ':20260926'), first 8 hex digits,
  converted to integer modulo 100: 0-69 train, 70-84 validation, 85-99 test.
- Every stay of a subject shares the split. Ratios are approximate, not exact.
  This is a random patient split, not a temporal or external validation study.

## Candidate selection and quality rules

- Heart rate: 220045; SpO2: 220277; respiratory rate: 220210 and 224690.
- Blood pressure: 220050/51/52 (arterial), 220179/80/81 (noninvasive),
  225309/10/12 (alternate ART channels). Channels remain separate.
- Confirm all item IDs, source table and units against local d_items.
- Require exact expected units after trimming and case normalization. Unit
  mismatches are excluded, not silently converted.
- Public mimic-code plausibility ranges: HR (0,300), SpO2 (0,100], RR (0,70),
  SBP (0,400), DBP/MAP (0,300). These are cleaning bounds, not normal/alert limits.
- Exclude warning !=0; missing warning is treated as unflagged. Retain original
  selected records in the restricted database for later sensitivity checks.
- Remove exact duplicate model-relevant rows including storetime. Multiple
  nonidentical records at one charttime remain and contribute to the mean.
- Per-rule exclusion counts overlap; do not sum them as a sequential exclusion flow.
- Reference: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/vitalsign.sql

## Hourly heart-rate representation

- Right-closed one-hour bins (t-1h,t], aligned to calendar-hour boundaries in the
  deidentified timestamps. Include only bins entirely inside the ICU stay.
- Hourly observed target: arithmetic mean of valid charted HR records.
- At prediction time t, reconstruct each of the six historical input bins using
  only records already stored by t (storetime <=t). Null storetime cannot establish
  input availability. A delayed record becomes eligible at later prediction times
  after it has actually been stored; it never changes an earlier prediction.
- The diagnostic first run froze each bin at its own closure and never reused
  delayed records. Its severe attrition motivated this data-availability correction.
  Its baseline metrics are not the final v2 results. No model parameters were tuned.
- storetime earlier than charttime is counted for review but retained; the input
  boundary must still be at or after charttime. This is a remaining data-quality issue.
- Fill only one missing input bin from the immediately preceding unfilled observed
  bin reconstructed at the same prediction time. Up to seven chart bins are read
  to allow this fill for the earliest of six input bins. Longer gaps remain missing.
  Filling never crosses ICU stays or includes a partial boundary bin.
- Preserve six observed/imputed masks alongside the six input values.
- Never impute labels. Six usable consecutive inputs and a next-hour observed
  label are required for an eligible window. Exclusions can create selection bias.
- Prediction target is the MEAN OF THE NEXT HOUR, not an exact point measurement
  at t+60 minutes. This operational definition needs to be explicit in reporting.
- Normalization is unnecessary for these raw-unit baselines. Before LSTM, fit
  any scaler on training data only and freeze it for validation/test.
- `hourly` exports provide observed labels and diagnostic availability at each
  bin's own closure. `windows` exports hold the final as-of reconstructed model
  inputs; use them for training, not the diagnostic hourly feature column.

## Fixed baselines and evaluation

1. Persistence: predict the last available input-hour HR mean (x5).
2. Six-hour mean: arithmetic mean of x0...x5.
3. Training mean: average training-window label; never fit it on validation/test.

All three are evaluated on identical windows. Report micro MAE and RMSE in bpm
and macro MAE averaged first within patient and then across patients.
Longer stays contribute more windows to micro metrics. No model selection or
hyperparameter tuning is performed in this run. Test results are preliminary
descriptive results and have been inspected; do not call this an untouched final
test. Further model development should use validation and a documented final
evaluation policy. No confidence intervals, external validation, alert utility or
clinical deployment performance have been established.

## Validation

- Synthetic tests exercise boundary alignment, delayed charting, missing storetime,
  finite-value/range/unit/warning filters, duplicates, stays, bounded filling,
  future-value perturbation, input windows and known baseline errors.
- Real-data assertions check patient separation, exact horizon, observed labels,
  stay boundaries, nonmissing inputs and Parquet export row counts.
- Assertions establish implementation consistency, not clinical validity.
