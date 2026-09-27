"""Local MIMIC preprocessing; stdout contains stage names only, never records.

Patient-level databases, datasets, and errors stay outside the code workspace.
Only pre-defined cohort-level summary statistics are written to reports/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback

import duckdb

ROOT = Path(__file__).resolve().parents[1]
# itemid, name, unit, exclusive lower bound, upper bound, upper inclusive
VARIABLES = [
    (220045, "heart_rate", "bpm", 0, 300, False),
    (220277, "spo2", "%", 0, 100, True),
    (220210, "respiratory_rate", "insp/min", 0, 70, False),
    (224690, "respiratory_rate_total", "insp/min", 0, 70, False),
    (220050, "sbp_arterial", "mmHg", 0, 400, False),
    (220051, "dbp_arterial", "mmHg", 0, 300, False),
    (220052, "map_arterial", "mmHg", 0, 300, False),
    (220179, "sbp_noninvasive", "mmHg", 0, 400, False),
    (220180, "dbp_noninvasive", "mmHg", 0, 300, False),
    (220181, "map_noninvasive", "mmHg", 0, 300, False),
    (225309, "sbp_art_alternate", "mmHg", 0, 400, False),
    (225310, "dbp_art_alternate", "mmHg", 0, 300, False),
    (225312, "map_art_alternate", "mmHg", 0, 300, False),
]


def sql_path(path):
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def records(con, query):
    cursor = con.execute(query)
    names = [x[0] for x in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def stage(name):
    print(name, flush=True)


def register_variables(con):
    con.execute("CREATE OR REPLACE TABLE variables(itemid INTEGER, name VARCHAR, unit VARCHAR, lower_bound DOUBLE, upper_bound DOUBLE, upper_inclusive BOOLEAN)")
    con.executemany("INSERT INTO variables VALUES (?, ?, ?, ?, ?, ?)", VARIABLES)


def build_clean_events(con):
    """Apply mutually auditable record-level quality flags without losing raw data."""
    con.execute("""
        CREATE OR REPLACE TABLE assessed AS
        SELECT e.*, v.name, s.intime, s.outtime, s.split,
          s.stay_id IS NOT NULL AS matched_stay,
          coalesce(e.charttime >= s.intime AND e.charttime <= s.outtime, false) AS within_stay,
          coalesce(isfinite(e.valuenum) AND e.valuenum > v.lower_bound AND
            (e.valuenum < v.upper_bound OR (v.upper_inclusive AND e.valuenum = v.upper_bound)), false) AS value_valid,
          coalesce(lower(trim(e.valueuom)) = lower(v.unit), false) AS unit_valid,
          coalesce(e.warning, 0) = 0 AS warning_clear
        FROM candidates e JOIN variables v USING(itemid)
        LEFT JOIN stays s ON e.stay_id = s.stay_id AND e.subject_id = s.subject_id AND e.hadm_id = s.hadm_id
    """)
    # Exact duplicate model-relevant records removed, keeping distinct store times.
    con.execute("""
        CREATE OR REPLACE TABLE clean_events AS
        SELECT DISTINCT subject_id, hadm_id, stay_id, itemid, name, charttime, storetime,
            valuenum, valueuom, split
        FROM assessed
        WHERE matched_stay AND within_stay AND value_valid AND unit_valid AND warning_clear
    """)


def build_hourly(con):
    """Full UTC-like deidentified calendar hour bins; no cross-stay fill.

    At boundary t, (t-1h, t] is the current hour. Full stay-contained bins only.
    Features use only records with storetime <= t. Labels use observed means.
    One empty feature bin can use the preceding observed feature, never more.
    """
    con.execute("""
        CREATE OR REPLACE TABLE hourly_observed AS
        WITH tagged AS (
          SELECT *, CASE WHEN charttime = date_trunc('hour', charttime)
              THEN charttime ELSE date_trunc('hour', charttime) + INTERVAL '1 hour' END AS t
          FROM clean_events WHERE itemid = 220045
        )
        SELECT subject_id, stay_id, t,
          avg(valuenum) AS observed_hr,
          avg(valuenum) FILTER (WHERE storetime IS NOT NULL AND storetime <= t) AS timely_hr,
          count(*) AS n_observations,
          count(*) FILTER (WHERE storetime IS NOT NULL AND storetime <= t) AS n_timely
        FROM tagged GROUP BY subject_id, stay_id, t
    """)
    con.execute("""
        CREATE OR REPLACE TABLE hourly AS
        WITH grid AS (
          SELECT s.subject_id, s.stay_id, s.split, g.t
          FROM stays s,
          LATERAL generate_series(
            CASE WHEN s.intime = date_trunc('hour', s.intime)
              THEN s.intime + INTERVAL '1 hour'
              ELSE date_trunc('hour', s.intime) + INTERVAL '2 hours' END,
            date_trunc('hour', s.outtime), INTERVAL '1 hour'
          ) AS g(t)
        ), merged AS (
          SELECT g.*, h.observed_hr, h.timely_hr,
            coalesce(h.n_observations, 0) AS n_observations,
            coalesce(h.n_timely, 0) AS n_timely
          FROM grid g LEFT JOIN hourly_observed h USING(subject_id, stay_id, t)
        )
        SELECT *,
          coalesce(timely_hr, lag(timely_hr) OVER w) AS feature_hr,
          timely_hr IS NOT NULL AS observed_mask,
          lead(observed_hr) OVER w AS target_hr,
          lead(t) OVER w AS target_time
        FROM merged WINDOW w AS (PARTITION BY subject_id, stay_id ORDER BY t)
    """)


def build_windows(con):
    """Reconstruct each historical bin AS OF the prediction boundary.

    Delayed measurements become usable after they are stored, but cannot change
    earlier predictions. Seven raw historical bins permit one-bin forward fill
    for each of the six input bins, always from an observed older bin.
    """
    con.execute("""
      CREATE OR REPLACE TABLE asof_inputs AS
      WITH bucketed AS (
        SELECT e.*, CASE WHEN charttime=date_trunc('hour',charttime) THEN charttime
          ELSE date_trunc('hour',charttime)+INTERVAL '1 hour' END AS b
        FROM clean_events e WHERE itemid=220045
      ), full_bins AS (
        SELECT b.* FROM bucketed b JOIN stays s USING(stay_id)
        WHERE b.b-INTERVAL '1 hour'>=s.intime AND b.b<=s.outtime
      ), available AS (
        SELECT e.subject_id,e.stay_id,e.b + offsets.k*INTERVAL '1 hour' AS t,
          offsets.k, e.valuenum
        FROM full_bins e CROSS JOIN range(0,7) AS offsets(k)
        WHERE e.storetime IS NOT NULL AND e.storetime<=e.b+offsets.k*INTERVAL '1 hour'
      )
      SELECT subject_id,stay_id,t,
        avg(valuenum) FILTER(WHERE k=0) AS r0,
        avg(valuenum) FILTER(WHERE k=1) AS r1,
        avg(valuenum) FILTER(WHERE k=2) AS r2,
        avg(valuenum) FILTER(WHERE k=3) AS r3,
        avg(valuenum) FILTER(WHERE k=4) AS r4,
        avg(valuenum) FILTER(WHERE k=5) AS r5,
        avg(valuenum) FILTER(WHERE k=6) AS r6
      FROM available GROUP BY subject_id,stay_id,t
    """)
    feature_terms = [f"coalesce(a.r{lag},a.r{lag+1}) AS x{i}" for i, lag in enumerate(range(5, -1, -1))]
    mask_terms = [f"a.r{lag} IS NOT NULL AS mask{i}" for i, lag in enumerate(range(5, -1, -1))]
    con.execute(f"""
      CREATE OR REPLACE TABLE windows AS
      WITH lagged AS (
        SELECT h.subject_id, h.stay_id, h.split, h.t, h.target_time, h.target_hr,
          {', '.join(feature_terms + mask_terms)},
          h.t-INTERVAL '5 hours' AS first_input_time
        FROM hourly h LEFT JOIN asof_inputs a USING(subject_id,stay_id,t)
        JOIN stays s ON h.stay_id=s.stay_id
        WHERE h.t-INTERVAL '6 hours'>=s.intime
      )
      SELECT *, x5 AS prediction_persistence,
        (x0+x1+x2+x3+x4+x5)/6.0 AS prediction_history_mean
      FROM lagged
      WHERE {' AND '.join(f'x{i} IS NOT NULL' for i in range(6))}
        AND target_hr IS NOT NULL AND target_time = t + INTERVAL '1 hour'
        AND first_input_time = t - INTERVAL '5 hours'
    """)


def quality_summary(con):
    return records(con, """
      SELECT v.itemid, v.name,
        (SELECT count(*) FROM candidates c WHERE c.itemid=v.itemid) AS candidate_records,
        (SELECT count(*) FROM assessed a WHERE a.itemid=v.itemid AND NOT matched_stay) AS unmatched_records,
        (SELECT count(*) FROM assessed a WHERE a.itemid=v.itemid AND matched_stay AND NOT within_stay) AS outside_stay_records,
        (SELECT count(*) FROM assessed a WHERE a.itemid=v.itemid AND NOT value_valid) AS invalid_value_records,
        (SELECT count(*) FROM assessed a WHERE a.itemid=v.itemid AND NOT unit_valid) AS unit_mismatch_records,
        (SELECT count(*) FROM assessed a WHERE a.itemid=v.itemid AND NOT warning_clear) AS warning_records,
        (SELECT count(*) FROM clean_events c WHERE c.itemid=v.itemid) AS clean_records,
        (SELECT count(DISTINCT stay_id) FROM clean_events c WHERE c.itemid=v.itemid) AS stays_with_clean_record
      FROM variables v ORDER BY v.itemid
    """)


def safe_summary(con):
    """Return only a fixed allow-list of full-cohort aggregates. No record samples."""
    train_mean = con.execute("SELECT avg(target_hr) FROM windows WHERE split='train'").fetchone()[0]
    if train_mean is None:
        raise RuntimeError("No training windows")
    con.execute("CREATE OR REPLACE TABLE training_constant AS SELECT ?::DOUBLE AS value", [train_mean])
    metrics = []
    for model, expression in [("persistence", "prediction_persistence"), ("six_hour_mean", "prediction_history_mean"), ("training_mean", "(SELECT value FROM training_constant)")]:
        rows = records(con, f"""
          WITH errors AS (
            SELECT split, subject_id, target_hr - {expression} AS error FROM windows
          ), patients AS (
            SELECT split, subject_id, avg(abs(error)) AS patient_mae FROM errors GROUP BY split, subject_id
          )
          SELECT split, count(*) AS windows, count(DISTINCT subject_id) AS patients,
            avg(abs(error)) AS mae_bpm, sqrt(avg(error*error)) AS rmse_bpm,
            (SELECT avg(patient_mae) FROM patients p WHERE p.split=e.split) AS patient_macro_mae_bpm
          FROM errors e GROUP BY split ORDER BY split
        """)
        for row in rows:
            row["model"] = model
        metrics.extend(rows)
    asserts = {
        "patient_split_overlap": con.execute("SELECT count(*) FROM (SELECT subject_id FROM stays GROUP BY subject_id HAVING count(DISTINCT split)>1)").fetchone()[0],
        "window_split_overlap": con.execute("SELECT count(*) FROM (SELECT subject_id FROM windows GROUP BY subject_id HAVING count(DISTINCT split)>1)").fetchone()[0],
        "bad_horizon": con.execute("SELECT count(*) FROM windows WHERE target_time != t + INTERVAL '1 hour'").fetchone()[0],
        "unobserved_targets": con.execute("SELECT count(*) FROM windows w LEFT JOIN hourly h ON w.stay_id=h.stay_id AND w.target_time=h.t WHERE h.observed_hr IS NULL OR w.target_hr != h.observed_hr").fetchone()[0],
        "bad_stay_bounds": con.execute("SELECT count(*) FROM windows w JOIN stays s USING(stay_id) WHERE w.first_input_time-INTERVAL '1 hour'<s.intime OR w.target_time>s.outtime").fetchone()[0],
        "missing_inputs": con.execute("SELECT count(*) FROM windows WHERE x0 IS NULL OR x1 IS NULL OR x2 IS NULL OR x3 IS NULL OR x4 IS NULL OR x5 IS NULL").fetchone()[0],
        "asof_pivot_disagrees": con.execute("SELECT count(*) FROM windows w JOIN asof_inputs a USING(subject_id,stay_id,t) WHERE w.x0 IS DISTINCT FROM coalesce(a.r5,a.r6) OR w.x5 IS DISTINCT FROM coalesce(a.r0,a.r1)").fetchone()[0],
    }
    if any(asserts.values()):
        raise RuntimeError("Integrity assertion failed")
    cohort = records(con, "SELECT split,count(*) AS stays,count(DISTINCT subject_id) AS patients FROM stays GROUP BY split ORDER BY split")
    hourly = records(con, """SELECT split,count(*) AS full_hour_bins,
      count(observed_hr) AS observed_bins,count(timely_hr) AS timely_bins,
      count(feature_hr) AS usable_feature_bins,
      count(*) FILTER(WHERE feature_hr IS NOT NULL AND timely_hr IS NULL) AS filled_bins
      FROM hourly GROUP BY split ORDER BY split""")
    intervals = records(con, """WITH times AS (
      SELECT DISTINCT stay_id,charttime FROM clean_events WHERE itemid=220045
    ), gaps AS (
      SELECT epoch(charttime-lag(charttime) OVER(PARTITION BY stay_id ORDER BY charttime))/60 AS minutes FROM times
    ) SELECT count(*) AS positive_intervals, median(minutes) AS median_minutes,
      quantile_cont(minutes,0.25) AS p25_minutes, quantile_cont(minutes,0.75) AS p75_minutes,
      avg(CASE WHEN minutes=60 THEN 1.0 ELSE 0.0 END) AS fraction_exactly_60min
      FROM gaps WHERE minutes>0""")[0]
    delay = records(con, """SELECT count(*) AS clean_hr_records,
      count(*) FILTER(WHERE storetime IS NULL) AS missing_storetime,
      count(*) FILTER(WHERE storetime<charttime) AS storetime_before_charttime,
      count(*) FILTER(WHERE storetime>CASE WHEN charttime=date_trunc('hour',charttime)
        THEN charttime ELSE date_trunc('hour',charttime)+INTERVAL '1 hour' END) AS after_bin_close
      FROM clean_events WHERE itemid=220045""")[0]
    duplicate_count = con.execute("SELECT (SELECT count(*) FROM assessed WHERE matched_stay AND within_stay AND value_valid AND unit_valid AND warning_clear)-(SELECT count(*) FROM clean_events)").fetchone()[0]
    window_coverage=records(con,"""SELECT w.split,count(*) AS windows,count(DISTINCT w.subject_id) AS patients,
      count(DISTINCT w.stay_id) AS stays,
      sum(CAST(NOT mask0 AS INTEGER)+CAST(NOT mask1 AS INTEGER)+CAST(NOT mask2 AS INTEGER)+CAST(NOT mask3 AS INTEGER)+CAST(NOT mask4 AS INTEGER)+CAST(NOT mask5 AS INTEGER)) AS imputed_input_slots,
      6*count(*) AS total_input_slots
      FROM windows w GROUP BY w.split ORDER BY w.split""")
    eligible_origins=records(con,"""SELECT h.split,count(*) AS origins_with_six_hours_and_observed_target
      FROM hourly h JOIN stays s USING(stay_id)
      WHERE h.target_hr IS NOT NULL AND h.t-INTERVAL '6 hours'>=s.intime GROUP BY h.split ORDER BY h.split""")
    delay_quantiles=records(con,"""SELECT median(epoch(storetime-charttime)/60) AS median_minutes,
      quantile_cont(epoch(storetime-charttime)/60,0.9) AS p90_minutes
      FROM clean_events WHERE itemid=220045 AND storetime>=charttime""")[0]
    return {"cohort_by_split":cohort,"quality_by_item":quality_summary(con),"hourly_by_split":hourly,
            "baseline_metrics":metrics,"integrity_checks":asserts,"heart_rate_intervals":intervals,
            "heart_rate_recording_delay":delay,"duplicate_model_records_removed":duplicate_count,
            "training_target_mean_bpm":train_mean,"window_coverage_by_split":window_coverage,
            "eligible_origins_by_split":eligible_origins,"nonnegative_recording_delay":delay_quantiles,
            "input_policy":"Each historical bin reconstructed using only records stored by the current prediction time; one-bin forward fill from an observed older bin."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default=str(ROOT/"configs/pipeline.json"))
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config["lookback_hours"] != 6:
        raise ValueError("This version supports six-hour lookback only")
    raw=Path(config["raw_root"]).resolve()
    private=Path(config["private_root"]).resolve()/config["run_id"]
    if private.is_relative_to(ROOT) or private.is_relative_to(raw):
        raise ValueError("Restricted output must be outside source and code directories")
    private.mkdir(parents=True,exist_ok=True)
    start=time.time()
    con=None
    try:
        fingerprint=hashlib.sha256((Path(__file__).read_bytes()+json.dumps(config,sort_keys=True).encode())).hexdigest()
        marker=private/"run_fingerprint.txt"
        if marker.exists() and marker.read_text()!=fingerprint:
            raise RuntimeError("Existing run has different code/config; use a new run_id")
        marker.write_text(fingerprint)
        (private/"pipeline_source.py").write_bytes(Path(__file__).read_bytes())
        (private/"config_snapshot.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
        stage("1/7 Verifying source checksums (no patient output)")
        manifest={line.split(maxsplit=1)[1].strip().lstrip('*'):line.split()[0] for line in (raw/"SHA256SUMS.txt").read_text().splitlines() if line.strip()}
        checks=[]
        for name in ["icu/d_items.csv.gz","icu/icustays.csv.gz","hosp/patients.csv.gz","icu/chartevents.csv.gz"]:
            p=raw/name
            with p.open("rb") as f:
                digest=hashlib.file_digest(f,"sha256").hexdigest()
            ok=digest==manifest.get(name)
            checks.append({"file":name,"bytes":p.stat().st_size,"sha256":digest,"matches_local_manifest":ok})
            if not ok:
                raise RuntimeError("Source checksum mismatch")
        con=duckdb.connect(str(private/"pipeline.duckdb"))
        con.execute(f"SET memory_limit='{config['memory_limit']}'")
        con.execute(f"SET threads={int(config['threads'])}")
        con.execute("SET enable_progress_bar=false")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory={sql_path(private/'spill')}")
        register_variables(con)
        stage("2/7 Reading metadata and constructing patient-level splits")
        con.execute(f"CREATE OR REPLACE TABLE items AS SELECT * FROM read_csv({sql_path(raw/'icu/d_items.csv.gz')},header=true)")
        mismatches=con.execute("SELECT count(*) FROM variables v LEFT JOIN items d USING(itemid) WHERE d.itemid IS NULL OR d.linksto!='chartevents' OR lower(trim(d.unitname))!=lower(v.unit)").fetchone()[0]
        if mismatches:
            raise RuntimeError("Item dictionary metadata mismatch")
        con.execute(f"CREATE OR REPLACE TABLE all_stays AS SELECT * FROM read_csv({sql_path(raw/'icu/icustays.csv.gz')},header=true)")
        con.execute(f"CREATE OR REPLACE TABLE patients AS SELECT * FROM read_csv({sql_path(raw/'hosp/patients.csv.gz')},header=true)")
        con.execute(f"""CREATE OR REPLACE TABLE stays AS
          WITH eligible AS (
            SELECT s.*, p.anchor_age,
              p.anchor_age+year(s.intime)-p.anchor_year AS approximate_age,
              CAST('0x'||substr(sha256(CAST(s.subject_id AS VARCHAR)||':{config['seed']}'),1,8) AS UBIGINT)%100 AS bucket
            FROM all_stays s JOIN patients p USING(subject_id)
            WHERE s.outtime>s.intime AND p.anchor_age+year(s.intime)-p.anchor_year>=18
          ) SELECT *, CASE WHEN bucket<70 THEN 'train' WHEN bucket<85 THEN 'validation' ELSE 'test' END AS split FROM eligible""")
        if con.execute("SELECT count(*)-count(DISTINCT stay_id) FROM stays").fetchone()[0]:
            raise RuntimeError("Duplicate cohort stay IDs")
        stage("3/7 Scanning complete compressed chartevents; this is the long step")
        completed_scan=private/"scan_complete.txt"
        if not completed_scan.exists():
            item_list=','.join(str(v[0]) for v in VARIABLES)
            con.execute(f"""CREATE OR REPLACE TABLE candidates AS
              SELECT try_cast(subject_id AS BIGINT) AS subject_id,try_cast(hadm_id AS BIGINT) AS hadm_id,
                try_cast(stay_id AS BIGINT) AS stay_id,try_cast(itemid AS INTEGER) AS itemid,
                try_cast(charttime AS TIMESTAMP) AS charttime,try_cast(storetime AS TIMESTAMP) AS storetime,
                try_cast(valuenum AS DOUBLE) AS valuenum,valueuom,try_cast(warning AS INTEGER) AS warning
              FROM read_csv({sql_path(raw/'icu/chartevents.csv.gz')},header=true,all_varchar=true,ignore_errors=false)
              WHERE itemid IN ({','.join(repr(str(v[0])) for v in VARIABLES)})""")
            con.execute("CHECKPOINT")
            completed_scan.write_text("Complete scan succeeded; not a sample")
        stage("4/7 Filtering measurements and constructing hourly heart-rate data")
        build_clean_events(con)
        build_hourly(con)
        stage("5/7 Constructing six-hour windows and evaluating fixed baselines")
        build_windows(con)
        summary=safe_summary(con)
        stage("6/7 Exporting local restricted datasets and validation artifacts")
        for table in ["stays","clean_events","hourly","windows"]:
            for split in ["train","validation","test"]:
                path=private/f"{table}_{split}.parquet"
                con.execute(f"COPY (SELECT * FROM {table} WHERE split='{split}') TO {sql_path(path)} (FORMAT PARQUET,COMPRESSION ZSTD)")
                exported=con.execute(f"SELECT count(*) FROM read_parquet({sql_path(path)})").fetchone()[0]
                expected=con.execute(f"SELECT count(*) FROM {table} WHERE split='{split}'").fetchone()[0]
                if expected != exported:
                    raise RuntimeError("Export count mismatch")
        summary.update({"created_at_utc":datetime.now(timezone.utc).isoformat(),"run_id":config["run_id"],
          "config":config,"code_config_sha256":fingerprint,"input_checksums":checks,
          "full_chartevents_scan":True,"export_row_counts_match":True,
          "cohort_excluded_stays":con.execute("SELECT (SELECT count(*) FROM all_stays)-(SELECT count(*) FROM stays)").fetchone()[0],
          "elapsed_seconds":round(time.time()-start,1),
          "versions":{p:importlib.metadata.version(p) for p in ['duckdb','numpy','pandas','pyarrow']},
          "python":sys.version.split()[0],"restricted_output_root":str(private),
          "test_metrics_status":"preliminary descriptive run; no tuning performed; test set has now been inspected"})
        # These aggregate outputs intentionally contain no subject/stay IDs or clinical timestamps.
        reports=ROOT/"reports"
        reports.mkdir(exist_ok=True)
        payload=json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)
        (private/"aggregate_summary.json").write_text(payload,encoding="utf-8")
        (reports/"aggregate_summary.json").write_text(payload,encoding="utf-8")
        (private/"SUCCESS.txt").write_text("All integrity and export checks passed.")
        stage("7/7 SUCCESS. Aggregate summary saved; patient-level outputs remain local.")
    except Exception as exc:
        (private/"error_private.log").write_text(traceback.format_exc(),encoding="utf-8")
        safe_error={"exception_class":type(exc).__name__,"frames":[{"file":Path(f.filename).name,"line":f.lineno,"function":f.name} for f in traceback.extract_tb(exc.__traceback__)]}
        (private/"error_safe.json").write_text(json.dumps(safe_error,indent=2),encoding="utf-8")
        # Never print an exception message/traceback from a real-data operation.
        print('FAILED; exception class: '+type(exc).__name__+'. Detailed error retained locally.',flush=True)
        return 1
    finally:
        if con is not None:
            con.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
