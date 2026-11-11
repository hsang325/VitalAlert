# MIMIC 기록 → 시간별 심박수 → 6시간 입력/다음 시간 정답 → baseline 평가
# 진입점 main(). candidates(후보), assessed(검사), clean_events(정제), hourly(시간 격자), windows(예측 창) 순서
# asof_inputs: 예측 시점까지 저장된 기록으로 재구성한 과거 입력
# baseline: 최신값 유지, 6시간 평균, 학습 정답 평균
# 환자별 표·상세 오류는 로컬 경로, 보고용 집계는 reports/에 저장

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
SPLITS = ("train", "validation", "test")
SOURCE_FILES = (
    "icu/d_items.csv.gz",
    "icu/icustays.csv.gz",
    "hosp/patients.csv.gz",
    "icu/chartevents.csv.gz",
)

# 항목 번호, 이름, 단위, 하한, 상한, 상한 포함 여부 순서
# 후보 항목 전체의 정제 품질을 집계하되 예측 입력은 심박수만 사용
# 하한 제외. 마지막 값이 True인 항목만 상한 포함
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
    # 작은따옴표가 포함된 파일 경로도 SQL 문자열로 처리
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def records(con, query):
    # 집계 결과를 열 이름으로 접근할 수 있는 사전 목록으로 변환
    cursor = con.execute(query)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def register_variables(con):
    # 항목별 단위·정제 범위를 SQL 조회용 표로 등록
    con.execute("""
        CREATE OR REPLACE TABLE variables (
            itemid INTEGER,
            name VARCHAR,
            unit VARCHAR,
            lower_bound DOUBLE,
            upper_bound DOUBLE,
            upper_inclusive BOOLEAN
        )
    """)
    con.executemany("INSERT INTO variables VALUES (?, ?, ?, ?, ?, ?)", VARIABLES)


def build_clean_events(con):
    # candidates에서 환자·체류·값·단위 검사 후 clean_events 구성
    # 제외 사유별 집계를 위해 검사 결과는 assessed에 보관. 통과 기록의 중복 제거
    # 체류 연결 실패 등 비교값이 없으면 coalesce(..., false)로 제외 처리
    # 심박수는 storetime 누락·시간 역전 기록을 입력과 정답 모두에서 제외
    # 지연 저장 기록은 유지하고, 입력 가능 여부는 예측 시점에서 판단
    # 값 범위는 정제용이며 임상 정상 범위가 아님
    con.execute("""
        CREATE OR REPLACE TABLE assessed AS
        SELECT
            e.*,
            v.name,
            s.intime,
            s.outtime,
            s.split,
            s.stay_id IS NOT NULL AS matched_stay,
            coalesce(
                e.charttime >= s.intime AND e.charttime <= s.outtime,
                false
            ) AS within_stay,
            coalesce(
                isfinite(e.valuenum)
                AND e.valuenum > v.lower_bound
                AND (
                    e.valuenum < v.upper_bound
                    OR (v.upper_inclusive AND e.valuenum = v.upper_bound)
                ),
                false
            ) AS value_valid,
            coalesce(lower(trim(e.valueuom)) = lower(v.unit), false) AS unit_valid,
            coalesce(e.warning, 0) = 0 AS warning_clear,
            CASE
                WHEN e.itemid = 220045
                    THEN coalesce(e.storetime >= e.charttime, false)
                ELSE true
            END AS timestamp_valid
        FROM candidates e
        JOIN variables v USING (itemid)
        LEFT JOIN stays s
            ON e.stay_id = s.stay_id
            AND e.subject_id = s.subject_id
            AND e.hadm_id = s.hadm_id
    """)

    # 저장 시각에 따라 입력 가능 시점이 달라지므로 storetime이 다르면 별도 기록으로 유지
    con.execute("""
        CREATE OR REPLACE TABLE clean_events AS
        SELECT DISTINCT
            subject_id, hadm_id, stay_id, itemid, name,
            charttime, storetime, valuenum, valueuom, split
        FROM assessed
        WHERE matched_stay
            AND within_stay
            AND value_valid
            AND unit_valid
            AND warning_clear
            AND timestamp_valid
    """)


def build_hourly(con):
    # 정제 심박수를 (t-1시간, t] 구간으로 집계. 시작 경계 제외, 끝 경계 포함
    # observed_hr: 측정 구간 평균. timely_hr: 구간 종료까지 저장된 값의 평균
    # charttime은 측정 시각, storetime은 저장 시각으로 지연 가능
    # feature_hr은 구간 종료 당시 가용성을 보는 진단용 값
    # 실제 모델 입력은 build_windows()에서 예측 시점 기준으로 재계산
    # 다음 구간 observed_hr을 정답으로 사용. 정답 결측은 보완하지 않음
    # 정각 기록은 그 정각에 끝나는 구간에 포함
    con.execute("""
        CREATE OR REPLACE TABLE hourly_observed AS
        WITH tagged AS (
            SELECT
                *,
                CASE
                    WHEN charttime = date_trunc('hour', charttime) THEN charttime
                    ELSE date_trunc('hour', charttime) + INTERVAL '1 hour'
                END AS t
            FROM clean_events
            WHERE itemid = 220045
        )
        SELECT
            subject_id,
            stay_id,
            t,
            avg(valuenum) AS observed_hr,
            avg(valuenum) FILTER (
                WHERE storetime IS NOT NULL AND storetime <= t
            ) AS timely_hr,
            count(*) AS n_observations,
            count(*) FILTER (
                WHERE storetime IS NOT NULL AND storetime <= t
            ) AS n_timely
        FROM tagged
        GROUP BY subject_id, stay_id, t
    """)

    # 긴 결측이 사라지지 않도록 grid에 빈 시간도 포함
    # 입실·퇴실에 걸친 불완전한 1시간 구간 제외
    # lag/lead는 같은 환자·체류 안에서 이전/다음 구간 참조
    con.execute("""
        CREATE OR REPLACE TABLE hourly AS
        WITH grid AS (
            SELECT s.subject_id, s.stay_id, s.split, g.t
            FROM stays s,
            LATERAL generate_series(
                CASE
                    WHEN s.intime = date_trunc('hour', s.intime)
                        THEN s.intime + INTERVAL '1 hour'
                    ELSE date_trunc('hour', s.intime) + INTERVAL '2 hours'
                END,
                date_trunc('hour', s.outtime),
                INTERVAL '1 hour'
            ) AS g(t)
        ),
        merged AS (
            SELECT
                g.*,
                h.observed_hr,
                h.timely_hr,
                coalesce(h.n_observations, 0) AS n_observations,
                coalesce(h.n_timely, 0) AS n_timely
            FROM grid g
            LEFT JOIN hourly_observed h USING (subject_id, stay_id, t)
        )
        SELECT
            *,
            coalesce(timely_hr, lag(timely_hr) OVER w) AS feature_hr,
            timely_hr IS NOT NULL AS observed_mask,
            lead(observed_hr) OVER w AS target_hr,
            lead(t) OVER w AS target_time
        FROM merged
        WINDOW w AS (PARTITION BY subject_id, stay_id ORDER BY t)
    """)


def build_windows(con):
    # 과거 6시간 입력과 다음 시간 정답으로 예측 창 구성
    # storetime <= 예측 시점 t인 기록만 구간별 평균에 포함
    # 결측은 바로 이전 관측 구간으로 한 번만 보완. 가장 오래된 입력의 보완을 위해 7개 구간 조회
    # bucketed: 측정 구간 끝 b, full_bins: 체류 안의 완전한 구간
    # available: 각 기록이 과거 0~6번째 입력으로 쓰이는 예측 시점
    # 늦게 저장된 기록을 과거 예측 입력에 소급 반영하지 않음
    con.execute("""
        CREATE OR REPLACE TABLE asof_inputs AS
        WITH bucketed AS (
            SELECT
                e.*,
                CASE
                    WHEN charttime = date_trunc('hour', charttime) THEN charttime
                    ELSE date_trunc('hour', charttime) + INTERVAL '1 hour'
                END AS b
            FROM clean_events e
            WHERE itemid = 220045
        ),
        full_bins AS (
            SELECT b.*
            FROM bucketed b
            JOIN stays s USING (stay_id)
            WHERE b.b - INTERVAL '1 hour' >= s.intime
                AND b.b <= s.outtime
        ),
        available AS (
            SELECT
                e.subject_id,
                e.stay_id,
                e.b + offsets.k * INTERVAL '1 hour' AS t,
                offsets.k,
                e.valuenum
            FROM full_bins e
            CROSS JOIN range(0, 7) AS offsets(k)
            WHERE e.storetime IS NOT NULL
                AND e.storetime <= e.b + offsets.k * INTERVAL '1 hour'
        )
        SELECT
            subject_id, stay_id, t,
            avg(valuenum) FILTER (WHERE k = 0) AS r0,
            avg(valuenum) FILTER (WHERE k = 1) AS r1,
            avg(valuenum) FILTER (WHERE k = 2) AS r2,
            avg(valuenum) FILTER (WHERE k = 3) AS r3,
            avg(valuenum) FILTER (WHERE k = 4) AS r4,
            avg(valuenum) FILTER (WHERE k = 5) AS r5,
            avg(valuenum) FILTER (WHERE k = 6) AS r6
        FROM available
        GROUP BY subject_id, stay_id, t
    """)

    # r0은 최신 구간. 모델 입력은 오래된 순서이므로 x0에 r5 배치
    feature_terms = [
        f"coalesce(a.r{lag}, a.r{lag + 1}) AS x{index}" for index, lag in enumerate(range(5, -1, -1))
    ]
    # mask=True는 해당 구간의 사용 가능한 관측값 존재를 의미
    # 이전 값으로 보완한 구간은 mask=False 유지
    mask_terms = [f"a.r{lag} IS NOT NULL AS mask{index}" for index, lag in enumerate(range(5, -1, -1))]
    input_columns = ",\n                ".join(feature_terms + mask_terms)
    complete_inputs = " AND ".join(f"x{index} IS NOT NULL" for index in range(6))

    # 입력 6개와 다음 시간 관측 정답이 모두 있는 창만 사용
    # x5는 최신 입력으로 persistence baseline에 사용
    con.execute(f"""
        CREATE OR REPLACE TABLE windows AS
        WITH lagged AS (
            SELECT
                h.subject_id, h.stay_id, h.split, h.t, h.target_time, h.target_hr,
                {input_columns},
                h.t - INTERVAL '5 hours' AS first_input_time
            FROM hourly h
            LEFT JOIN asof_inputs a USING (subject_id, stay_id, t)
            JOIN stays s ON h.stay_id = s.stay_id
            WHERE h.t - INTERVAL '6 hours' >= s.intime
        )
        SELECT
            *,
            x5 AS prediction_persistence,
            (x0 + x1 + x2 + x3 + x4 + x5) / 6.0 AS prediction_history_mean
        FROM lagged
        WHERE {complete_inputs}
            AND target_hr IS NOT NULL
            AND target_time = t + INTERVAL '1 hour'
            AND first_input_time = t - INTERVAL '5 hours'
    """)


def quality_summary(con):
    # 항목별 후보 수·제외 사유·정제 후 기록 및 체류 수 집계
    # 한 기록에 여러 제외 사유가 겹칠 수 있으므로 사유별 건수의 합은 총 제외 수와 다름
    # timestamp_excluded_records: 다른 검사를 통과했지만 저장 시각 누락·역전으로 제외된 수
    # 시간 제외 건수는 중복 제거 전 기준
    return records(
        con,
        """
        SELECT
            v.itemid,
            v.name,
            (
                SELECT count(*) FROM candidates c
                WHERE c.itemid = v.itemid
            ) AS candidate_records,
            (
                SELECT count(*) FROM assessed a
                WHERE a.itemid = v.itemid AND NOT matched_stay
            ) AS unmatched_records,
            (
                SELECT count(*) FROM assessed a
                WHERE a.itemid = v.itemid AND matched_stay AND NOT within_stay
            ) AS outside_stay_records,
            (
                SELECT count(*) FROM assessed a
                WHERE a.itemid = v.itemid AND NOT value_valid
            ) AS invalid_value_records,
            (
                SELECT count(*) FROM assessed a
                WHERE a.itemid = v.itemid AND NOT unit_valid
            ) AS unit_mismatch_records,
            (
                SELECT count(*) FROM assessed a
                WHERE a.itemid = v.itemid AND NOT warning_clear
            ) AS warning_records,
            (
                SELECT count(*) FROM assessed a
                WHERE a.itemid = v.itemid
                    AND matched_stay
                    AND within_stay
                    AND value_valid
                    AND unit_valid
                    AND warning_clear
                    AND NOT timestamp_valid
            ) AS timestamp_excluded_records,
            (
                SELECT count(*) FROM clean_events c
                WHERE c.itemid = v.itemid
            ) AS clean_records,
            (
                SELECT count(DISTINCT stay_id) FROM clean_events c
                WHERE c.itemid = v.itemid
            ) AS stays_with_clean_record
        FROM variables v
        ORDER BY v.itemid
    """,
    )


def baseline_metrics(con):
    # 같은 windows에서 분할별 baseline 오차 계산
    # persistence: 최신 입력, six_hour_mean: 입력 6개 평균
    # training_mean: 학습 정답에서만 계산한 상수. 검증·테스트 정답은 계산에 사용하지 않음
    train_mean = con.execute("SELECT avg(target_hr) FROM windows WHERE split = 'train'").fetchone()[0]
    if train_mean is None:
        raise RuntimeError("No training windows")

    con.execute(
        "CREATE OR REPLACE TABLE training_constant AS SELECT ?::DOUBLE AS value",
        [train_mean],
    )
    models = [
        ("persistence", "prediction_persistence"),
        ("six_hour_mean", "prediction_history_mean"),
        ("training_mean", "(SELECT value FROM training_constant)"),
    ]
    metrics = []

    # MAE: 절대오차 평균. RMSE: 큰 오차에 더 민감한 제곱평균제곱근 오차
    # 환자별 MAE도 별도 계산해 긴 체류 환자의 많은 창 수로 인한 영향 완화
    for model, expression in models:
        rows = records(
            con,
            f"""
            WITH errors AS (
                SELECT split, subject_id, target_hr - {expression} AS error
                FROM windows
            ),
            patients AS (
                SELECT split, subject_id, avg(abs(error)) AS patient_mae
                FROM errors
                GROUP BY split, subject_id
            )
            SELECT
                split,
                count(*) AS windows,
                count(DISTINCT subject_id) AS patients,
                avg(abs(error)) AS mae_bpm,
                sqrt(avg(error * error)) AS rmse_bpm,
                (
                    SELECT avg(patient_mae) FROM patients p
                    WHERE p.split = e.split
                ) AS patient_macro_mae_bpm
            FROM errors e
            GROUP BY split
            ORDER BY split
        """,
        )
        for row in rows:
            row["model"] = model
        metrics.extend(rows)

    return train_mean, metrics


def check_integrity(con):
    # 환자 분할 중복·시간 범위·입력·정답 규칙 검사
    # 각 쿼리는 위반 행 수 반환. 하나라도 0이 아니면 결과 저장 전 중단
    queries = {
        "patient_split_overlap": """
            SELECT count(*) FROM (
                SELECT subject_id FROM stays
                GROUP BY subject_id
                HAVING count(DISTINCT split) > 1
            )
        """,
        "window_split_overlap": """
            SELECT count(*) FROM (
                SELECT subject_id FROM windows
                GROUP BY subject_id
                HAVING count(DISTINCT split) > 1
            )
        """,
        "bad_horizon": """
            SELECT count(*) FROM windows
            WHERE target_time != t + INTERVAL '1 hour'
        """,
        "unobserved_targets": """
            SELECT count(*)
            FROM windows w
            LEFT JOIN hourly h
                ON w.stay_id = h.stay_id AND w.target_time = h.t
            WHERE h.observed_hr IS NULL OR w.target_hr != h.observed_hr
        """,
        "bad_stay_bounds": """
            SELECT count(*)
            FROM windows w
            JOIN stays s USING (stay_id)
            WHERE w.first_input_time - INTERVAL '1 hour' < s.intime
                OR w.target_time > s.outtime
        """,
        "missing_inputs": """
            SELECT count(*) FROM windows
            WHERE x0 IS NULL OR x1 IS NULL OR x2 IS NULL
                OR x3 IS NULL OR x4 IS NULL OR x5 IS NULL
        """,
        "asof_pivot_disagrees": """
            SELECT count(*)
            FROM windows w
            JOIN asof_inputs a USING (subject_id, stay_id, t)
            WHERE w.x0 IS DISTINCT FROM coalesce(a.r5, a.r6)
                OR w.x5 IS DISTINCT FROM coalesce(a.r0, a.r1)
        """,
    }
    checks = {name: con.execute(query).fetchone()[0] for name, query in queries.items()}
    if any(checks.values()):
        raise RuntimeError("Integrity assertion failed")
    return checks


def safe_summary(con):
    # 예측 오차와 데이터 품질을 보고용 집계로 구성
    # 환자·체류·시간 구간 수, 측정 간격, 저장 지연, 보완 비율 포함. 환자별 행은 포함하지 않음
    train_mean, metrics = baseline_metrics(con)
    checks = check_integrity(con)

    cohort = records(
        con,
        """
        SELECT split, count(*) AS stays, count(DISTINCT subject_id) AS patients
        FROM stays
        GROUP BY split
        ORDER BY split
    """,
    )
    hourly = records(
        con,
        """
        SELECT
            split,
            count(*) AS full_hour_bins,
            count(observed_hr) AS observed_bins,
            count(timely_hr) AS timely_bins,
            count(feature_hr) AS usable_feature_bins,
            count(*) FILTER (
                WHERE feature_hr IS NOT NULL AND timely_hr IS NULL
            ) AS filled_bins
        FROM hourly
        GROUP BY split
        ORDER BY split
    """,
    )

    # 측정 간격은 charttime 차이, 저장 지연은 storetime-charttime
    # 측정 빈도와 전산 기록 지연을 구분하기 위한 별도 집계
    intervals = records(
        con,
        """
        WITH times AS (
            SELECT DISTINCT stay_id, charttime
            FROM clean_events
            WHERE itemid = 220045
        ),
        gaps AS (
            SELECT
                epoch(
                    charttime - lag(charttime) OVER (
                        PARTITION BY stay_id ORDER BY charttime
                    )
                ) / 60 AS minutes
            FROM times
        )
        SELECT
            count(*) AS positive_intervals,
            median(minutes) AS median_minutes,
            quantile_cont(minutes, 0.25) AS p25_minutes,
            quantile_cont(minutes, 0.75) AS p75_minutes,
            avg(CASE WHEN minutes = 60 THEN 1.0 ELSE 0.0 END)
                AS fraction_exactly_60min
        FROM gaps
        WHERE minutes > 0
    """,
    )[0]
    delay = records(
        con,
        """
        SELECT
            count(*) AS clean_hr_records,
            count(*) FILTER (WHERE storetime IS NULL) AS missing_storetime,
            count(*) FILTER (WHERE storetime < charttime)
                AS storetime_before_charttime,
            count(*) FILTER (
                WHERE storetime > CASE
                    WHEN charttime = date_trunc('hour', charttime) THEN charttime
                    ELSE date_trunc('hour', charttime) + INTERVAL '1 hour'
                END
            ) AS after_bin_close
        FROM clean_events
        WHERE itemid = 220045
    """,
    )[0]
    delay_quantiles = records(
        con,
        """
        SELECT
            median(epoch(storetime - charttime) / 60) AS median_minutes,
            quantile_cont(epoch(storetime - charttime) / 60, 0.9) AS p90_minutes
        FROM clean_events
        WHERE itemid = 220045 AND storetime >= charttime
    """,
    )[0]

    # 양쪽에 같은 정제 조건 적용. 시간 제외 건수와 중복 제거 건수의 혼합 방지
    duplicate_count = con.execute("""
        SELECT
            (
                SELECT count(*) FROM assessed
                WHERE matched_stay AND within_stay
                    AND value_valid AND unit_valid AND warning_clear
                    AND timestamp_valid
            ) - (SELECT count(*) FROM clean_events)
    """).fetchone()[0]
    # 예측 가능한 시간 구간 비율과 실제 입력 슬롯의 보완 비율을 별도 집계
    window_coverage = records(
        con,
        """
        SELECT
            w.split,
            count(*) AS windows,
            count(DISTINCT w.subject_id) AS patients,
            count(DISTINCT w.stay_id) AS stays,
            sum(
                CAST(NOT mask0 AS INTEGER) + CAST(NOT mask1 AS INTEGER)
                + CAST(NOT mask2 AS INTEGER) + CAST(NOT mask3 AS INTEGER)
                + CAST(NOT mask4 AS INTEGER) + CAST(NOT mask5 AS INTEGER)
            ) AS imputed_input_slots,
            6 * count(*) AS total_input_slots
        FROM windows w
        GROUP BY w.split
        ORDER BY w.split
    """,
    )
    eligible_origins = records(
        con,
        """
        SELECT
            h.split,
            count(*) AS origins_with_six_hours_and_observed_target
        FROM hourly h
        JOIN stays s USING (stay_id)
        WHERE h.target_hr IS NOT NULL AND h.t - INTERVAL '6 hours' >= s.intime
        GROUP BY h.split
        ORDER BY h.split
    """,
    )

    return {
        "cohort_by_split": cohort,
        "quality_by_item": quality_summary(con),
        "hourly_by_split": hourly,
        "baseline_metrics": metrics,
        "integrity_checks": checks,
        "heart_rate_intervals": intervals,
        "heart_rate_recording_delay": delay,
        "duplicate_model_records_removed": duplicate_count,
        "training_target_mean_bpm": train_mean,
        "window_coverage_by_split": window_coverage,
        "eligible_origins_by_split": eligible_origins,
        "nonnegative_recording_delay": delay_quantiles,
        "input_policy": (
            "Each historical bin reconstructed using only records stored by "
            "the current prediction time; one-bin forward fill from an observed older bin."
        ),
    }


def resolve_run_paths(config):
    # 입출력 경로 확인. 원본·소스 폴더 아래에 환자별 결과를 저장하지 않음
    if config["lookback_hours"] != 6:
        raise ValueError("This version supports six-hour lookback only")

    raw = Path(config["raw_root"]).resolve()
    private = Path(config["private_root"]).resolve() / config["run_id"]
    if private.is_relative_to(ROOT) or private.is_relative_to(raw):
        raise ValueError("Restricted output must be outside source and code directories")
    return raw, private


def prepare_run(config, private):
    # 코드·설정 사본과 해시로 실행 식별
    # 같은 run_id는 코드·설정이 모두 일치할 때만 재사용. 주석 변경도 파일 해시에 반영

    source = Path(__file__).read_bytes()
    fingerprint = hashlib.sha256(source + json.dumps(config, sort_keys=True).encode()).hexdigest()
    marker = private / "run_fingerprint.txt"
    if marker.exists() and marker.read_text() != fingerprint:
        raise RuntimeError("Existing run has different code/config; use a new run_id")

    marker.write_text(fingerprint)
    (private / "pipeline_source.py").write_bytes(source)
    (private / "config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return fingerprint


def verify_sources(raw):
    # 원본 4개 파일의 해시를 배포 SHA256SUMS.txt와 대조
    manifest = {}
    for line in (raw / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            manifest[name.strip().lstrip("*")] = digest

    checks = []
    for name in SOURCE_FILES:
        path = raw / name
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()

        matches = digest == manifest.get(name)
        checks.append(
            {
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "matches_local_manifest": matches,
            }
        )
        if not matches:
            raise RuntimeError("Source checksum mismatch")

    return checks


def configure_database(con, config, private):
    # DuckDB 메모리·스레드·임시 저장 경로 설정
    # 메모리 초과 시 spill 사용. 환자 정보가 포함될 수 있어 로컬 전용 경로에 배치
    con.execute(f"SET memory_limit='{config['memory_limit']}'")
    con.execute(f"SET threads={int(config['threads'])}")
    con.execute("SET enable_progress_bar=false")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory={sql_path(private / 'spill')}")
    register_variables(con)


def build_cohort(con, raw, seed):
    # 측정 항목 사전 확인 후 성인 ICU 체류의 환자별 분할
    # subject_id: 환자, hadm_id: 입원, stay_id: ICU 체류
    # 같은 환자의 여러 입원·체류가 다른 분할에 섞이지 않도록 환자 단위 사용
    con.execute(f"""
        CREATE OR REPLACE TABLE items AS
        SELECT * FROM read_csv({sql_path(raw / "icu/d_items.csv.gz")}, header=true)
    """)
    mismatches = con.execute("""
        SELECT count(*)
        FROM variables v
        LEFT JOIN items d USING (itemid)
        WHERE d.itemid IS NULL
            OR d.linksto != 'chartevents'
            OR lower(trim(d.unitname)) != lower(v.unit)
    """).fetchone()[0]
    if mismatches:
        raise RuntimeError("Item dictionary metadata mismatch")

    con.execute(f"""
        CREATE OR REPLACE TABLE all_stays AS
        SELECT * FROM read_csv({sql_path(raw / "icu/icustays.csv.gz")}, header=true)
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE patients AS
        SELECT * FROM read_csv({sql_path(raw / "hosp/patients.csv.gz")}, header=true)
    """)
    # anchor_age와 기준 연도로 입실 나이 근사
    # 환자 번호와 seed 해시를 0~99 버킷에 배정해 약 70/15/15로 분할
    # 같은 환자의 체류는 같은 버킷 유지
    con.execute(f"""
        CREATE OR REPLACE TABLE stays AS
        WITH eligible AS (
            SELECT
                s.*,
                p.anchor_age,
                p.anchor_age + year(s.intime) - p.anchor_year AS approximate_age,
                CAST(
                    '0x' || substr(
                        sha256(CAST(s.subject_id AS VARCHAR) || ':{seed}'), 1, 8
                    ) AS UBIGINT
                ) % 100 AS bucket
            FROM all_stays s
            JOIN patients p USING (subject_id)
            WHERE s.outtime > s.intime
                AND p.anchor_age + year(s.intime) - p.anchor_year >= 18
        )
        SELECT
            *,
            CASE
                WHEN bucket < 70 THEN 'train'
                WHEN bucket < 85 THEN 'validation'
                ELSE 'test'
            END AS split
        FROM eligible
    """)
    duplicate_stays = con.execute("SELECT count(*) - count(DISTINCT stay_id) FROM stays").fetchone()[0]
    if duplicate_stays:
        raise RuntimeError("Duplicate cohort stay IDs")


def scan_candidates(con, raw, private):
    # 전체 chartevents에서 후보 측정 항목 추출
    # 코드·설정 일치 확인 후 완료된 스캔은 재사용
    completed_scan = private / "scan_complete.txt"
    if completed_scan.exists():
        return

    # CSV를 문자열로 읽은 뒤 필요한 열만 변환. 변환 실패 값은 NULL 처리
    # 이후 정제 단계에서 제외 여부 판정. CSV 파싱 오류는 숨기지 않음
    item_ids = ", ".join(repr(str(variable[0])) for variable in VARIABLES)
    con.execute(f"""
        CREATE OR REPLACE TABLE candidates AS
        SELECT
            try_cast(subject_id AS BIGINT) AS subject_id,
            try_cast(hadm_id AS BIGINT) AS hadm_id,
            try_cast(stay_id AS BIGINT) AS stay_id,
            try_cast(itemid AS INTEGER) AS itemid,
            try_cast(charttime AS TIMESTAMP) AS charttime,
            try_cast(storetime AS TIMESTAMP) AS storetime,
            try_cast(valuenum AS DOUBLE) AS valuenum,
            valueuom,
            try_cast(warning AS INTEGER) AS warning
        FROM read_csv(
            {sql_path(raw / "icu/chartevents.csv.gz")},
            header=true, all_varchar=true, ignore_errors=false
        )
        WHERE itemid IN ({item_ids})
    """)
    # DB 반영 후 스캔 완료 표식 기록
    con.execute("CHECKPOINT")
    completed_scan.write_text("Complete scan succeeded; not a sample")


def export_datasets(con, private):
    # 분할별 환자 표를 Parquet로 내보내고 행 수 확인
    # Parquet는 필요한 열만 읽을 수 있는 표 형식. 실제 기록은 private 경로에만 저장
    for table in ("stays", "clean_events", "hourly", "windows"):
        for split in SPLITS:
            path = private / f"{table}_{split}.parquet"
            con.execute(f"""
                COPY (SELECT * FROM {table} WHERE split = '{split}')
                TO {sql_path(path)} (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
            exported = con.execute(f"SELECT count(*) FROM read_parquet({sql_path(path)})").fetchone()[0]
            expected = con.execute(f"SELECT count(*) FROM {table} WHERE split = '{split}'").fetchone()[0]
            if expected != exported:
                raise RuntimeError("Export count mismatch")


def save_summary(con, summary, config, private, fingerprint, checks, start):
    # 집계에 실행 설정·입력 체크섬·버전을 추가해 재현 근거 기록
    # 내보내기 확인 후 SUCCESS.txt로 실행 완료 표시
    excluded_stays = con.execute("""
        SELECT (SELECT count(*) FROM all_stays) - (SELECT count(*) FROM stays)
    """).fetchone()[0]
    summary.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": config["run_id"],
            "config": config,
            "code_config_sha256": fingerprint,
            "input_checksums": checks,
            "full_chartevents_scan": True,
            "export_row_counts_match": True,
            "cohort_excluded_stays": excluded_stays,
            "elapsed_seconds": round(time.time() - start, 1),
            "versions": {
                package: importlib.metadata.version(package)
                for package in ("duckdb", "numpy", "pandas", "pyarrow")
            },
            "python": sys.version.split()[0],
            "restricted_output_root": str(private),
            "test_metrics_status": (
                "preliminary descriptive run; no tuning performed; test set has now been inspected"
            ),
        }
    )

    # 공개 집계에 환자·체류 식별자와 실제 측정 시각은 포함하지 않음
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    (private / "aggregate_summary.json").write_text(payload, encoding="utf-8")
    (reports / "aggregate_summary.json").write_text(payload, encoding="utf-8")
    (private / "SUCCESS.txt").write_text("All integrity and export checks passed.")


def save_private_error(private, exc):
    # 상세 오류는 로컬 로그, 화면에는 오류 종류만 출력
    # DB 오류에 실제 행 값이 포함될 수 있으므로 요약에는 코드 위치만 기록
    (private / "error_private.log").write_text(traceback.format_exc(), encoding="utf-8")
    safe_error = {
        "exception_class": type(exc).__name__,
        "frames": [
            {
                "file": Path(frame.filename).name,
                "line": frame.lineno,
                "function": frame.name,
            }
            for frame in traceback.extract_tb(exc.__traceback__)
        ],
    }
    (private / "error_safe.json").write_text(json.dumps(safe_error, indent=2), encoding="utf-8")
    print(
        f"FAILED; exception class: {type(exc).__name__}. Detailed error retained locally.",
        flush=True,
    )


def main():
    # 설정 확인 → 원본 검사 → 전처리 → 평가 → 결과 저장
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/pipeline.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw, private = resolve_run_paths(config)
    private.mkdir(parents=True, exist_ok=True)

    # 연결 전 실패도 처리할 수 있도록 초기값 None 사용. finally에서 열린 DB 연결 해제
    start = time.time()
    con = None
    try:
        fingerprint = prepare_run(config, private)

        print("1/7 Verifying source checksums (no patient output)", flush=True)
        checks = verify_sources(raw)
        con = duckdb.connect(str(private / "pipeline.duckdb"))
        configure_database(con, config, private)

        print("2/7 Reading metadata and constructing patient-level splits", flush=True)
        build_cohort(con, raw, config["seed"])

        print(
            "3/7 Scanning complete compressed chartevents; this is the long step",
            flush=True,
        )
        scan_candidates(con, raw, private)

        print(
            "4/7 Filtering measurements and constructing hourly heart-rate data",
            flush=True,
        )
        build_clean_events(con)
        build_hourly(con)

        print("5/7 Constructing six-hour windows and evaluating fixed baselines", flush=True)
        build_windows(con)
        summary = safe_summary(con)

        print("6/7 Exporting local restricted datasets and validation artifacts", flush=True)
        export_datasets(con, private)
        save_summary(con, summary, config, private, fingerprint, checks, start)
        print(
            "7/7 SUCCESS. Aggregate summary saved; patient-level outputs remain local.",
            flush=True,
        )
    except Exception as exc:
        save_private_error(private, exc)
        return 1
    finally:
        if con is not None:
            con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
