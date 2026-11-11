# 저장된 테스트 예측 → 표시 규칙 평가 → 로컬 시계열 사례
# 진입점 main(). build_review()에서 집계, select_cases()에서 사례 선택
# 집계 JSON은 reports/, 실제 사례·그래프·출처는 로컬 전용 경로에 저장
# 표시 기준은 사후 시연 규칙. 모델 재학습·재선택 없음

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from alerting import alert_metrics, classify, validate_policy


# 저장된 예측만 사용. 그래프 실행 환경에 PyTorch를 불러오지 않음
ROOT = Path(__file__).resolve().parents[1]
INPUT_COLUMNS = [f"x{index}" for index in range(6)]
MASK_COLUMNS = [f"mask{index}" for index in range(6)]


def file_hash(path):
    # 평가 파일 식별용 해시 계산. 환자 행 출력 없음
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    # UTF-8 JSON 저장. 유효한 결과로 오인하지 않도록 NaN 거부
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def case_payload(frame, case_id, label, anchor_position, source):
    # 식별자·달력 시각 대신 사례 별칭과 첫 예측 기준 상대 시간 사용
    # 상대 시간으로 변환한 실제 사례도 로컬 전용 자료에 해당
    first_time = frame.iloc[0]["t"]
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "hour": float((row["t"] - first_time).total_seconds() / 3600),
                "inputs": [float(row[name]) for name in INPUT_COLUMNS],
                "masks": [int(row[name]) for name in MASK_COLUMNS],
                "actual": float(row["target_hr"]),
                "prediction": float(row["predicted_lstm_bpm"]),
                "baseline": float(row["predicted_six_hour_mean_bpm"]),
            }
        )
    return {"id": case_id, "label": label, "source": source, "anchor": anchor_position, "rows": rows}


def select_cases(frame, policy):
    # 체류 변경 또는 연속 1시간 간격이 끊기는 지점에서 구간 분리
    # 일치·누락·과다 표시·큰 오차 설명용 목적 표집. 전체 성능의 대표 표본은 아님
    frame = frame.sort_values(["stay_id", "t"]).reset_index(drop=True).copy()
    new_segment = frame["stay_id"].ne(frame["stay_id"].shift()) | frame["t"].diff().ne(pd.Timedelta(hours=1))
    frame["segment"] = new_segment.cumsum()
    lengths = frame.groupby("segment")["segment"].transform("size")
    eligible = lengths >= policy["minimum_segment_windows"]
    actual = classify(frame["target_hr"].to_numpy(), policy) != 0
    predicted = classify(frame["predicted_lstm_bpm"].to_numpy(), policy) != 0
    definitions = [
        ("within", "범위 안 일치 중심", ~actual & ~predicted),
        ("matched", "범위 밖 표시 일치 중심", actual & predicted),
        ("missed", "범위 밖 표시 누락 중심", actual & ~predicted),
        ("extra", "범위 밖 과다 표시 중심", ~actual & predicted),
        ("large_error", "절대오차 최대 중심", np.ones(len(frame), dtype=bool)),
    ]
    cases = []
    provenance = []
    for key, label, condition in definitions:
        positions = np.flatnonzero(eligible.to_numpy() & condition)
        if not len(positions):
            continue
        if key == "large_error":
            errors = np.abs(frame.iloc[positions]["target_hr"] - frame.iloc[positions]["predicted_lstm_bpm"])
            anchor = int(positions[int(np.argmax(errors.to_numpy()))])
        else:
            anchor = int(positions[0])
        segment_positions = np.flatnonzero(frame["segment"].to_numpy() == frame.iloc[anchor]["segment"])
        start = max(int(segment_positions[0]), anchor - policy["maximum_case_windows"] // 2)
        end = min(int(segment_positions[-1]) + 1, start + policy["maximum_case_windows"])
        selected = frame.iloc[start:end]
        cases.append(case_payload(selected, key, label, anchor - start, "local_mimic"))
        provenance.append(
            {
                "case_id": key,
                "subject_id": int(frame.iloc[anchor]["subject_id"]),
                "stay_id": int(frame.iloc[anchor]["stay_id"]),
                "anchor_time": str(frame.iloc[anchor]["t"]),
            }
        )
    return cases, provenance


def render_private_cases(cases, policy, output):
    # 사례별 실제값·LSTM·baseline 곡선을 로컬 PNG로 저장
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for case in cases:
        hours = [row["hour"] + 1 for row in case["rows"]]
        figure, axis = plt.subplots(figsize=(10, 4), layout="constrained")
        axis.axhspan(policy["lower_bpm"], policy["upper_bpm"], alpha=0.06, color="green")
        for key, label, style in (
            ("actual", "Observed hourly mean", "-"),
            ("prediction", "LSTM", "--"),
            ("baseline", "6-hour mean", ":"),
        ):
            axis.plot(hours, [row[key] for row in case["rows"]], style, label=label)
        for bound in (policy["lower_bpm"], policy["upper_bpm"]):
            axis.axhline(bound, color="#bd6841", linewidth=1, linestyle=":")
        axis.set(
            xlabel="Target hour end relative to first prediction (hours)",
            ylabel="Heart rate (bpm)",
            title=f"LOCAL ONLY / {case['id']} / retrospective illustration",
        )
        axis.legend()
        figure.savefig(output / f"case_{case['id']}.png", dpi=160)
        plt.close(figure)


def build_review(policy, private_evaluation, public_evaluation, output):
    # 저장된 예측 파일 확인 후 전체 창에 같은 표시 규칙 적용
    evaluation = json.loads((public_evaluation / "test_summary.json").read_text(encoding="utf-8"))
    prediction_path = private_evaluation / "test_predictions.parquet"
    if file_hash(prediction_path) != evaluation["prediction_file_sha256"]:
        raise ValueError("Frozen prediction export changed")
    frame = pq.read_table(prediction_path).to_pandas()
    results = {
        name: alert_metrics(frame["target_hr"].to_numpy(), frame[f"predicted_{name}_bpm"].to_numpy(), policy)
        for name in ("persistence", "six_hour_mean", "training_mean", "lstm")
    }
    cases, provenance = select_cases(frame, policy)
    write_json(output / "cases_private.json", {"cases": cases, "policy": policy, "source": "local_mimic"})
    write_json(output / "case_provenance_private.json", provenance)
    render_private_cases(cases, policy, output)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": policy["run_id"],
        "evaluation_id": policy["evaluation_id"],
        "policy": policy,
        "metrics": results,
        "private_case_count": len(cases),
        "prediction_file_sha256": evaluation["prediction_file_sha256"],
        "source_sha256": {
            name: file_hash(ROOT / "scripts" / name) for name in ("alerting.py", "prepare_alert_review.py")
        },
        "case_selection": "First eligible row per category in stay/time order; maximum absolute error for large_error; contiguous segments >=12 windows, at most 24 displayed.",
        "limitations": [
            "Post-test exploratory engineering rule; thresholds are not clinical or optimized.",
            "Window-level labels of next-hour means, not diagnoses or independent alert events.",
            "Overlapping windows; no confidence intervals or patient-level alert burden estimate.",
            "Purpose-selected illustrations are not representative performance estimates.",
        ],
    }


def main():
    policy = json.loads((ROOT / "configs/alert.json").read_text(encoding="utf-8"))
    validate_policy(policy)
    model_config = json.loads((ROOT / "configs/lstm.json").read_text(encoding="utf-8"))
    private_model = Path(model_config["private_model_root"]) / model_config["model_run_id"]
    output = private_model / policy["run_id"]
    public = ROOT / "reports" / policy["run_id"]
    if output.exists() or public.exists():
        raise FileExistsError("Review run already exists")
    if output.resolve().is_relative_to(ROOT):
        raise ValueError("Actual cases require a private directory")
    output.mkdir(parents=True)
    shutil.copy2(ROOT / "configs/alert.json", output / "alert_config_snapshot.json")
    report = build_review(
        policy, private_model / policy["evaluation_id"], ROOT / "reports" / policy["evaluation_id"], output
    )
    public.mkdir(parents=True)
    write_json(public / "alert_summary.json", report)
    write_json(output / "alert_summary.json", report)
    (output / "SUCCESS.txt").write_text("Exploratory alert review complete.\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": policy["run_id"],
                "metrics": report["metrics"]["lstm"],
                "private_cases": report["private_case_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
