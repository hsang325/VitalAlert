# 저장된 시험 예측과 전처리 입력을 대조해 실제 시연 사례 한 개 준비
# main()에서 해시 확인 → 연속 구간 선택 → 입력·정답 대조 → LSTM 재추론 → 로컬 저장
# 사례에는 별칭과 상대 시간만 사용하지만 환자 파생 자료이므로 프로젝트 밖에 보관

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Predictor

# 전체 시험 창의 GPU 재현은 저장값과 일치, CPU 최대 차이는 0.035 bpm 미만으로 확인
# 화면의 0.1 bpm 표시 단위보다 작은 장치 차이 허용값. 임상 정확도 기준이 아님
REPLAY_TOLERANCE_BPM = 0.05


def file_hash(path):
    # 기존 평가 때 사용한 모델과 시험 파일이 그대로인지 확인하는 파일 지문
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def choose_segment(frame):
    # 환자·체류·시간 순으로 첫 12시간 이상 연속 구간 선택, 최대 24개 창 사용
    # 예측 오차나 경보 결과를 선택 조건으로 사용하지 않음. 전체 환자의 대표 표본은 아님
    ordered = frame.sort_values(["subject_id", "stay_id", "t"]).reset_index(drop=True)
    changed = (
        ordered["subject_id"].ne(ordered["subject_id"].shift())
        | ordered["stay_id"].ne(ordered["stay_id"].shift())
        | ordered["t"].diff().ne(pd.Timedelta(hours=1))
    )
    segments = changed.cumsum()
    lengths = segments.groupby(segments).transform("size")
    eligible = ordered.index[lengths >= 12]
    if len(eligible) == 0:
        raise ValueError("No continuous test segment with at least 12 windows")
    segment = segments.iloc[eligible[0]]
    return ordered.loc[segments == segment].head(24).copy()


def verify_and_export(args):
    # 정답·mask·6개 입력을 원래 시험 Parquet와 다시 대조하고 기존 평가 모델로 추론
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Use a new private output directory outside the project")
    summary = json.loads((args.evaluation_dir / "test_summary.json").read_text(encoding="utf-8"))
    predictions_path = args.evaluation_dir / "test_predictions.parquet"
    hashes = {
        "checkpoint": file_hash(args.checkpoint),
        "predictions": file_hash(predictions_path),
        "test_inputs": file_hash(args.test_windows),
    }
    if (
        hashes["checkpoint"] != summary["checkpoint_sha256"]
        or hashes["predictions"] != summary["prediction_file_sha256"]
        or hashes["test_inputs"] != summary["test_input_sha256"]
    ):
        raise ValueError("Files differ from the recorded test evaluation")
    frame = pq.read_table(predictions_path).to_pandas()
    selected = choose_segment(frame)
    subject = int(selected.iloc[0]["subject_id"])
    stay = int(selected.iloc[0]["stay_id"])
    original = pq.read_table(args.test_windows, filters=[("subject_id", "=", subject), ("stay_id", "=", stay)]).to_pandas()
    original = original.set_index("t").loc[selected["t"]]
    columns = ["target_hr"] + [f"x{i}" for i in range(6)] + [f"mask{i}" for i in range(6)]
    if not np.array_equal(selected[columns].to_numpy(), original[columns].to_numpy()):
        raise ValueError("Selected inputs or targets differ from original test windows")
    if not np.array_equal(selected["target_time"].to_numpy(), original["target_time"].to_numpy()):
        raise ValueError("Selected target intervals differ")
    predictor = Predictor(args.checkpoint, "cpu")
    start = selected.iloc[0]["t"]
    rows = []
    for _, row in selected.iterrows():
        values = [float(row[f"x{i}"]) for i in range(6)]
        masks = [int(row[f"mask{i}"]) for i in range(6)]
        prediction = predictor.predict(values, masks)
        baseline = float(np.mean(values))
        if abs(prediction - float(row["predicted_lstm_bpm"])) > REPLAY_TOLERANCE_BPM:
            raise ValueError("LSTM replay exceeds the verified device tolerance")
        if abs(baseline - float(row["predicted_six_hour_mean_bpm"])) > 1e-8:
            raise ValueError("Baseline replay differs")
        rows.append({
            "hour": float((row["t"] - start).total_seconds() / 3600),
            "inputs": values, "masks": masks, "actual": float(row["target_hr"]),
            "prediction": prediction, "baseline": baseline,
            "evaluation_prediction": float(row["predicted_lstm_bpm"]),
        })
    policy = json.loads((ROOT / "configs/alert.json").read_text(encoding="utf-8"))
    bundle = {
        "source": "local_mimic", "policy": policy,
        "checkpoint_sha256": hashes["checkpoint"],
        "cases": [{
            "id": "test_demo", "label": "실제 시험 사례 1 · 로컬 전용",
            "source": "local_mimic", "anchor": 0, "rows": rows,
        }],
    }
    selection_rule = "First segment in subject/stay/time order with >=12 consecutive hourly windows; first <=24 windows; selection does not use errors or alerts."
    output.mkdir(parents=True)
    (output / "cases_private.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {
        "subject_id": subject, "stay_id": stay, "start_time": str(start),
        "evaluation_id": summary["evaluation_id"], "selection_rule": selection_rule,
        "source_hashes": hashes,
    }
    (output / "provenance_private.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    # 공개 검증 기록에는 통과 여부·파일 해시·개수만 기록. 환자별 값이나 오차는 포함하지 않음
    verification = {
        "status": "passed", "case_count": 1, "window_count": len(rows),
        "source_hashes": hashes, "selection_rule": selection_rule,
        "test_inputs_and_targets_match": True, "lstm_replay_within_tolerance": True,
        "replay_tolerance_bpm": REPLAY_TOLERANCE_BPM,
        "tolerance_basis": "Full-test CPU versus stored GPU comparison: maximum difference <0.035 bpm; GPU replay exact; UI display step 0.1 bpm. Not a clinical tolerance.",
        "baseline_replay_matches": True, "retraining_performed": False,
    }
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    return verification


def main():
    parser = argparse.ArgumentParser(description="Prepare one locally verified retrospective demo case")
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-windows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        verification = verify_and_export(args)
    except Exception as error:
        # 처리 중인 환자 행을 예외 메시지로 노출하지 않음
        location = traceback.extract_tb(error.__traceback__)[-1]
        parser.exit(1, f"Case preparation failed ({type(error).__name__}, {Path(location.filename).name}:{location.lineno}). Check source files and output location locally.\n")
    print(json.dumps({"status": verification["status"], "case_count": verification["case_count"], "window_count": verification["window_count"]}))


if __name__ == "__main__":
    main()
