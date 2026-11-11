# 검증으로 선택한 고정 LSTM과 baseline 3종의 테스트 성능 평가
# main() → verify_frozen_selection() → evaluate_test()
# 학습 결과·데이터 확인 후 추론만 수행. 정규화 재학습·모델 재선택 없음
# 환자별 예측은 로컬 경로, 집계는 reports/에 저장

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import traceback

# CUDA 재현성 환경 설정이 먼저 적용되도록 학습 모듈을 torch보다 앞서 로드
from train_lstm import (
    HeartRateLSTM,
    INPUT_COLUMNS,
    MASK_COLUMNS,
    ROOT,
    file_hash,
    predict_bpm,
    regression_metrics,
    set_reproducibility,
    transform_inputs,
    write_json,
)
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


METRICS = ("mae_bpm", "rmse_bpm", "patient_macro_mae_bpm")


def verify_frozen_selection(manifest_path, config_path, pipeline_config):
    # manifest: 테스트 평가 전에 고정한 모델·설정 기록
    # 내용 해시로 학습 당시 가중치와 설정의 일치 여부 확인
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if (
        not manifest["selection_is_frozen"]
        or not manifest["verification_checks"]
        or not all(manifest["verification_checks"].values())
    ):
        raise ValueError("Selection is not verified and frozen")
    if config["test_patient_data_allowed"] or config["selection_metric"] != "validation_mae_bpm":
        raise ValueError("Training selection policy changed")
    if manifest["model_run_id"] != config["model_run_id"]:
        raise ValueError("Model run mismatch")
    if manifest["data_run_id"] != config["data_run_id"] or config["data_run_id"] != pipeline_config["run_id"]:
        raise ValueError("Data run mismatch")

    model_directory = Path(config["private_model_root"]) / config["model_run_id"]
    checkpoint_path = model_directory / "selected.pt"
    protected_hashes = {
        manifest_path: file_hash(manifest_path),
        checkpoint_path: manifest["checkpoint_sha256"],
        model_directory / "scaler.json": manifest["scaler_sha256"],
        ROOT / "scripts/train_lstm.py": manifest["training_source_sha256"],
        model_directory / "train_lstm_source.py": manifest["training_source_sha256"],
        Path(config_path): manifest["training_config_sha256"],
    }
    for path, expected in protected_hashes.items():
        if file_hash(path) != expected:
            raise ValueError("Frozen artifact hash mismatch")
    if not (model_directory / "SUCCESS.txt").is_file():
        raise ValueError("Training is incomplete")

    # 전처리 실행과 학습 입력 파일의 일치 여부 확인. 환자 행은 출력하지 않음
    data_directory = Path(pipeline_config["private_root"]) / config["data_run_id"]
    if not (data_directory / "SUCCESS.txt").is_file():
        raise ValueError("Preprocessing is incomplete")
    summary_path = data_directory / "aggregate_summary.json"
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    training_path = model_directory / "training_summary.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if (
        source["run_id"] != config["data_run_id"]
        or source["code_config_sha256"] != manifest["preprocessing_fingerprint"]
    ):
        raise ValueError("Preprocessing provenance mismatch")
    if (
        training["selected_candidate"] != manifest["selected_candidate"]
        or training["selected_epoch"] != manifest["epoch"]
    ):
        raise ValueError("Selected candidate mismatch")
    if training["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise ValueError("Training checkpoint mismatch")
    if set(training["input_file_sha256"]) != {"train", "validation"}:
        raise ValueError("Training input provenance is incomplete")
    for split, expected in training["input_file_sha256"].items():
        path = data_directory / f"windows_{split}.parquet"
        if file_hash(path) != expected:
            raise ValueError("Training input hash mismatch")
        protected_hashes[path] = expected
    protected_hashes[summary_path] = file_hash(summary_path)
    protected_hashes[training_path] = file_hash(training_path)

    scaler = json.loads((model_directory / "scaler.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    candidate = next(row for row in config["candidates"] if row["name"] == manifest["selected_candidate"])
    if (
        checkpoint["epoch"] != manifest["epoch"]
        or checkpoint["data_run_id"] != config["data_run_id"]
        or checkpoint["hidden_size"] != candidate["hidden_size"]
        or checkpoint["input_features"] != config["input_features"]
        or checkpoint["prediction_head"] != config["prediction_head"]
    ):
        raise ValueError("Checkpoint metadata mismatch")
    if scaler != checkpoint["scaler"] or scaler != training["scaler"] or scaler["fitted_split"] != "train":
        raise ValueError("Frozen training scaler mismatch")
    if not np.isfinite([scaler["mean_bpm"], scaler["scale_bpm"]]).all() or scaler["scale_bpm"] <= 0:
        raise ValueError("Invalid frozen scaler")
    return config, manifest, source, checkpoint, data_directory, protected_hashes


def baseline_predictions(values, training_target_mean):
    # 학습 정답 평균은 전처리에서 계산한 상수 사용. 테스트 정답으로 재계산하지 않음
    # 최신값 유지·6시간 평균도 LSTM과 동일한 입력 6개 사용
    if not np.isfinite(training_target_mean):
        raise ValueError("Invalid training target mean")
    return {
        "persistence": values[:, -1],
        "six_hour_mean": values.mean(axis=1),
        "training_mean": np.full(len(values), training_target_mean, dtype=np.float64),
    }


def infer_test_table(table, checkpoint, device, batch_size, training_target_mean):
    # subject_id: 환자별 평균 오차 집계용 식별자. 모델 입력에는 포함하지 않음
    # mask: 실제 관측 1, 이전 구간으로 보완한 값 0. 학습 때의 정규화 적용
    values = np.column_stack([table[name].to_numpy() for name in INPUT_COLUMNS])
    masks = np.column_stack([table[name].to_numpy() for name in MASK_COLUMNS])
    targets = table["target_hr"].to_numpy().astype(np.float64)
    subjects = table["subject_id"].to_numpy()
    if not len(targets) or not np.isfinite(targets).all() or table["subject_id"].null_count:
        raise ValueError("Invalid test labels or patients")
    features = torch.from_numpy(transform_inputs(values, masks, checkpoint["scaler"])).to(device)
    model = HeartRateLSTM(checkpoint["hidden_size"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.requires_grad_(False)

    predictions = baseline_predictions(values, training_target_mean)
    predictions["lstm"] = predict_bpm(model, features, checkpoint["scaler"], batch_size)
    metrics = {
        name: regression_metrics(targets, prediction, subjects) for name, prediction in predictions.items()
    }
    return predictions, metrics


def evaluate_test(verified, output, evaluation_id):
    # 선택된 모델 하나로 전체 테스트 추론. 다른 후보를 다시 비교하지 않음
    config, manifest, source, checkpoint, data_directory, protected_hashes = verified
    test_path = data_directory / "windows_test.parquet"
    test_hash = file_hash(test_path)
    protected_hashes[test_path] = test_hash
    columns = ["subject_id", "stay_id", "t", "target_time", "target_hr"] + INPUT_COLUMNS + MASK_COLUMNS
    table = pq.read_table(test_path, columns=columns)
    coverage = next(row for row in source["window_coverage_by_split"] if row["split"] == "test")
    test_subjects = table["subject_id"].to_numpy()
    if len(table) != coverage["windows"] or len(np.unique(test_subjects)) != coverage["patients"]:
        raise ValueError("Test coverage mismatch")

    # 학습·검증 환자와 테스트 환자의 중복 확인. 식별자 열만 조회
    for split in ("train", "validation"):
        other = pq.read_table(data_directory / f"windows_{split}.parquet", columns=["subject_id"])
        if np.intersect1d(test_subjects, other["subject_id"].to_numpy()).size:
            raise ValueError("Test patients overlap development patients")
    predictions, metrics = infer_test_table(
        table,
        checkpoint,
        config["device"],
        config["evaluation_batch_size"],
        source["training_target_mean_bpm"],
    )

    # 동일한 입력·정답인지 확인하기 위해 전처리의 baseline 지표와 대조
    for name in ("persistence", "six_hour_mean", "training_mean"):
        expected = next(
            row for row in source["baseline_metrics"] if row["split"] == "test" and row["model"] == name
        )
        if any(abs(metrics[name][metric] - expected[metric]) > 1e-8 for metric in METRICS):
            raise ValueError("Test baseline replay mismatch")
    for path, expected in protected_hashes.items():
        if file_hash(path) != expected:
            raise ValueError("Protected artifact changed during evaluation")

    # 사례별 예측은 로컬 시계열 분석용 Parquet로 저장. 공개 집계에는 포함하지 않음
    for name, prediction in predictions.items():
        table = table.append_column(f"predicted_{name}_bpm", pa.array(prediction))
    prediction_path = output / "test_predictions.parquet"
    pq.write_table(table, prediction_path, compression="zstd")
    restored = pq.read_table(prediction_path)
    if not restored.equals(table):
        raise ValueError("Prediction export roundtrip mismatch")

    # 비교 기준은 검증에서 선택한 6시간 평균. 테스트 성능으로 다시 선택하지 않음
    reference = metrics["six_hour_mean"]["mae_bpm"]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_id": evaluation_id,
        "model_run_id": config["model_run_id"],
        "data_run_id": config["data_run_id"],
        "split": "test",
        "selected_candidate": manifest["selected_candidate"],
        "selected_epoch": manifest["epoch"],
        "metrics": metrics,
        "mae_reduction_vs_validation_selected_baseline_percent": 100
        * (reference - metrics["lstm"]["mae_bpm"])
        / reference,
        "comparison_reference": "six_hour_mean",
        "model_selection_split": "validation",
        "training_or_tuning_performed": False,
        "scaler_refitted": False,
        "patient_overlap_with_train_or_validation": 0,
        "baseline_replay_verified": True,
        "prediction_export_roundtrip_verified": True,
        "protected_artifacts_unchanged": True,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "scaler_sha256": manifest["scaler_sha256"],
        "preprocessing_fingerprint": manifest["preprocessing_fingerprint"],
        "test_input_sha256": test_hash,
        "prediction_file_sha256": file_hash(prediction_path),
        "evaluation_source_sha256": file_hash(Path(__file__)),
        "protected_file_sha256": {str(path): digest for path, digest in protected_hashes.items()},
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "pyarrow": pa.__version__},
        "device": config["device"],
        "limitations": [
            "Historical baseline test results were inspected before this frozen LSTM evaluation; this is not an untouched test.",
            "One seed and two validation-selected candidates; no confidence intervals or external validation.",
            "LSTM uses observation masks in addition to HR; baselines use HR alone.",
            "Hourly-mean regression errors do not establish event detection, clinical alert safety or utility.",
        ],
    }
    write_json(output / "test_summary.json", report)
    return report


def main():
    # 새 평가 ID로 결과 폴더 생성. 기존 실행 결과는 덮어쓰지 않음
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "reports/lstm-v1/selection_manifest.json"))
    parser.add_argument("--config", default=str(ROOT / "configs/lstm.json"))
    parser.add_argument("--evaluation-id", default="test-v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.evaluation_id):
        raise ValueError("Invalid evaluation ID")
    pipeline_config = json.loads((ROOT / "configs/pipeline.json").read_text(encoding="utf-8"))
    verified = verify_frozen_selection(args.manifest, args.config, pipeline_config)
    config = verified[0]
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    output = (Path(config["private_model_root"]) / config["model_run_id"] / args.evaluation_id).resolve()
    if output.is_relative_to(ROOT) or output.is_relative_to(Path(pipeline_config["raw_root"]).resolve()):
        raise ValueError("Patient predictions must remain outside source and raw directories")
    report_directory = ROOT / "reports" / args.evaluation_id
    if output.exists() or report_directory.exists():
        raise FileExistsError("Evaluation ID already exists; inspect the existing result")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "evaluate_lstm_source.py")
    shutil.copy2(args.manifest, output / "selection_manifest_snapshot.json")
    try:
        set_reproducibility(config["seed"], config["cpu_threads"])
        report = evaluate_test(verified, output, args.evaluation_id)
        report_directory.mkdir(parents=True, exist_ok=False)
        write_json(report_directory / "test_summary.json", report)
        shutil.copy2(args.manifest, report_directory / "selection_manifest_snapshot.json")
        (output / "SUCCESS.txt").write_text(
            "Frozen-model test evaluation and baseline replay passed.\n", encoding="utf-8"
        )
        print(json.dumps({"evaluation_id": args.evaluation_id, "metrics": report["metrics"]}), flush=True)
    except Exception as exc:
        (output / "error_private.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Evaluation failed; exception class: {type(exc).__name__}. Details retained locally.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
