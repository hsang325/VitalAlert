# 전처리된 6시간 입력으로 LSTM 학습 후 검증 MAE로 모델 선택
# main() → load_split() → fit_scaler() → train_candidate()
# 학습·검증 파일만 사용. 테스트 파일 접근 차단
# 입력: 심박수와 관측 여부(mask). 출력: 최신 심박수에 더할 보정값
# 정규화는 학습 입력에서만 계산. 모델·개인별 배열은 로컬 경로에 보관

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import shutil
import time
import traceback

# CUDA 초기화 전에 결정적 행렬 연산 설정 적용
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
INPUT_COLUMNS = [f"x{index}" for index in range(6)]
MASK_COLUMNS = [f"mask{index}" for index in range(6)]


def file_hash(path):
    # 실행에 사용한 코드·데이터 파일 식별용 SHA-256
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    # UTF-8 집계 JSON 저장. NaN은 유효한 결과로 저장하지 않음
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_split(data_directory, split):
    # 학습·검증 Parquet에서 입력·mask·정답·환자 번호 조회. 테스트 분할은 허용하지 않음
    if split not in ("train", "validation"):
        raise ValueError("Training only permits train and validation splits")
    path = Path(data_directory) / f"windows_{split}.parquet"
    columns = INPUT_COLUMNS + MASK_COLUMNS + ["target_hr", "subject_id"]
    table = pq.read_table(path, columns=columns)
    values = np.column_stack([table[name].to_numpy() for name in INPUT_COLUMNS])
    masks = np.column_stack([table[name].to_numpy() for name in MASK_COLUMNS])
    targets = table["target_hr"].to_numpy().astype(np.float64)
    subjects = table["subject_id"].to_numpy()
    if len(targets) == 0 or not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("Empty or nonfinite prediction data")
    if not np.isin(masks, [False, True]).all():
        raise ValueError("Invalid observation masks")
    return {"values": values, "masks": masks, "targets": targets, "subjects": subjects}


def fit_scaler(values, split="train"):
    # 동일한 bpm 단위인 입력 6개에 공통 평균·표준편차 적용
    # 겹치는 창의 모든 학습 입력 슬롯 사용. 환자별 균등 가중 없음
    if split != "train":
        raise ValueError("Scaler fitting is restricted to training data")
    if values.ndim != 2 or values.shape[1] != 6 or not values.size or not np.isfinite(values).all():
        raise ValueError("Scaler requires finite six-hour training inputs")
    mean = float(np.mean(values, dtype=np.float64))
    standard_deviation = float(np.std(values, dtype=np.float64))
    return {
        "mean_bpm": mean,
        "scale_bpm": standard_deviation if standard_deviation > 1e-12 else 1.0,
        "fitted_split": "train",
        "input_slots": int(values.size),
        "ddof": 0,
    }


def transform_inputs(values, masks, scaler):
    # 채널 0: 표준화한 심박수, 채널 1: 관측 1/보완 0
    # 보완값도 동일하게 표준화하고 mask는 그대로 유지
    if values.shape != masks.shape or values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected matching six-hour inputs and masks")
    if not np.isfinite(values).all() or not np.isin(masks, [0, 1]).all():
        raise ValueError("Invalid model inputs")
    standardized = (values - scaler["mean_bpm"]) / scaler["scale_bpm"]
    return np.stack([standardized, masks], axis=-1).astype(np.float32)


class HeartRateLSTM(nn.Module):
    # 최신 심박수의 보정량을 학습하는 LSTM과 선형 출력층
    # 입력 모양: (사례 수, 과거 6구간, 심박수·mask 2채널)
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.correction = nn.Linear(hidden_size, 1)
        # 초기 예측은 최신값 유지와 동일하도록 보정층을 0으로 초기화
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, inputs):
        _, (hidden, _) = self.lstm(inputs)
        adjustment = self.correction(hidden[-1]).squeeze(-1)
        return inputs[:, -1, 0] + adjustment


def set_reproducibility(seed, cpu_threads):
    # seed와 결정적 연산 설정. 다른 버전·장치 간 완전한 수치 일치는 보장하지 않음
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def predict_bpm(model, features, scaler, batch_size):
    # 모델 출력을 bpm으로 역변환해 반환. 파일 출력 없음
    model.eval()
    predictions = torch.empty(len(features), device=features.device)
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = start + batch_size
            predictions[start:end] = model(features[start:end])
    standardized = predictions.cpu().numpy().astype(np.float64)
    return standardized * scaler["scale_bpm"] + scaler["mean_bpm"]


def regression_metrics(targets, predictions, subjects):
    # 전체 창의 MAE/RMSE와 환자별 평균 MAE 계산
    if len(targets) == 0 or targets.shape != predictions.shape or not np.isfinite(predictions).all():
        raise ValueError("Invalid regression predictions")
    errors = predictions - targets
    absolute = np.abs(errors)
    _, patient_index = np.unique(subjects, return_inverse=True)
    patient_sums = np.bincount(patient_index, weights=absolute)
    patient_counts = np.bincount(patient_index)
    return {
        "windows": len(targets),
        "patients": len(patient_counts),
        "mae_bpm": float(absolute.mean()),
        "rmse_bpm": float(np.sqrt(np.mean(errors * errors))),
        "patient_macro_mae_bpm": float(np.mean(patient_sums / patient_counts)),
    }


def train_candidate(
    candidate, config, train_features, train_targets, validation_features, validation, scaler, output
):
    # 후보마다 같은 seed·전체 학습 창 사용. 검증 MAE가 가장 낮은 가중치 저장
    set_reproducibility(config["seed"], config["cpu_threads"])
    model = HeartRateLSTM(candidate["hidden_size"]).to(train_features.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    checkpoint_path = output / f"{candidate['name']}.pt"
    best_mae = float("inf")
    significant_best = float("inf")
    epochs_without_improvement = 0
    history = []
    best_result = None
    for epoch in range(1, config["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        indices = torch.randperm(len(train_features), device=train_features.device)
        training_loss_sum = torch.zeros((), dtype=torch.float64, device=train_features.device)
        for batch_indices in indices.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_features[batch_indices])
            loss = torch.mean(torch.abs(prediction - train_targets[batch_indices]))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            training_loss_sum += loss.detach().double() * len(batch_indices)

        # 검증 전체를 bpm 단위로 평가. 검증 데이터의 역전파 없음
        predictions = predict_bpm(model, validation_features, scaler, config["evaluation_batch_size"])
        metrics = regression_metrics(validation["targets"], predictions, validation["subjects"])
        row = {
            "epoch": epoch,
            "online_training_mae_bpm": float(
                training_loss_sum.item() / len(train_features) * scaler["scale_bpm"]
            ),
            "validation": metrics,
            "seconds": round(time.perf_counter() - epoch_start, 3),
        }
        history.append(row)
        if metrics["mae_bpm"] < best_mae:
            best_mae = metrics["mae_bpm"]
            best_result = {"epoch": epoch, "validation": metrics}
            torch.save(
                {
                    "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "hidden_size": candidate["hidden_size"],
                    "scaler": scaler,
                    "epoch": epoch,
                    "data_run_id": config["data_run_id"],
                    "input_features": config["input_features"],
                    "prediction_head": config["prediction_head"],
                },
                checkpoint_path,
            )

        # 작은 개선도 최선 가중치에 반영. 조기 종료 대기는 min_delta 이상 개선 시 초기화
        if best_mae < significant_best - config["min_delta_bpm"]:
            significant_best = best_mae
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        write_json(output / f"{candidate['name']}_history.json", history)
        print(json.dumps({"candidate": candidate["name"], **row}), flush=True)
        if epochs_without_improvement >= config["patience"]:
            break

    return {
        **candidate,
        **best_result,
        "epochs_completed": len(history),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": checkpoint_path.name,
        "history": history,
    }


def select_candidate(results):
    # 검증 MAE 최소 후보 선택. 동률이면 검증 RMSE, 모델 크기 순서로 비교
    return min(
        results,
        key=lambda result: (
            result["validation"]["mae_bpm"],
            result["validation"]["rmse_bpm"],
            result["hidden_size"],
        ),
    )


def run_training(config, pipeline_config, output):
    # 전처리 완료 여부·실행 ID 확인 후 학습·검증 데이터 로드
    data_directory = Path(pipeline_config["private_root"]) / config["data_run_id"]
    if not (data_directory / "SUCCESS.txt").is_file():
        raise ValueError("Preprocessing run is incomplete")
    source_summary = json.loads((data_directory / "aggregate_summary.json").read_text(encoding="utf-8"))
    if source_summary["run_id"] != config["data_run_id"]:
        raise ValueError("Data run ID mismatch")
    train = load_split(data_directory, "train")
    validation = load_split(data_directory, "validation")
    for split, data in (("train", train), ("validation", validation)):
        expected = next(
            row["windows"] for row in source_summary["window_coverage_by_split"] if row["split"] == split
        )
        if len(data["targets"]) != expected:
            raise ValueError("Prediction row count mismatch")
    if np.intersect1d(train["subjects"], validation["subjects"]).size:
        raise ValueError("Training and validation patients overlap")

    scaler = fit_scaler(train["values"])
    write_json(output / "scaler.json", scaler)
    train_features = torch.from_numpy(transform_inputs(train["values"], train["masks"], scaler)).to(
        config["device"]
    )
    validation_features = torch.from_numpy(
        transform_inputs(validation["values"], validation["masks"], scaler)
    ).to(config["device"])
    train_targets = torch.as_tensor(
        (train["targets"] - scaler["mean_bpm"]) / scaler["scale_bpm"],
        dtype=torch.float32,
        device=config["device"],
    )

    # 같은 검증 정답으로 baseline 재계산 후 전처리 집계와 대조
    training_mean = float(train["targets"].mean())
    baselines = {}
    for name, predictions in (
        ("persistence", validation["values"][:, -1]),
        ("six_hour_mean", validation["values"].mean(axis=1)),
        ("training_mean", np.full(len(validation["targets"]), training_mean)),
    ):
        metrics = regression_metrics(validation["targets"], predictions, validation["subjects"])
        expected = next(
            row
            for row in source_summary["baseline_metrics"]
            if row["split"] == "validation" and row["model"] == name
        )
        if (
            abs(metrics["mae_bpm"] - expected["mae_bpm"]) > 1e-8
            or abs(metrics["rmse_bpm"] - expected["rmse_bpm"]) > 1e-8
        ):
            raise ValueError("Validation baseline mismatch")
        baselines[name] = metrics

    print(
        json.dumps(
            {
                "training_windows": len(train["targets"]),
                "validation_windows": len(validation["targets"]),
                "device": config["device"],
            }
        ),
        flush=True,
    )
    results = [
        train_candidate(
            candidate, config, train_features, train_targets, validation_features, validation, scaler, output
        )
        for candidate in config["candidates"]
    ]
    selected = select_candidate(results)
    shutil.copy2(output / selected["checkpoint"], output / "selected.pt")

    # 저장한 모델을 다시 로드해 검증 지표 재현 확인. 테스트 데이터 사용 없음
    checkpoint = torch.load(output / "selected.pt", map_location="cpu", weights_only=True)
    reloaded = HeartRateLSTM(checkpoint["hidden_size"]).to(config["device"])
    reloaded.load_state_dict(checkpoint["state_dict"])
    replay = regression_metrics(
        validation["targets"],
        predict_bpm(reloaded, validation_features, checkpoint["scaler"], config["evaluation_batch_size"]),
        validation["subjects"],
    )
    if abs(replay["mae_bpm"] - selected["validation"]["mae_bpm"]) > 1e-8:
        raise ValueError("Checkpoint reload mismatch")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_run_id": config["model_run_id"],
        "data_run_id": config["data_run_id"],
        "config": config,
        "scaler": scaler,
        "training_windows": len(train["targets"]),
        "validation_windows": len(validation["targets"]),
        "validation_baselines": baselines,
        "candidates": results,
        "selected_candidate": selected["name"],
        "selected_epoch": selected["epoch"],
        "selected_validation": replay,
        "selection_split": "validation",
        "test_patient_data_loaded": False,
        "test_evaluated": False,
        "checkpoint_reload_verified": True,
        "patient_split_overlap": 0,
        "checkpoint_sha256": file_hash(output / "selected.pt"),
        "training_source_sha256": file_hash(Path(__file__)),
        "preprocessing_fingerprint": source_summary["code_config_sha256"],
        "input_file_sha256": {
            split: file_hash(data_directory / f"windows_{split}.parquet") for split in ("train", "validation")
        },
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pyarrow")},
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if config["device"] == "cuda" else "CPU",
        "limitations": [
            "Validation was used for early stopping and model selection; it is not final test performance.",
            "One seed and two model sizes; no confidence intervals or clinical alert validation.",
            "LSTM also uses observation masks; baselines use the same six HR values without masks.",
            "Historical test baseline results were previously inspected; no new test patient data were loaded here.",
            "Online training MAE summarizes batches during parameter updates, not a final frozen-model training evaluation.",
        ],
    }
    write_json(output / "training_summary.json", report)
    report_directory = ROOT / "reports" / config["model_run_id"]
    report_directory.mkdir(parents=True, exist_ok=False)
    write_json(report_directory / "training_summary.json", report)
    (output / "SUCCESS.txt").write_text(
        "Training, validation selection and checkpoint reload passed. Test not evaluated.\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": selected["name"],
                "epoch": selected["epoch"],
                "validation": replay,
                "test_evaluated": False,
            }
        ),
        flush=True,
    )
    return report


def freeze_selection(config, config_path, output, report):
    # 선택한 모델·설정의 평가용 manifest 작성. 테스트 데이터 접근 없음
    candidate_path = output / f"{report['selected_candidate']}.pt"
    checks = {
        "candidate_matches_selected": file_hash(candidate_path) == report["checkpoint_sha256"],
        "checkpoint_replay_verified": report["checkpoint_reload_verified"],
        "scaler_training_only": report["scaler"]["fitted_split"] == "train",
        "test_not_loaded": not report["test_patient_data_loaded"],
    }
    if not all(checks.values()):
        raise ValueError("Cannot freeze an inconsistent training result")
    manifest = {
        "model_run_id": config["model_run_id"],
        "data_run_id": config["data_run_id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_candidate": report["selected_candidate"],
        "epoch": report["selected_epoch"],
        "checkpoint_path": str((output / "selected.pt").resolve()),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "scaler_sha256": file_hash(output / "scaler.json"),
        "training_source_sha256": report["training_source_sha256"],
        "training_config_sha256": file_hash(config_path),
        "preprocessing_fingerprint": report["preprocessing_fingerprint"],
        "selection_rule": "minimum validation MAE",
        "selection_is_frozen": True,
        "test_evaluation_status": "pending",
        "verification_checks": checks,
    }
    write_json(ROOT / "reports" / config["model_run_id"] / "selection_manifest.json", manifest)


def main():
    # 새 모델 실행 폴더에서만 학습. 기존 결과 덮어쓰기 방지
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/lstm.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pipeline_config = json.loads((ROOT / "configs/pipeline.json").read_text(encoding="utf-8"))
    if config["test_patient_data_allowed"] or config["selection_metric"] != "validation_mae_bpm":
        raise ValueError("Training requires validation-only selection")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config["model_run_id"]):
        raise ValueError("Invalid model run ID")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    output = (Path(config["private_model_root"]) / config["model_run_id"]).resolve()
    if output.is_relative_to(ROOT) or output.is_relative_to(Path(pipeline_config["raw_root"]).resolve()):
        raise ValueError("Models must remain outside the source and raw-data directories")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config_snapshot.json", config)
    shutil.copy2(__file__, output / "train_lstm_source.py")
    try:
        set_reproducibility(config["seed"], config["cpu_threads"])
        report = run_training(config, pipeline_config, output)
        freeze_selection(config, args.config, output, report)
    except Exception as exc:
        (output / "error_private.log").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Training failed; exception class: {type(exc).__name__}. Details retained locally.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
