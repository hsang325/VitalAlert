# VitalAlert

과거 6시간의 심박수로 다음 한 시간의 평균 심박수를 예측하는 로컬 웹 애플리케이션입니다.
Predicts next-hour mean heart rate from the previous six hours in a local web application.

[한국어](#korean) · [English](#english)

<a id="korean"></a>
## 한국어 실행 안내

### 1. 필수 종속성 설치

Python **3.12**를 설치하고 GitHub에서 받은 소스를 압축 해제합니다. VS Code에서 `app.py`가 있는 폴더를 열고 **터미널 → 새 터미널**을 선택합니다. 모든 명령은 이 폴더에서 실행하며 아래는 **Windows PowerShell / CPU** 기준입니다.

```powershell
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

첫 명령에서 Python 3.12인지 확인합니다. Windows 실행기가 있다면 `py -3.12 -m venv .venv`로 환경을 만들 수도 있습니다. 위 패키지는 LSTM 실행용이며 **평균 baseline만 실행할 때는 Python만 필요**합니다.
전처리·학습·평가 또는 저장된 예측의 집계 재계산에는 다음 패키지도 설치합니다.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS에서는 `.venv\Scripts\python.exe` 대신 `.venv/bin/python`을 사용합니다. macOS는 PyTorch 설치의 `--index-url` 옵션을 생략합니다. GPU 사용 시 [공식 PyTorch 설치 안내](https://pytorch.org/get-started/previous-versions/#v2-11-0)를 따릅니다.

### 2. 웹 애플리케이션 시작

```powershell
.venv\Scripts\python.exe app.py
```

브라우저에서 **http://127.0.0.1:8765** 를 엽니다. 종료는 터미널에서 `Ctrl+C`입니다. 포트가 사용 중이면 `--port 8767`을 붙이고 주소도 `http://127.0.0.1:8767`로 바꿉니다.
이 실행은 **가상 사례와 6시간 평균 baseline**이며 LSTM은 다음 단계로 불러옵니다.

### 3. 학습된 모델 불러오기

`selected.pt`는 이 프로젝트의 학습 스크립트가 만든 모델이며 정규화 정보도 포함합니다. 아래 경로를 자신의 모델 위치로 바꿉니다. GitHub에는 모델이 포함되지 않으므로 없다면 5번의 직접 학습 순서를 따릅니다.

```powershell
.venv\Scripts\python.exe app.py --checkpoint "C:/your-model/selected.pt" --device cpu
```

화면의 모델 표시가 **LSTM**인지 확인합니다. 과거→최근 순서로 가상 값 `80, 82, 84, 86, 88, 90`을 입력해 예측할 수 있습니다.
준비된 실제 사례가 있다면 같은 모델로 만든 웹용 파일을 추가합니다.

```powershell
.venv\Scripts\python.exe app.py --checkpoint "C:/your-model/selected.pt" --cases "C:/your-cases/cases_private.json" --device cpu
```

사례 선택 → 시점 이동 → **사후 정답 함께 보기**로 실제값과 예측 오차를 확인합니다. 정답은 예측 입력에 사용되지 않습니다. 웹용 사례는 아래 `prepare_demo_case.py`로 만들며 경보 분석용 JSON으로 대체하지 않습니다.

### 4. MIMIC-IV 없이 가능한 기능

| 기능 | 필요한 것 | MIMIC 원본 |
|---|---|---|
| 가상 사례·수동 입력·평균 baseline·범위 표시 | Python과 소스 | 불필요 |
| LSTM 예측 | 모델 실행 패키지와 `selected.pt` | 불필요 |
| 실제 사례의 정답·오차 확인 | 해당 모델과 준비된 웹용 `cases_private.json` | 불필요 |
| 기존 시험 예측의 집계 재계산 | 전체 패키지와 5번의 예측·평가 파일 | 불필요 |
| 전처리부터 새 모델 학습 | 전체 패키지와 접근 권한이 있는 MIMIC-IV 2.2 | 필요 |

실제 사례·예측 파일·모델은 공개 저장소에 포함하지 않습니다. 환자 파생 자료는 로컬에 보관합니다. 50 미만·120 초과 표시는 시연용 규칙이며 임상적으로 검증된 경보가 아닙니다.

### 5. 보고된 평가 결과 재현에 필요한 파일

**기존 수치 확인**과 **새 모델 학습**은 다릅니다. 모델 파일만으로는 MAE를 계산할 수 없고 실제 정답이 필요합니다.

| 확인 방법 | 필요한 파일 |
|---|---|
| 기존 예측으로 회귀·경보 집계 재계산 | 당시 `test_predictions.parquet`, `test_summary.json`, 일치하는 `configs/alert.json` |
| 기존 모델로 시험 추론 재실행 | 해당 실행의 `selected.pt`, `windows_test.parquet`, `windows_train.parquet`, `windows_validation.parquet`, `scaler.json`, `training_summary.json`, `selection_manifest.json`, `train_lstm_source.py`, 전처리 `aggregate_summary.json`, 전처리·모델의 `SUCCESS.txt`, 당시 소스와 경로·학습 설정 |

저장된 예측의 집계를 다시 계산하려면 경로를 자신의 파일 위치로 바꿉니다.

```powershell
.venv\Scripts\python.exe scripts/summarize_predictions.py --predictions "C:/your-evaluation/test_predictions.parquet" --summary "C:/your-evaluation/test_summary.json" --output "reports/aggregate_replay.json"
```

결과는 `reports/aggregate_replay.json`입니다. 출력이 이미 있으면 새 이름을 사용합니다. 시험 추론 재실행은 아래 평가 명령을 사용하되 당시 소스·설정·파일 해시가 일치해야 합니다. 현재 코드에 과거 manifest만 넣으면 거부될 수 있습니다. `results/reference_metrics.json`은 기존 결과 기록이며 재평가 결과가 아닙니다.

<details>
<summary>MIMIC-IV 2.2로 직접 학습·평가하고 웹 사례 만들기</summary>

모델이 없는 사용자를 위한 전체 실행 순서입니다. 1번의 전체 패키지를 설치한 뒤 경로를 설정합니다.

| 설정 파일 | 키 | 기본값 |
|---|---|---|
| `configs/pipeline.json` | `raw_root` | `../mimic-iv-2.2` |
| `configs/pipeline.json` | `private_root` | `../vitalalert-data` |
| `configs/lstm.json` | `private_model_root` | `../vitalalert-data/models` |

`raw_root`는 정식 2.2 배포본의 `SHA256SUMS.txt`, `hosp/patients.csv.gz`, `icu/d_items.csv.gz`, `icu/icustays.csv.gz`, `icu/chartevents.csv.gz`가 있는 폴더입니다. CSV 압축은 풀지 않습니다.
결과 경로는 소스와 원본 폴더 바깥에 둡니다. 상대 경로는 `app.py`가 있는 폴더 기준이며 JSON 절대 경로는 `C:/data/...`처럼 적습니다. 경로를 변경했다면 아래 명령의 경로도 바꿉니다.

명령을 **하나씩 실행하고 정상 종료 후** 다음으로 넘어갑니다.

```powershell
.venv\Scripts\python.exe scripts/pipeline.py
.venv\Scripts\python.exe scripts/train_lstm.py
.venv\Scripts\python.exe scripts/evaluate_lstm.py
```

| 단계 | 기본 설정의 완료 확인 위치 |
|---|---|
| 전처리 | `../vitalalert-data/hr-v1/SUCCESS.txt` |
| 학습 | `../vitalalert-data/models/lstm-v1/SUCCESS.txt`, `selected.pt`, `training_summary.json` |
| 평가 | `../vitalalert-data/models/lstm-v1/test-v1/SUCCESS.txt`, `reports/test-v1/test_summary.json` |

실제 시험 사례를 웹에서 확인하려면 다음을 실행합니다.

```powershell
.venv\Scripts\python.exe scripts/prepare_demo_case.py --evaluation-dir "../vitalalert-data/models/lstm-v1/test-v1" --checkpoint "../vitalalert-data/models/lstm-v1/selected.pt" --test-windows "../vitalalert-data/hr-v1/windows_test.parquet" --output "../vitalalert-data/models/lstm-v1/demo-v1" --validation-output "reports/demo-v1/verification.json"
.venv\Scripts\python.exe app.py --checkpoint "../vitalalert-data/models/lstm-v1/selected.pt" --cases "../vitalalert-data/models/lstm-v1/demo-v1/cases_private.json" --device cpu
```

사례 생성의 `status: passed`를 확인합니다. 12개 이상 연속 시험 창이 있어야 하며, 사례가 준비되지 않으면 `--cases` 없이 모델만 실행할 수 있습니다.
추가 경보 집계는 `scripts/prepare_alert_review.py`를 실행하고 `reports/alert-v1/alert_summary.json`에서 확인합니다.

재실행은 기존 결과를 보존하고 새 ID를 사용합니다. 전처리 `run_id`와 LSTM `data_run_id`를 맞춥니다. 모델 `model_run_id`를 바꾸면 평가에 `--manifest reports/새모델ID/selection_manifest.json`을 전달합니다. 평가 ID는 `--evaluation-id 새평가ID`로 바꾸고 경보 설정의 `evaluation_id`도 맞춥니다. 경보 분석 `run_id`와 사례 출력 경로도 새 이름을 사용합니다. 학습·평가 사이에 소스·설정·입력을 변경하지 않습니다.

전체 처리는 충분한 메모리·저장 공간·시간이 필요합니다. 기본 CPU 학습은 후보 2개를 최대 12 epoch 학습하며 완료가 수렴을 뜻하지는 않습니다. GPU 학습은 시작 전에 `configs/lstm.json`의 `device`를 `cuda`로 설정합니다. 새 학습은 장치·패키지에 따라 모델과 수치가 달라질 수 있어 보고서 숫자와의 완전 일치를 보장하지 않습니다.

</details>

---

<a id="english"></a>
## English instructions

### 1. Install required dependencies

Install **Python 3.12** and extract the source downloaded from GitHub. In VS Code, open the folder containing `app.py`, then choose **Terminal → New Terminal**. Run all commands from that folder. These commands target **Windows PowerShell / CPU**.

```powershell
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

Check that the first command reports Python 3.12. With the Windows launcher, `py -3.12 -m venv .venv` is another option. These packages enable LSTM; **the baseline interface only needs Python**.
For preprocessing, training, evaluation or aggregate replay, also install:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Linux/macOS use `.venv/bin/python` instead of `.venv\Scripts\python.exe`. On macOS omit the PyTorch `--index-url` option. For GPU, follow the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v2-11-0).

### 2. Start the web application

```powershell
.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:8765**. Stop with `Ctrl+C` in the terminal. If the port is busy, add `--port 8767` and open `http://127.0.0.1:8767` instead.
This runs **synthetic cases with the six-hour mean baseline**. Load LSTM as described next.

### 3. Load a trained model

`selected.pt` is a checkpoint produced by this project's training script and includes normalization statistics. Replace the example path with your checkpoint location. Weights are not included on GitHub; train your own using section 5 if needed.

```powershell
.venv\Scripts\python.exe app.py --checkpoint "C:/your-model/selected.pt" --device cpu
```

Check that the interface identifies **LSTM**. Try the synthetic values `80, 82, 84, 86, 88, 90`, ordered oldest to latest.
To load a prepared actual case, add the web-case file generated for the same checkpoint:

```powershell
.venv\Scripts\python.exe app.py --checkpoint "C:/your-model/selected.pt" --cases "C:/your-cases/cases_private.json" --device cpu
```

Select the case, move through time and enable **사후 정답 함께 보기** (show retrospective actual targets) to compare actual values and prediction errors. Targets are not inference inputs. Generate web cases with `prepare_demo_case.py` below; analysis-only alert JSON files are not substitutes.

### 4. Features available without MIMIC-IV

| Feature | Requirements | Raw MIMIC files |
|---|---|---|
| Synthetic cases, manual inputs, mean baseline and range display | Python and source | Not needed |
| LSTM predictions | Model dependencies and `selected.pt` | Not needed |
| Actual-case targets and errors | Matching model and prepared web `cases_private.json` | Not needed |
| Recompute saved test aggregates | Full dependencies and prediction/evaluation files in section 5 | Not needed |
| Preprocess and train a new model | Full dependencies and authorized MIMIC-IV 2.2 access | Required |

Actual cases, predictions and weights are excluded from the public repository. Keep patient-derived artifacts local. The below-50 / above-120 display rule is a demonstration, not a clinically validated alarm.

### 5. Files needed to reproduce reported evaluation results

**Checking historical results** and **training a new model** are different. A checkpoint alone cannot reproduce MAE; actual targets are required.

| Method | Required files |
|---|---|
| Recompute regression/alert aggregates from saved predictions | Historical `test_predictions.parquet`, `test_summary.json`, matching `configs/alert.json` |
| Rerun test inference with the historical model | Matching `selected.pt`, `windows_test.parquet`, `windows_train.parquet`, `windows_validation.parquet`, `scaler.json`, `training_summary.json`, `selection_manifest.json`, `train_lstm_source.py`, preprocessing `aggregate_summary.json`, preprocessing/model `SUCCESS.txt`, original source and path/training configurations |

To recompute saved aggregates, replace the paths with your local files:

```powershell
.venv\Scripts\python.exe scripts/summarize_predictions.py --predictions "C:/your-evaluation/test_predictions.parquet" --summary "C:/your-evaluation/test_summary.json" --output "reports/aggregate_replay.json"
```

Results are written to `reports/aggregate_replay.json`; use a new name if it exists. To rerun inference, use the evaluation command below with matching historical source, configuration and file hashes. An old manifest paired with changed source may be rejected. `results/reference_metrics.json` is a historical record, not a newly computed evaluation.

<details>
<summary>Train, evaluate and prepare a web case from MIMIC-IV 2.2</summary>

This is the complete workflow for users without a checkpoint. Install the full dependencies in section 1, then configure paths:

| Configuration | Key | Default |
|---|---|---|
| `configs/pipeline.json` | `raw_root` | `../mimic-iv-2.2` |
| `configs/pipeline.json` | `private_root` | `../vitalalert-data` |
| `configs/lstm.json` | `private_model_root` | `../vitalalert-data/models` |

`raw_root` must contain the full 2.2 release files: `SHA256SUMS.txt`, `hosp/patients.csv.gz`, `icu/d_items.csv.gz`, `icu/icustays.csv.gz`, and `icu/chartevents.csv.gz`. Keep CSV files compressed.
Output paths must be outside both source and raw-data folders. Relative paths resolve from the folder containing `app.py`; JSON absolute paths can use `C:/data/...`. If you change configured paths, update command paths below too.

Run commands **one at a time**, proceeding only after successful completion:

```powershell
.venv\Scripts\python.exe scripts/pipeline.py
.venv\Scripts\python.exe scripts/train_lstm.py
.venv\Scripts\python.exe scripts/evaluate_lstm.py
```

| Stage | Completion evidence with default paths |
|---|---|
| Preprocessing | `../vitalalert-data/hr-v1/SUCCESS.txt` |
| Training | `../vitalalert-data/models/lstm-v1/SUCCESS.txt`, `selected.pt`, `training_summary.json` |
| Evaluation | `../vitalalert-data/models/lstm-v1/test-v1/SUCCESS.txt`, `reports/test-v1/test_summary.json` |

Prepare and open an actual test case:

```powershell
.venv\Scripts\python.exe scripts/prepare_demo_case.py --evaluation-dir "../vitalalert-data/models/lstm-v1/test-v1" --checkpoint "../vitalalert-data/models/lstm-v1/selected.pt" --test-windows "../vitalalert-data/hr-v1/windows_test.parquet" --output "../vitalalert-data/models/lstm-v1/demo-v1" --validation-output "reports/demo-v1/verification.json"
.venv\Scripts\python.exe app.py --checkpoint "../vitalalert-data/models/lstm-v1/selected.pt" --cases "../vitalalert-data/models/lstm-v1/demo-v1/cases_private.json" --device cpu
```

Check `status: passed` from case preparation. At least 12 consecutive test windows are required; without a prepared case, the model can still run without `--cases`.
For additional alert aggregates, run `scripts/prepare_alert_review.py` and inspect `reports/alert-v1/alert_summary.json`.

Preserve existing outputs and use new IDs for reruns. Match pipeline `run_id` and LSTM `data_run_id`. When changing `model_run_id`, pass evaluator `--manifest reports/NEW_MODEL_ID/selection_manifest.json`. Change evaluation ID with `--evaluation-id NEW_TEST_ID` and match alert-config `evaluation_id`. Also use new names for alert `run_id` and case output paths. Do not change source, configuration or inputs between training and evaluation.

Full processing needs sufficient memory, storage and time. Defaults train two candidates on CPU for at most 12 epochs; completion does not establish convergence. For GPU training, set `device` to `cuda` in `configs/lstm.json` before starting. New training can produce different models and metrics across hardware/software; exact agreement with historical report numbers is not guaranteed.

</details>
