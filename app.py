# 6시간 심박수 입력 → 다음 시간 평균 예측 → 브라우저 표시
# 기본 실행: 가상 데이터 + 6시간 평균 baseline
# --checkpoint 지정 시 보유한 LSTM 사용
# 진입점 main(), 화면·API 처리는 make_handler()

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import threading


ROOT = Path(__file__).resolve().parent


def validate_input(body):
    # 오래된 구간부터 최근 구간 순서. 연속 두 구간의 결측 보완은 허용하지 않음
    if not isinstance(body, dict) or set(body) != {"values", "masks"}:
        raise ValueError("Expected values and masks")
    values, masks = body["values"], body["masks"]
    if not isinstance(values, list) or not isinstance(masks, list) or len(values) != 6 or len(masks) != 6:
        raise ValueError("Expected six hourly inputs")
    if any(type(value) not in (int, float) for value in values + masks):
        raise ValueError("Expected numeric inputs")
    if any(not math.isfinite(value) or not 0 < value < 300 for value in values):
        raise ValueError("Expected 0 < HR < 300")
    if any(mask not in (0, 1) for mask in masks):
        raise ValueError("Expected binary masks")
    if any(left == right == 0 for left, right in zip(masks, masks[1:])):
        raise ValueError("Consecutive missing inputs are not supported")
    return values, masks


def prepare_manual_input(body):
    # 화면의 빈 관측값은 오류로 처리. 미측정으로 표시한 구간만 직전 관측값으로 한 번 보완
    # 첫 구간은 앞선 시간의 관측값을 받지 않으므로 이 화면에서는 보완 불가
    if not isinstance(body, dict) or set(body) != {"values", "masks"}:
        raise ValueError("6개 심박수와 관측 표시가 필요합니다.")
    values, masks = body["values"], body["masks"]
    if not isinstance(values, list) or not isinstance(masks, list) or len(values) != 6 or len(masks) != 6:
        raise ValueError("시간별 입력은 정확히 6개가 필요합니다.")
    if any(type(mask) is not int or mask not in (0, 1) for mask in masks):
        raise ValueError("관측 표시는 0 또는 1이어야 합니다.")
    prepared = []
    for index, (value, observed) in enumerate(zip(values, masks)):
        if not observed:
            if index == 0:
                raise ValueError("첫 구간은 이전 관측값이 없어 보완할 수 없습니다.")
            if not masks[index - 1]:
                raise ValueError("연속 두 구간의 미측정은 보완할 수 없습니다.")
            prepared.append(prepared[-1])
        else:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 300:
                raise ValueError(f"{index + 1}번째 구간에 0 초과 300 미만의 숫자를 입력하세요.")
            prepared.append(value)
    return validate_input({"values": prepared, "masks": masks})


class Predictor:
    # 모델·정규화 값·추론 잠금을 요청 간 공유
    # LSTM 사용 시에만 학습 패키지 로드. 기본 화면에는 Python만 필요
    def __init__(self, checkpoint_path=None, device="cpu"):
        self.lock = threading.Lock()
        self.device = device
        self.model = None
        self.identity = {"name": "6시간 평균 baseline", "kind": "baseline", "device": "CPU"}
        if checkpoint_path is None:
            return

        sys.path.insert(0, str(ROOT / "scripts"))
        from train_lstm import HeartRateLSTM, set_reproducibility
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        with Path(checkpoint_path).open("rb") as source:
            self.checkpoint_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        if checkpoint["input_features"] != ["standardized_heart_rate", "observed_mask"]:
            raise ValueError("Unsupported checkpoint inputs")
        if checkpoint["prediction_head"] != "last_heart_rate_plus_learned_correction":
            raise ValueError("Unsupported prediction head")
        self.scaler = checkpoint["scaler"]
        if (
            self.scaler["fitted_split"] != "train"
            or not math.isfinite(self.scaler["mean_bpm"])
            or not math.isfinite(self.scaler["scale_bpm"])
            or self.scaler["scale_bpm"] <= 0
        ):
            raise ValueError("Invalid training scaler")
        set_reproducibility(20261028, 4)
        self.model = HeartRateLSTM(checkpoint["hidden_size"]).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.requires_grad_(False)
        self.identity = {
            "name": f"LSTM · {checkpoint['hidden_size']}",
            "kind": "lstm",
            "epoch": checkpoint["epoch"],
            "device": device.upper(),
        }

    def predict(self, values, masks):
        # 저장된 학습 정규화로 입력 변환 후 bpm으로 역변환. CPU/GPU 간 미세한 수치 차이 가능
        if self.model is None:
            return math.fsum(values) / len(values)
        import numpy as np
        import torch
        from train_lstm import predict_bpm, transform_inputs

        features = transform_inputs(np.array([values]), np.array([masks]), self.scaler)
        with self.lock:
            predictions = predict_bpm(self.model, torch.from_numpy(features).to(self.device), self.scaler, 1)
        prediction = float(predictions[0])
        if not math.isfinite(prediction):
            raise ValueError("Model returned a nonfinite prediction")
        return prediction


def make_demo_cases(predictor):
    # 환자 기록 없이 수식으로 생성한 범위 안/밖·급변 시연 사례
    signals = [
        ("demo_within", "가상 · 범위 안 흐름", [80 + 4 * math.sin(hour / 3) for hour in range(30)]),
        ("demo_high", "가상 · 상한 초과 흐름", [137 + 7 * math.sin(hour / 4) for hour in range(30)]),
        ("demo_low", "가상 · 하한 미만 흐름", [40 + 2 * math.cos(hour / 3) for hour in range(30)]),
        (
            "demo_change",
            "가상 · 급격한 변화",
            [82 + math.sin(hour) if hour < 18 else 142 + 3 * math.sin(hour) for hour in range(30)],
        ),
    ]
    cases = []
    for case_id, label, signal in signals:
        rows = []
        for index in range(24):
            values = signal[index : index + 6]
            masks = [1] * 6
            rows.append(
                {
                    "hour": index,
                    "inputs": values,
                    "masks": masks,
                    "actual": signal[index + 6],
                    "prediction": predictor.predict(values, masks),
                    "baseline": math.fsum(values) / 6,
                }
            )
        cases.append({"id": case_id, "label": label, "source": "synthetic", "anchor": 12, "rows": rows})
    return cases


def load_local_cases(path, policy, predictor):
    # 로컬 사례의 6시간 입력으로 현재 모델을 재실행. 기존 사례의 예측값 교체
    bundle = json.loads(Path(path).read_text(encoding="utf-8"))
    if bundle.get("checkpoint_sha256") and bundle["checkpoint_sha256"] != predictor.checkpoint_sha256:
        raise ValueError("Case checkpoint mismatch")
    for key in ("lower_bpm", "upper_bpm", "boundary_rule", "purpose"):
        if bundle["policy"][key] != policy[key]:
            raise ValueError("Case display policy mismatch")
    cases = bundle["cases"]
    if not cases:
        raise ValueError("No local cases")
    for case in cases:
        for row in case["rows"]:
            values, masks = validate_input({"values": row["inputs"], "masks": row["masks"]})
            row["prediction"] = predictor.predict(values, masks)
            row["baseline"] = math.fsum(values) / len(values)
    return cases


def make_handler(state, predictor):
    # 허용된 화면 파일과 API만 제공. 환자 자료·경로를 포함할 수 있는 요청 로그는 출력하지 않음
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def allowed_host(self):
            return self.headers.get("Host") in (
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            )

        def respond(self, status, payload, content_type="application/json; charset=utf-8"):
            if isinstance(payload, dict):
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if not self.allowed_host():
                return self.respond(403, {"error": "Local host only"})
            if self.path == "/api/state":
                return self.respond(200, state)
            routes = {
                "/": ("index.html", "text/html"),
                "/app.js": ("app.js", "text/javascript"),
                "/app.css": ("app.css", "text/css"),
            }
            if self.path not in routes:
                return self.respond(404, {"error": "Not found"})
            name, content_type = routes[self.path]
            self.respond(200, (ROOT / "web" / name).read_bytes(), content_type + "; charset=utf-8")

        def do_POST(self):
            if not self.allowed_host() or self.headers.get("Origin") != "http://" + self.headers.get(
                "Host", ""
            ):
                return self.respond(403, {"error": "Local origin required"})
            if self.path != "/api/predict":
                return self.respond(404, {"error": "Not found"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.respond(415, {"error": "JSON required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    return self.respond(413, {"error": "Input too large or empty"})
                values, masks = prepare_manual_input(json.loads(self.rfile.read(length)))
                prediction = predictor.predict(values, masks)
                status = (
                    -1
                    if prediction < state["policy"]["lower_bpm"]
                    else 1
                    if prediction > state["policy"]["upper_bpm"]
                    else 0
                )
                self.respond(
                    200, {
                        "prediction": prediction, "baseline": math.fsum(values) / 6,
                        "status": status, "inputs": values, "masks": masks,
                    }
                )
            except (json.JSONDecodeError, UnicodeError):
                self.respond(400, {"error": "입력 형식을 확인하세요."})
            except ValueError as error:
                self.respond(400, {"error": str(error)})
            except (TypeError, OverflowError):
                self.respond(400, {"error": "6개 숫자와 관측 표시를 확인하세요."})
            except Exception:
                self.respond(500, {"error": "추론에 실패했습니다. 모델과 실행 환경을 확인하세요."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="VitalAlert local heart-rate prediction interface")
    parser.add_argument(
        "--checkpoint", type=Path, help="Your local selected.pt; omitted for baseline screen demo"
    )
    parser.add_argument("--cases", type=Path, help="Optional local cases_private.json; requires a checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.cases and not args.checkpoint:
        parser.error("--cases requires --checkpoint")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    policy = json.loads((ROOT / "configs/alert.json").read_text(encoding="utf-8"))
    if not 0 < policy["lower_bpm"] < policy["upper_bpm"] < 300 or policy["boundary_rule"] != "strict_outside":
        parser.error("Invalid display bounds")
    try:
        predictor = Predictor(args.checkpoint, args.device)
        cases = load_local_cases(args.cases, policy, predictor) if args.cases else make_demo_cases(predictor)
        reference = json.loads((ROOT / "results/reference_metrics.json").read_text(encoding="utf-8"))
        state = {
            "mode": "local" if args.cases else "demo",
            "model": predictor.identity,
            "policy": policy,
            "cases": cases,
            "test_metrics": reference["metrics"],
        }
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state, predictor))
    except ModuleNotFoundError:
        parser.exit(1, "LSTM dependencies are missing. Install requirements.txt first; see README.md.\n")
    except Exception as error:
        parser.exit(
            1, f"Startup failed ({type(error).__name__}). Check model/config files and port; see README.md.\n"
        )
    print(
        f"VitalAlert: http://127.0.0.1:{args.port} | {predictor.identity['name']} | Ctrl+C to stop",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
