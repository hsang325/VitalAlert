# 예측 평균을 설정 범위와 비교해 하한 미만/범위 안/상한 초과로 분류
# 정답에도 같은 기준을 적용해 일치·누락·과다 표시 집계
# validate_policy() → classify() → alert_metrics(). 임상 진단 기준으로 검증되지 않음

import numpy as np


def validate_policy(policy):
    # 경계값은 범위 안에 포함. 기본 표시 조건: 50 미만 또는 120 초과
    bounds = [policy["lower_bpm"], policy["upper_bpm"]]
    if not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1] <= 300:
        raise ValueError("Invalid display bounds")
    if (
        policy["boundary_rule"] != "strict_outside"
        or policy["purpose"] != "engineering_demonstration_not_clinical"
    ):
        raise ValueError("Unsupported alert policy")


def classify(values, policy):
    # 분류값: 하한 미만 -1, 범위 안 0, 상한 초과 1. 결측값은 정상으로 분류하지 않음
    validate_policy(policy)
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite alert input")
    return np.where(values < policy["lower_bpm"], -1, np.where(values > policy["upper_bpm"], 1, 0))


def alert_metrics(targets, predictions, policy):
    # 양성: 다음 시간 평균이 설정 범위 밖인 창
    # 겹치는 창 기준 집계이므로 독립 환자 수나 임상 사건 수와 다름
    actual = classify(targets, policy)
    predicted = classify(predictions, policy)
    if actual.ndim != 1 or actual.shape != predicted.shape or not actual.size:
        raise ValueError("Expected matching nonempty alert vectors")
    truth = actual != 0
    flagged = predicted != 0
    true_positive = int(np.sum(truth & flagged))
    false_positive = int(np.sum(~truth & flagged))
    false_negative = int(np.sum(truth & ~flagged))
    true_negative = int(np.sum(~truth & ~flagged))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "windows": len(actual),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": true_positive / precision_denominator if precision_denominator else None,
        "recall": true_positive / recall_denominator if recall_denominator else None,
        "predicted_outside_fraction": float(np.mean(flagged)),
        "actual_outside_fraction": float(np.mean(truth)),
        "three_class_order": ["below", "within", "above"],
        "three_class_confusion_actual_rows": [
            [
                int(np.sum((actual == actual_class) & (predicted == predicted_class)))
                for predicted_class in (-1, 0, 1)
            ]
            for actual_class in (-1, 0, 1)
        ],
    }
