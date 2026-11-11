# 저장된 시험 예측을 읽어 회귀·범위 표시 집계를 재계산하고 기존 평가와 대조
# main() → 예측 파일 해시 확인 → 네 모델 집계 → 기존 회귀 결과 대조 → 집계만 저장
# 원본 MIMIC이나 가중치는 필요 없지만 예측 Parquet는 접근이 허용된 로컬 자료여야 함

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import pyarrow.parquet as pq

from alerting import alert_metrics
from train_lstm import ROOT, file_hash, regression_metrics, write_json


def main():
    parser = argparse.ArgumentParser(description='Recompute aggregates from an existing local test prediction file')
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError('Choose a new output file')
        summary = json.loads(args.summary.read_text(encoding='utf-8'))
        prediction_hash = file_hash(args.predictions)
        if prediction_hash != summary['prediction_file_sha256']:
            raise ValueError('Prediction file differs from its evaluation record')
        names = ['persistence', 'six_hour_mean', 'training_mean', 'lstm']
        columns = ['subject_id', 'target_hr'] + [f'predicted_{name}_bpm' for name in names]
        table = pq.read_table(args.predictions, columns=columns)
        targets = table['target_hr'].to_numpy()
        subjects = table['subject_id'].to_numpy()
        policy = json.loads((ROOT / 'configs/alert.json').read_text(encoding='utf-8'))
        results = {}
        for name in names:
            predictions = table[f'predicted_{name}_bpm'].to_numpy()
            metrics = regression_metrics(targets, predictions, subjects)
            expected = summary['metrics'][name]
            if metrics['windows'] != expected['windows'] or metrics['patients'] != expected['patients']:
                raise ValueError('Evaluation population differs')
            for key in ['mae_bpm', 'rmse_bpm', 'patient_macro_mae_bpm']:
                if not np.isclose(metrics[key], expected[key], atol=1e-8, rtol=0):
                    raise ValueError('Aggregate differs from the recorded evaluation')
            results[name] = {'regression': metrics, 'alert': alert_metrics(targets, predictions, policy)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {
            'evaluation_id': summary['evaluation_id'], 'prediction_file_sha256': prediction_hash,
            'regression_matches_record': True, 'policy': policy, 'metrics': results,
            'inference_or_training_performed': False,
        })
    except Exception as error:
        # 환자 식별자나 행이 오류에 포함되지 않도록 종류·위치만 출력
        location = traceback.extract_tb(error.__traceback__)[-1]
        parser.exit(1, f'Aggregate replay failed: {type(error).__name__} at line {location.lineno}. Check paths, hashes and output filename.\n')
    print('Four-model aggregate replay matches the recorded regression evaluation.')


if __name__ == '__main__':
    main()
