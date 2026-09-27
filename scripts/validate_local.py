"""Recalculate 25 stored prediction windows per split from cleaned records."""
from collections import defaultdict
from datetime import timedelta, datetime, timezone
import json
from pathlib import Path
import statistics

import duckdb

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT/'configs/pipeline.json').read_text(encoding='utf-8'))
    private = Path(config['private_root'])/config['run_id']
    con = duckdb.connect(str(private/'pipeline.duckdb'), read_only=True)
    checked = 0
    mismatches = defaultdict(int)
    try:
        for split in ['train', 'validation', 'test']:
            windows = con.execute("""SELECT w.stay_id,w.t,w.target_hr,w.x0,w.x1,w.x2,w.x3,w.x4,w.x5,
              w.mask0,w.mask1,w.mask2,w.mask3,w.mask4,w.mask5,s.intime,s.outtime
              FROM windows w JOIN stays s USING(stay_id) WHERE w.split=?
              ORDER BY md5(CAST(w.stay_id AS VARCHAR)||CAST(w.t AS VARCHAR)) LIMIT 25""", [split]).fetchall()
            for row in windows:
                stay, t, target, *rest = row
                actual = rest[:6]
                masks = rest[6:12]
                intime, outtime = rest[12:]
                events = con.execute("""SELECT charttime,storetime,valuenum FROM clean_events
                  WHERE itemid=220045 AND stay_id=? AND charttime>? AND charttime<=?""",
                  [stay, t-timedelta(hours=7), t+timedelta(hours=1)]).fetchall()
                bins = defaultdict(list)
                observed_target = []
                for chart, stored, value in events:
                    floor = chart.replace(minute=0, second=0, microsecond=0)
                    end = floor if chart == floor else floor+timedelta(hours=1)
                    if end-timedelta(hours=1)<intime or end>outtime:
                        continue
                    if end == t+timedelta(hours=1):
                        observed_target.append(value)
                    if stored is not None and stored<=t and end<=t:
                        bins[end].append(value)
                raw_means = {k:statistics.fmean(v) for k, v in bins.items()}
                expected = []
                expected_masks = []
                for lag in range(5, -1, -1):
                    boundary = t-timedelta(hours=lag)
                    expected_masks.append(boundary in raw_means)
                    expected.append(raw_means.get(boundary, raw_means.get(boundary-timedelta(hours=1))))
                if any(x is None or abs(x-y)>1e-8 for x, y in zip(expected, actual)):
                    mismatches['input_values'] += 1
                if expected_masks != masks:
                    mismatches['input_masks'] += 1
                if not observed_target or abs(statistics.fmean(observed_target)-target)>1e-8:
                    mismatches['observed_target'] += 1
                checked += 1
        report = {'run_id':config['run_id'], 'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'independent_python_windows_checked':checked,
          'mismatch_counts':dict(mismatches), 'passed':checked == 75 and not mismatches,
          'patient_records_exported_to_code_workspace':False}
        # Synthetic test results are recorded separately by run_tests.py.
        (ROOT/'reports').mkdir(exist_ok=True)
        (ROOT/'reports/validation_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return 0 if report['passed'] else 1
    except Exception as exc:
        print('Independent validation failed; class='+type(exc).__name__)
        return 1
    finally:
        con.close()


if __name__ == '__main__':
    raise SystemExit(main())
