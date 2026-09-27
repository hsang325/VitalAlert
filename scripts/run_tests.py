"""Run synthetic tests and record their result without reading MIMIC data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'expected_failures': len(result.expectedFailures),
        'unexpected_successes': len(result.unexpectedSuccesses),
        'passed': result.testsRun > 0 and result.wasSuccessful(),
        'source_sha256': {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ['scripts/pipeline.py', 'tests/test_pipeline.py', 'scripts/run_tests.py']
        },
    }
    path = ROOT / 'reports/synthetic_test_summary.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
