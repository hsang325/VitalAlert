"""Deterministic, invented records only. Run before any real-data processing."""
from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from pipeline import register_variables, build_clean_events, build_hourly, build_windows, safe_summary


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.db = duckdb.connect()
        register_variables(self.db)
        self.start = datetime(2020, 1, 1)
        self.db.execute("CREATE TABLE stays(subject_id BIGINT,hadm_id BIGINT,stay_id BIGINT,intime TIMESTAMP,outtime TIMESTAMP,split VARCHAR)")
        self.db.execute("INSERT INTO stays VALUES (1,11,101,?,?, 'train'),(2,22,202,?,?, 'validation'),(3,33,303,?,?, 'test')", [self.start, self.start+timedelta(hours=12)]*3)
        self.db.execute("CREATE TABLE candidates(subject_id BIGINT,hadm_id BIGINT,stay_id BIGINT,itemid INTEGER,charttime TIMESTAMP,storetime TIMESTAMP,valuenum DOUBLE,valueuom VARCHAR,warning INTEGER)")

    def tearDown(self):
        self.db.close()

    def event(self, hour, value, stored=None, subject=1, stay=101, unit='bpm', warning=0, item=220045):
        chart = self.start+timedelta(hours=hour)
        store = self.start+timedelta(hours=stored) if stored is not None else chart
        self.db.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?)", [subject, subject*11, stay, item, chart, store, value, unit, warning])

    def build(self):
        build_clean_events(self.db)
        build_hourly(self.db)
        build_windows(self.db)

    def hour(self, hour, column):
        return self.db.execute(f"SELECT {column} FROM hourly WHERE stay_id=101 AND t=?", [self.start+timedelta(hours=hour)]).fetchone()[0]

    def test_right_closed_boundary_and_average(self):
        self.event(1, 80)
        self.event(1.5, 90)
        self.event(2, 100)
        self.build()
        self.assertEqual(self.hour(1, 'observed_hr'), 80)
        self.assertEqual(self.hour(2, 'observed_hr'), 95)
        self.assertEqual(self.hour(1, 'target_hr'), 95)

    def test_fill_one_bin_only_and_never_target(self):
        self.event(1, 80)
        self.event(4, 120)
        self.build()
        self.assertEqual(self.hour(2, 'feature_hr'), 80)
        self.assertIsNone(self.hour(3, 'feature_hr'))
        self.assertIsNone(self.hour(1, 'target_hr'))

    def test_late_record_cannot_leak_into_input(self):
        self.event(1, 80, stored=3)
        self.build()
        self.assertEqual(self.hour(1, 'observed_hr'), 80)
        self.assertIsNone(self.hour(1, 'feature_hr'))

    def test_missing_storetime_not_used_as_feature(self):
        self.event(1, 80)
        self.db.execute("UPDATE candidates SET storetime=NULL")
        self.build()
        self.assertIsNone(self.hour(1, 'feature_hr'))

    def test_invalid_values_and_units_and_warnings(self):
        for hour, value in [(1, 0), (2, 300), (3, -1), (4, float('nan'))]:
            self.event(hour, value)
        self.event(5, 80, unit='wrong')
        self.event(6, 80, warning=1)
        self.event(7, 90)
        self.build()
        self.assertEqual(self.db.execute('SELECT count(*) FROM clean_events').fetchone()[0], 1)

    def test_outside_stay_and_wrong_identifiers(self):
        self.event(-1, 80)
        self.event(13, 80)
        self.event(1, 90, subject=2, stay=101)
        self.build()
        self.assertEqual(self.db.execute('SELECT count(*) FROM clean_events').fetchone()[0], 0)

    def test_duplicate_records_removed(self):
        self.event(1, 80)
        self.event(1, 80)
        self.event(1, 100)
        self.build()
        self.assertEqual(self.hour(1, 'observed_hr'), 90)

    def test_no_cross_stay_fill(self):
        self.event(1, 80)
        self.build()
        self.assertEqual(self.db.execute('SELECT count(feature_hr) FROM hourly WHERE stay_id=202').fetchone()[0], 0)

    def test_six_hour_window_and_one_hour_horizon(self):
        for h in range(1, 9):
            self.event(h, 70+h)
        self.build()
        row = self.db.execute('SELECT x0,x1,x2,x3,x4,x5,target_hr FROM windows ORDER BY t LIMIT 1').fetchone()
        self.assertEqual(row, (71, 72, 73, 74, 75, 76, 77))
        self.assertEqual(self.db.execute('SELECT count(*) FROM windows').fetchone()[0], 2)

    def test_future_value_does_not_change_past_features(self):
        for h in range(1, 9):
            self.event(h, 70+h)
        self.build()
        before = self.db.execute("SELECT t,feature_hr FROM hourly WHERE stay_id=101 AND t<? ORDER BY t", [self.start+timedelta(hours=8)]).fetchall()
        self.db.execute("UPDATE candidates SET valuenum=200 WHERE charttime=?", [self.start+timedelta(hours=8)])
        self.build()
        after = self.db.execute("SELECT t,feature_hr FROM hourly WHERE stay_id=101 AND t<? ORDER BY t", [self.start+timedelta(hours=8)]).fetchall()
        self.assertEqual(before, after)

    def test_delayed_history_becomes_usable_when_available(self):
        for h in range(1, 9):
            self.event(h, 70+h, stored=h+0.5)
        self.build()
        row = self.db.execute('SELECT x0,x1,x2,x3,x4,x5,target_hr,mask5 FROM windows ORDER BY t LIMIT 1').fetchone()
        self.assertEqual(row, (71, 72, 73, 74, 75, 75, 77, False))

    def test_unavailable_future_update_cannot_change_earlier_window(self):
        for h in range(1, 9):
            self.event(h, 70+h)
        self.event(2, 200, stored=7)
        self.build()
        row = self.db.execute('SELECT x0,x1,x2,x3,x4,x5 FROM windows WHERE t=?', [self.start+timedelta(hours=6)]).fetchone()
        self.assertEqual(row, (71, 72, 73, 74, 75, 76))

    def test_two_missing_chart_bins_are_not_filled(self):
        for h in [1, 2, 3, 6, 7, 8]:
            self.event(h, 70+h)
        self.build()
        self.assertEqual(self.db.execute('SELECT count(*) FROM windows').fetchone()[0], 0)

    def test_partial_first_and_last_hours_excluded(self):
        self.db.execute("UPDATE stays SET intime=?,outtime=? WHERE stay_id=101", [self.start+timedelta(minutes=30), self.start+timedelta(hours=8, minutes=30)])
        self.build()
        first, last = self.db.execute('SELECT min(t),max(t) FROM hourly WHERE stay_id=101').fetchone()
        self.assertEqual(first, self.start+timedelta(hours=2))
        self.assertEqual(last, self.start+timedelta(hours=8))

    def test_evaluation_and_patient_split_checks(self):
        for subject, stay in [(1, 101), (2, 202), (3, 303)]:
            for h in range(1, 9):
                self.event(h, 70+h, subject=subject, stay=stay)
        self.build()
        summary = safe_summary(self.db)
        self.assertTrue(all(v == 0 for v in summary['integrity_checks'].values()))
        for metric in summary['baseline_metrics']:
            if metric['model'] == 'persistence':
                self.assertEqual(metric['mae_bpm'], 1)
                self.assertEqual(metric['rmse_bpm'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
