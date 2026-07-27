from __future__ import annotations

import unittest
from datetime import date, datetime

from timeseries_rows import split_timeseries_rows_for_write


class SplitTimeseriesRowsForWriteTests(unittest.TestCase):
    def test_zero_mode_splits_on_updated_to_time(self) -> None:
        rows = [
            {
                "instr_id": "A",
                "date1_dt": datetime(2026, 7, 25, 0, 0, 0),
                "data1": "1.000",
                "custom_fields": "{}",
            },
            {
                "instr_id": "A",
                "date1_dt": datetime(2026, 7, 27, 0, 0, 0),
                "data1": "2.000",
                "custom_fields": "{}",
            },
        ]

        past_rows, future_rows, duplicates_skipped = split_timeseries_rows_for_write(
            rows,
            updated_to_time=datetime(2026, 7, 26, 12, 0, 0),
            preserve_before_date=None,
        )

        self.assertEqual(duplicates_skipped, 0)
        self.assertEqual(len(past_rows), 1)
        self.assertEqual(len(future_rows), 1)
        self.assertEqual(past_rows[0][1], "2026-07-25 00:00:00")
        self.assertEqual(future_rows[0][1], "2026-07-27 00:00:00")

    def test_snapshot_mode_does_not_reinsert_preserved_future_rows(self) -> None:
        """Reproducing the futuredata duplicate bug.

        Snapshot deletes only date1 >= preserve_before_date, then must not re-insert
        earlier future-classified rows that remain in the table.
        """
        rows = [
            {
                "instr_id": "LP-1",
                "date1_dt": datetime(2026, 7, 27, 0, 0, 0),
                "data1": "1.000",
                "custom_fields": '{"calculation1": "1.000"}',
            },
            {
                "instr_id": "LP-1",
                "date1_dt": datetime(2026, 7, 28, 0, 0, 0),
                "data1": "2.000",
                "custom_fields": '{"calculation1": "2.000"}',
            },
            {
                "instr_id": "LP-1",
                "date1_dt": datetime(2026, 8, 1, 0, 0, 0),
                "data1": "3.000",
                "custom_fields": '{"calculation1": "3.000"}',
            },
        ]

        past_rows, future_rows, duplicates_skipped = split_timeseries_rows_for_write(
            rows,
            updated_to_time=datetime(2026, 7, 26, 12, 0, 0),
            preserve_before_date=date(2026, 8, 1),
        )

        self.assertEqual(duplicates_skipped, 0)
        self.assertEqual(past_rows, [])
        self.assertEqual(len(future_rows), 1)
        self.assertEqual(future_rows[0][1], "2026-08-01 00:00:00")

    def test_exact_duplicate_payloads_are_skipped(self) -> None:
        rows = [
            {
                "instr_id": "A",
                "date1_dt": datetime(2026, 7, 27, 0, 0, 0),
                "data1": "1.000",
                "custom_fields": "{}",
            },
            {
                "instr_id": "A",
                "date1_dt": datetime(2026, 7, 27, 0, 0, 0),
                "data1": "1.000",
                "custom_fields": "{}",
            },
        ]

        past_rows, future_rows, duplicates_skipped = split_timeseries_rows_for_write(
            rows,
            updated_to_time=datetime(2026, 7, 26, 12, 0, 0),
        )

        self.assertEqual(duplicates_skipped, 1)
        self.assertEqual(past_rows, [])
        self.assertEqual(len(future_rows), 1)

    def test_falls_back_to_date1_when_date1_dt_missing(self) -> None:
        rows = [
            {
                "instr_id": "A",
                "date1": "2026-07-27 00:00:00",
                "data1": "1.000",
                "custom_fields": "{}",
            }
        ]

        past_rows, future_rows, duplicates_skipped = split_timeseries_rows_for_write(
            rows,
            updated_to_time=datetime(2026, 7, 26, 12, 0, 0),
        )

        self.assertEqual(duplicates_skipped, 0)
        self.assertEqual(past_rows, [])
        self.assertEqual(len(future_rows), 1)


if __name__ == "__main__":
    unittest.main()
