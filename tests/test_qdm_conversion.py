import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"

import sys

sys.path.insert(0, str(TOOLS_DIR))

import convert_parquet_to_duckdb as convert_parquet_to_duckdb
import preprocess_qdm_tick_csv as preprocess_qdm_tick_csv


class CalendarFilterTests(unittest.TestCase):
    def test_filter_excludes_maintenance_window_after_1700(self):
        origin_ts = pd.Timestamp("2000-01-01 17:00:00")
        full_index = convert_parquet_to_duckdb._build_full_index(
            pd.Timestamp("2024-01-02 16:59:10"),
            pd.Timestamp("2024-01-02 17:02:10"),
            "30s",
            origin_ts,
        )

        filtered = convert_parquet_to_duckdb._filter_index_by_calendar(full_index, "30s")

        self.assertIn(pd.Timestamp("2024-01-02 16:59:00"), filtered)
        self.assertIn(pd.Timestamp("2024-01-02 16:59:30"), filtered)
        self.assertNotIn(pd.Timestamp("2024-01-02 17:00:00"), filtered)
        self.assertNotIn(pd.Timestamp("2024-01-02 17:00:30"), filtered)


class CandleBuildTests(unittest.TestCase):
    def test_build_candles_drops_empty_gap_rows_before_indicators(self):
        idx = pd.to_datetime(
            [
                "2024-01-02 10:00:10",
                "2024-01-02 10:00:40",
                "2024-01-02 10:03:10",
            ]
        ).tz_localize("America/New_York")
        df_ticks = pd.DataFrame(
            {"price": [10.0, 11.0, 13.0], "volume": [1.0, 1.0, 1.0]},
            index=idx,
        )

        candles = convert_parquet_to_duckdb.build_candles(df_ticks, "30s")

        self.assertEqual(
            list(candles.index),
            [
                pd.Timestamp("2024-01-02 10:00:00"),
                pd.Timestamp("2024-01-02 10:00:30"),
                pd.Timestamp("2024-01-02 10:03:00"),
            ],
        )
        self.assertFalse(candles["open"].isna().any())
        self.assertFalse(candles["EMA20"].isna().all())


class InputTimezoneTests(unittest.TestCase):
    def test_load_qdm_ticks_supports_explicit_timezone_or_naive_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            csv_path.write_text(
                "DateTime,Bid,Volume\n"
                "20260315 16:59:59.000,3000.1,1\n"
                "20260315 17:00:00.000,3000.2,\n",
                encoding="utf-8",
            )

            localized = preprocess_qdm_tick_csv.load_qdm_ticks(csv_path, input_tz="America/New_York")
            naive = preprocess_qdm_tick_csv.load_qdm_ticks(csv_path, input_tz=None)

        self.assertEqual(str(localized.index.tz), "America/New_York")
        self.assertIsNone(naive.index.tz)


if __name__ == "__main__":
    unittest.main()
