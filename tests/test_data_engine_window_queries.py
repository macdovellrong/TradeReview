import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import pandas as pd

from engine.data_engine import DataEngine


class DataEngineWindowQueryTests(unittest.TestCase):
    def test_duckdb_load_keeps_tick_dataframe_unloaded_and_exposes_metadata(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "sample.duckdb"
            con = duckdb.connect(str(db_path))
            con.execute(
                "CREATE TABLE ticks AS SELECT * FROM (VALUES "
                "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 1.0),"
                "(TIMESTAMP '2026-04-01 09:01:00', 101.0, 2.0)"
                ") AS t(timestamp, price, volume)"
            )
            con.execute(
                "CREATE TABLE candles_1m AS SELECT * FROM (VALUES "
                "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 101.0, 99.0, 100.5, 1.0),"
                "(TIMESTAMP '2026-04-01 09:01:00', 100.5, 102.0, 100.0, 101.5, 2.0)"
                ") AS t(timestamp, open, high, low, close, volume)"
            )
            con.close()

            engine = DataEngine(parquet_file=str(db_path))

            self.assertIsNone(engine.df_ticks)
            self.assertEqual(engine.tick_count, 2)
            self.assertEqual(engine.tick_start, pd.Timestamp("2026-04-01 09:00:00"))
            self.assertEqual(engine.tick_end, pd.Timestamp("2026-04-01 09:01:00"))

    def test_get_candles_window_reads_only_requested_time_range(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "sample.duckdb"
            con = duckdb.connect(str(db_path))
            con.execute(
                "CREATE TABLE ticks AS SELECT * FROM (VALUES "
                "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 1.0)"
                ") AS t(timestamp, price, volume)"
            )
            con.execute(
                "CREATE TABLE candles_1m AS SELECT * FROM (VALUES "
                "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 101.0, 99.0, 100.5, 1.0),"
                "(TIMESTAMP '2026-04-01 09:01:00', 100.5, 102.0, 100.0, 101.5, 2.0),"
                "(TIMESTAMP '2026-04-01 09:02:00', 101.5, 103.0, 101.0, 102.5, 3.0)"
                ") AS t(timestamp, open, high, low, close, volume)"
            )
            con.close()

            engine = DataEngine(parquet_file=str(db_path))
            df = engine.get_candles_window(
                "1min",
                pd.Timestamp("2026-04-01 09:01:00"),
                pd.Timestamp("2026-04-01 09:02:00"),
            )

            self.assertEqual(list(df["close"]), [101.5, 102.5])

    def test_get_candles_still_loads_precomputed_duckdb_table(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "sample.duckdb"
            con = duckdb.connect(str(db_path))
            con.execute(
                "CREATE TABLE ticks AS SELECT * FROM (VALUES "
                "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 1.0)"
                ") AS t(timestamp, price, volume)"
            )
            con.execute(
                "CREATE TABLE candles_1m AS SELECT * FROM (VALUES "
                "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 101.0, 99.0, 100.5, 1.0),"
                "(TIMESTAMP '2026-04-01 09:01:00', 100.5, 102.0, 100.0, 101.5, 2.0)"
                ") AS t(timestamp, open, high, low, close, volume)"
            )
            con.close()

            engine = DataEngine(parquet_file=str(db_path))
            df = engine.get_candles("1min")

            self.assertEqual(list(df["close"]), [100.5, 101.5])

    def test_get_candles_window_returns_empty_frame_for_empty_range(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "sample.duckdb"
            con = duckdb.connect(str(db_path))
            con.execute(
                "CREATE TABLE ticks AS SELECT * FROM (VALUES "
                "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 1.0)"
                ") AS t(timestamp, price, volume)"
            )
            con.execute(
                "CREATE TABLE candles_1m AS SELECT * FROM (VALUES "
                "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 101.0, 99.0, 100.5, 1.0)"
                ") AS t(timestamp, open, high, low, close, volume)"
            )
            con.close()

            engine = DataEngine(parquet_file=str(db_path))
            df = engine.get_candles_window(
                "1min",
                pd.Timestamp("2026-04-02 09:01:00"),
                pd.Timestamp("2026-04-02 09:02:00"),
            )

            self.assertTrue(df.empty)
            self.assertEqual(df.index.name, "timestamp")


if __name__ == "__main__":
    unittest.main()
