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


if __name__ == "__main__":
    unittest.main()
