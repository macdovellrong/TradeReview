import unittest

import pandas as pd


class FakeDataEngine:
    def __init__(
        self,
        df_ticks=None,
        error=None,
        warnings=None,
        duckdb_path=None,
        tick_count=0,
        tick_start=None,
    ):
        self.parquet_file = None
        self.df_ticks = df_ticks
        self._duckdb_path = duckdb_path
        self.tick_count = tick_count
        self.tick_start = tick_start
        self.last_load_error = error
        self.last_load_warnings = warnings or []
        self.loaded = False

    def load_data(self):
        self.loaded = True


class DataLoadingFacadeTests(unittest.TestCase):
    def test_load_returns_initial_time_after_100000_ticks_for_large_dataset(self):
        from ui.services.data_loading import DataLoadingFacade

        index = pd.date_range("2026-04-24", periods=100001, freq="s", tz="America/New_York")
        engine = FakeDataEngine(pd.DataFrame({"price": range(len(index)), "volume": 1}, index=index))
        facade = DataLoadingFacade(engine)

        result = facade.load("sample.duckdb")

        self.assertTrue(result.success)
        self.assertEqual(result.initial_time, index[100000])
        self.assertEqual(result.warnings, ())

    def test_load_returns_error_when_engine_has_no_ticks(self):
        from ui.services.data_loading import DataLoadingFacade

        engine = FakeDataEngine(df_ticks=None, error="bad file")
        facade = DataLoadingFacade(engine)

        result = facade.load("bad.duckdb")

        self.assertFalse(result.success)
        self.assertEqual(result.error, "bad file")

    def test_load_uses_duckdb_metadata_when_ticks_are_not_materialized(self):
        from ui.services.data_loading import DataLoadingFacade

        initial_time = pd.Timestamp("2026-04-01 09:00:00")
        engine = FakeDataEngine(
            df_ticks=None,
            duckdb_path="sample.duckdb",
            tick_count=2,
            tick_start=initial_time,
            warnings=["metadata only"],
        )
        facade = DataLoadingFacade(engine)

        result = facade.load("sample.duckdb")

        self.assertTrue(result.success)
        self.assertEqual(result.initial_time, initial_time)
        self.assertEqual(result.warnings, ("metadata only",))


if __name__ == "__main__":
    unittest.main()
