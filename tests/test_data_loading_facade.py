import unittest

import pandas as pd


class FakeDataEngine:
    def __init__(self, df_ticks=None, error=None, warnings=None):
        self.parquet_file = None
        self.df_ticks = df_ticks
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


if __name__ == "__main__":
    unittest.main()
