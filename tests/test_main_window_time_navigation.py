import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


class MainWindowTimeNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def create_window(self):
        with patch("ui.main_window.QTimer.singleShot", lambda *args, **kwargs: None):
            window = MainWindow()
        window.timer.stop()
        self.addCleanup(window.close)
        return window

    def attach_ticks(self, window):
        index = pd.date_range(
            "2026-04-16 09:30:00",
            periods=6,
            freq="1min",
            tz="UTC",
        )
        window.engine.df_ticks = pd.DataFrame(
            {"price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]},
            index=index,
        )
        window.current_time = index[2]
        window._update_date_edit_bounds()
        window._set_date_edit(window.current_time)
        return index

    def test_jump_to_time_clamps_to_loaded_tick_range(self):
        window = self.create_window()
        index = self.attach_ticks(window)

        with patch.object(window, "refresh_all_charts"), patch.object(window, "_center_charts_on_time"):
            window.jump_to_time(index[-1] + pd.Timedelta(minutes=10))

        self.assertEqual(window.current_time, index[-1])

    def test_on_step_forward_uses_selected_combo_step(self):
        window = self.create_window()
        index = self.attach_ticks(window)
        window.current_time = index[1]
        window.combo_step.setCurrentText("5m")

        with patch.object(window, "refresh_all_charts"):
            window.on_step_forward()

        self.assertEqual(window.current_time, index[-1])

    def test_refresh_single_chart_uses_window_query_for_duckdb_metadata_mode(self):
        window = self.create_window()
        chart = window.charts[0]
        index = pd.to_datetime(
            ["2026-04-16 09:30:00", "2026-04-16 09:31:00", "2026-04-16 09:32:00"]
        )
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "close": [101.0, 102.0, 103.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "volume": [1.0, 1.0, 1.0],
            },
            index=index,
        )

        class FakeDuckDBEngine:
            df_ticks = None
            _duckdb_path = "sample.duckdb"

            def __init__(self, frame):
                self.frame = frame
                self.calls = []

            def get_candles_window(self, period, start_time, end_time):
                self.calls.append((period, start_time, end_time))
                return self.frame

        engine = FakeDuckDBEngine(df)
        window.engine = engine
        window.current_time = pd.Timestamp("2026-04-16 09:31:00", tz="UTC")
        window.chk_replay.setChecked(False)

        with patch.object(chart, "update_chart_window", create=True) as update_chart_window:
            window.refresh_single_chart(chart, auto_scale=True)

        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0][0], chart.current_period)
        update_chart_window.assert_called_once()
        self.assertEqual(update_chart_window.call_args.kwargs["highlight_idx"], 1)

    def test_reload_chart_window_queries_buffered_range(self):
        window = self.create_window()
        chart = window.charts[0]
        index = pd.to_datetime(
            ["2026-04-16 09:30:00", "2026-04-16 09:31:00", "2026-04-16 09:32:00"]
        )
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "close": [101.0, 102.0, 103.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "volume": [1.0, 1.0, 1.0],
            },
            index=index,
        )

        class FakeDuckDBEngine:
            _duckdb_path = "sample.duckdb"

            def __init__(self, frame):
                self.frame = frame
                self.calls = []

            def get_candles_window(self, period, start_time, end_time):
                self.calls.append((period, start_time, end_time))
                return self.frame

        engine = FakeDuckDBEngine(df)
        window.engine = engine

        with patch.object(chart, "update_chart_window") as update_chart_window:
            window.reload_chart_window(
                chart,
                pd.Timestamp("2026-04-16 10:00:00"),
                pd.Timestamp("2026-04-16 11:00:00"),
            )

        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(engine.calls[0][0], chart.current_period)
        self.assertEqual(engine.calls[0][1], pd.Timestamp("2026-04-16 08:00:00"))
        self.assertEqual(engine.calls[0][2], pd.Timestamp("2026-04-16 13:00:00"))
        update_chart_window.assert_called_once_with(df, auto_scale=False)

    def test_reload_chart_window_uses_lod_period_for_wide_view(self):
        window = self.create_window()
        chart = window.charts[0]
        chart.set_period("1min")
        index = pd.date_range("2021-01-01", periods=10, freq="1D")
        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "close": [101.0] * 10,
                "high": [102.0] * 10,
                "low": [99.0] * 10,
                "volume": [1.0] * 10,
            },
            index=index,
        )

        class FakeDuckDBEngine:
            _duckdb_path = "sample.duckdb"

            def __init__(self, frame):
                self.frame = frame
                self.calls = []

            def get_candles_window(self, period, start_time, end_time):
                self.calls.append((period, start_time, end_time))
                return self.frame

        engine = FakeDuckDBEngine(df)
        window.engine = engine

        with patch.object(chart, "update_chart_window") as update_chart_window:
            window.reload_chart_window(
                chart,
                pd.Timestamp("2021-01-01"),
                pd.Timestamp("2026-01-01"),
            )

        self.assertEqual(engine.calls[0][0], "1D")
        self.assertEqual(chart.active_display_period, "1D")
        update_chart_window.assert_called_once_with(df, auto_scale=False)


if __name__ == "__main__":
    unittest.main()
