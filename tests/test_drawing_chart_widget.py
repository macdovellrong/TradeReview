import os
import unittest
from pathlib import Path
import sys

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.fib_config import FibLevelsConfig, FibSettings
from ui.chart_widget import ChartWidget


APP = QApplication.instance() or QApplication([])


class ChartWidgetDrawingTests(unittest.TestCase):
    def test_macd_rsi_button_toggles_indicator_panels(self):
        chart = ChartWidget("1min")

        self.assertTrue(chart.ax_macd.isVisible())
        self.assertTrue(chart.ax_rsi.isVisible())

        chart.btn_toggle_macd_rsi.click()

        self.assertFalse(chart.ax_macd.isVisible())
        self.assertFalse(chart.ax_rsi.isVisible())

        chart.btn_toggle_macd_rsi.click()

        self.assertTrue(chart.ax_macd.isVisible())
        self.assertTrue(chart.ax_rsi.isVisible())

    def test_bollinger_button_toggles_band_visibility(self):
        chart = ChartWidget("1min")
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
                "BB_Upper": [103.0, 104.0, 105.0],
                "BB_Lower": [97.0, 98.0, 99.0],
            },
            index=index,
        )

        chart.update_chart(df)

        self.assertIn("BB_Upper", chart.indicator_items)
        self.assertIn("BB_Lower", chart.indicator_items)
        self.assertTrue(chart.indicator_items["BB_Upper"].isVisible())
        self.assertTrue(chart.indicator_items["BB_Lower"].isVisible())

        chart.btn_toggle_bb.click()

        self.assertFalse(chart.indicator_items["BB_Upper"].isVisible())
        self.assertFalse(chart.indicator_items["BB_Lower"].isVisible())

    def test_set_draw_mode_uses_fib_extension_snapshot(self):
        chart = ChartWidget("1min")
        chart.set_fib_settings(
            FibSettings(
                retracement=FibLevelsConfig(enabled_levels=[0.5], custom_levels_text=""),
                extension=FibLevelsConfig(enabled_levels=[1.0, 1.618], custom_levels_text="2.0"),
            )
        )

        chart.set_draw_mode("fib_ext")

        self.assertEqual(chart.active_drawing_session.config_snapshot["levels"], [1.0, 1.618, 2.0])

    def test_add_drawing_supports_point_based_fib_spec(self):
        chart = ChartWidget("1min")
        index = pd.to_datetime(["2026-04-16 09:30:00", "2026-04-16 10:00:00"])
        chart.full_df = pd.DataFrame({"close": [100.0, 120.0]}, index=index)

        chart.add_drawing(
            {
                "id": 10,
                "type": "fib",
                "points": [
                    {"dt": index[0], "price": 100.0},
                    {"dt": index[1], "price": 120.0},
                ],
                "config_snapshot": {"levels": [0.5, 0.618]},
            }
        )

        self.assertIn(10, chart.drawings)
        self.assertEqual(len(chart.drawings[10]), 6)

    def test_update_chart_window_sets_local_window_data(self):
        chart = ChartWidget("1min")
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

        chart.update_chart_window(df, auto_scale=True, highlight_idx=1)
        chart.ax.setXRange(0, 1, padding=0)

        self.assertEqual(list(chart.current_df.index), list(index))
        self.assertEqual(len(chart.current_x), 3)
        self.assertEqual(chart.get_visible_time_range(), (index[0], index[1]))

    def test_window_mode_requests_reload_when_view_moves_past_loaded_data(self):
        chart = ChartWidget("1min")
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
        received = []
        chart.sig_window_reload_requested.connect(lambda start, end: received.append((start, end)))

        chart.update_chart_window(df)
        chart.ax.setXRange(30, 32, padding=0)
        received.clear()
        chart.on_range_changed()

        self.assertEqual(len(received), 1)
        self.assertGreater(received[0][0], index[-1])

    def test_set_time_view_range_maps_dates_to_loaded_window_indices(self):
        chart = ChartWidget("1D")
        index = pd.date_range("2026-04-01", periods=10, freq="1D")
        df = pd.DataFrame(
            {
                "open": range(10),
                "close": range(10),
                "high": range(1, 11),
                "low": range(10),
                "volume": [1.0] * 10,
            },
            index=index,
        )

        chart.update_chart_window(df)
        chart.set_time_view_range(index[2], index[5])

        x_min, x_max = chart.ax.vb.viewRange()[0]
        self.assertAlmostEqual(x_min, 2.0)
        self.assertAlmostEqual(x_max, 5.0)

    def test_set_time_view_range_adds_right_padding_at_last_candle(self):
        chart = ChartWidget("1D")
        index = pd.date_range("2026-04-01", periods=10, freq="1D")
        df = pd.DataFrame(
            {
                "open": range(10),
                "close": range(10),
                "high": range(1, 11),
                "low": range(10),
                "volume": [1.0] * 10,
            },
            index=index,
        )

        chart.update_chart_window(df)
        chart.set_time_view_range(index[5], index[-1])

        _x_min, x_max = chart.ax.vb.viewRange()[0]
        self.assertGreater(x_max, len(index) - 1)

    def test_auto_scale_last_candle_padding_does_not_request_reload(self):
        chart = ChartWidget("1min")
        index = pd.date_range("2026-04-16 09:30:00", periods=10, freq="1min")
        df = pd.DataFrame(
            {
                "open": range(10),
                "close": range(10),
                "high": range(1, 11),
                "low": range(10),
                "volume": [1.0] * 10,
            },
            index=index,
        )
        received = []
        chart.sig_window_reload_requested.connect(lambda start, end: received.append((start, end)))

        chart.update_chart_window(df, auto_scale=True, highlight_idx=len(index) - 1)
        received.clear()
        chart.on_range_changed()

        _x_min, x_max = chart.ax.vb.viewRange()[0]
        self.assertGreater(x_max, len(index) - 1)
        self.assertEqual(received, [])


if __name__ == "__main__":
    unittest.main()
