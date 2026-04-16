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
from ui.main_window import ChartWidget


APP = QApplication.instance() or QApplication([])


class ChartWidgetDrawingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
