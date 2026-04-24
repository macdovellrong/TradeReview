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


if __name__ == "__main__":
    unittest.main()
