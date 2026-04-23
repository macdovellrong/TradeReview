import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


class MainWindowLayoutAndCrosshairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def create_window(self):
        with patch("ui.main_window.QTimer.singleShot", lambda *args, **kwargs: None):
            window = MainWindow()
        window.timer.stop()

        def close_floating_windows():
            for floating_window in list(window.floating_windows):
                floating_window.close()
            window.floating_windows.clear()

        self.addCleanup(window.close)
        self.addCleanup(close_floating_windows)
        return window

    def test_chart_count_limits_crosshair_registration_to_enabled_charts(self):
        window = self.create_window()

        window.combo_chart_count.setCurrentText("2")
        window.on_chart_count_changed("2")

        registered_charts = list(window.crosshair_sync_controller.iter_charts())

        self.assertEqual(registered_charts, window.charts[:2])
        self.assertNotIn(window.charts[2], registered_charts)
        self.assertNotIn(window.charts[3], registered_charts)

    def test_detach_and_close_window_round_trips_chart_attachment_state(self):
        window = self.create_window()
        chart = window.charts[1]

        self.assertFalse(chart.is_detached)
        self.assertEqual(len(window.floating_windows), 0)

        window.detach_chart(chart, refresh_layout=False)

        self.assertTrue(chart.is_detached)
        self.assertEqual(len(window.floating_windows), 1)

        floating_window = window.floating_windows[0]
        floating_window.close()

        self.assertFalse(chart.is_detached)
        self.assertEqual(len(window.floating_windows), 0)


if __name__ == "__main__":
    unittest.main()
