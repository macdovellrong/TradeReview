import unittest
from pathlib import Path
from unittest.mock import patch
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


class MainWindowCrosshairSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def create_window(self):
        with patch("ui.main_window.QTimer.singleShot", lambda *args, **kwargs: None):
            window = MainWindow()
        window.timer.stop()
        self.addCleanup(window.close)
        self.addCleanup(lambda: window.floating_windows.clear())
        return window

    def test_window_title_is_tradereview(self):
        window = self.create_window()

        self.assertEqual(window.windowTitle(), "TradeReview")

    def test_attached_chart_emits_crosshair_to_detached_chart(self):
        window = self.create_window()
        source_chart = window.charts[0]
        target_chart = window.charts[1]

        window.detach_chart(target_chart, refresh_layout=False)
        self.assertTrue(target_chart.is_detached)

        with patch.object(target_chart, "sync_crosshair") as sync_crosshair:
            source_chart.sig_mouse_moved_with_price.emit(123.0, 456.0)

        sync_crosshair.assert_called_once_with(123.0, 456.0)

    def test_detached_chart_emits_crosshair_back_to_attached_chart(self):
        window = self.create_window()
        source_chart = window.charts[0]
        target_chart = window.charts[1]

        window.detach_chart(source_chart, refresh_layout=False)
        self.assertTrue(source_chart.is_detached)

        with patch.object(target_chart, "sync_crosshair") as sync_crosshair:
            source_chart.sig_mouse_moved_with_price.emit(789.0, 321.0)

        sync_crosshair.assert_called_once_with(789.0, 321.0)


if __name__ == "__main__":
    unittest.main()
