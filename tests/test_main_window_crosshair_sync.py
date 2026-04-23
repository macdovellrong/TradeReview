import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import os


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ui.chart_window import FloatingChartWindow
from ui.main_window import MainWindow


class MainWindowCrosshairSyncTests(unittest.TestCase):
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

    def test_window_title_is_tradereview(self):
        window = self.create_window()

        self.assertEqual(window.windowTitle(), "TradeReview")

    def test_attached_chart_emits_crosshair_to_detached_chart(self):
        window = self.create_window()
        source_chart = window.charts[0]
        target_chart = window.charts[1]

        window.detach_chart(target_chart, refresh_layout=False)
        self.assertTrue(target_chart.is_detached)
        self.assertIsInstance(window.floating_windows[0], FloatingChartWindow)
        self.assertEqual(window.floating_windows[0].__class__.__module__, "ui.chart_window")

        with patch.object(target_chart, "sync_crosshair") as sync_crosshair:
            source_chart.sig_mouse_moved_with_price.emit(123.0, 456.0)

        sync_crosshair.assert_called_once_with(123.0, 456.0)

    def test_chart_mouse_move_delegates_to_controller(self):
        window = self.create_window()
        source_chart = window.charts[0]

        with patch.object(window.crosshair_sync_controller, "sync_from") as sync_from:
            source_chart.sig_mouse_moved_with_price.emit(222.0, 333.0)

        sync_from.assert_called_once_with(source_chart, 222.0, 333.0)

    def test_detached_chart_emits_crosshair_back_to_attached_chart(self):
        window = self.create_window()
        source_chart = window.charts[0]
        target_chart = window.charts[1]

        window.detach_chart(source_chart, refresh_layout=False)
        self.assertTrue(source_chart.is_detached)

        with patch.object(target_chart, "sync_crosshair") as sync_crosshair:
            source_chart.sig_mouse_moved_with_price.emit(789.0, 321.0)

        sync_crosshair.assert_called_once_with(789.0, 321.0)

    def test_disabled_charts_do_not_receive_crosshair_sync(self):
        window = self.create_window()
        window.combo_chart_count.setCurrentText("2")
        window.on_chart_count_changed("2")

        source_chart = window.charts[0]
        enabled_target = window.charts[1]
        disabled_chart_1 = window.charts[2]
        disabled_chart_2 = window.charts[3]

        with (
            patch.object(enabled_target, "sync_crosshair") as enabled_sync,
            patch.object(disabled_chart_1, "sync_crosshair") as disabled_sync_1,
            patch.object(disabled_chart_2, "sync_crosshair") as disabled_sync_2,
        ):
            source_chart.sig_mouse_moved_with_price.emit(111.0, 222.0)

        enabled_sync.assert_called_once_with(111.0, 222.0)
        disabled_sync_1.assert_not_called()
        disabled_sync_2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
