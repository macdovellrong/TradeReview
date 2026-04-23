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


class MainWindowReplayControllerTests(unittest.TestCase):
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
        index = pd.date_range("2026-04-24 09:30:00", periods=3, freq="1min", tz="UTC")
        window.engine.df_ticks = pd.DataFrame({"price": [100.0, 101.0, 102.0]}, index=index)
        window.current_time = index[0]
        window._update_date_edit_bounds()
        window._set_date_edit(window.current_time)
        return index

    def test_on_mode_change_enables_controller_and_play_button(self):
        window = self.create_window()

        with patch.object(window, "refresh_all_charts"), patch.object(window, "_ensure_replay_engine"):
            window.chk_replay.setChecked(True)

        self.assertTrue(window.replay_controller.enabled)
        self.assertFalse(window.is_playing)
        self.assertTrue(window.btn_play.isEnabled())
        self.assertEqual(window.btn_play.text(), "Play")

    def test_toggle_play_and_set_speed_keep_window_and_controller_in_sync(self):
        window = self.create_window()

        with patch.object(window, "refresh_all_charts"), patch.object(window, "_ensure_replay_engine"):
            window.chk_replay.setChecked(True)

        window.set_speed(120)
        self.assertEqual(window.replay_speed, 120)
        self.assertEqual(window.replay_controller.speed, 120)

        window.toggle_play()
        self.assertTrue(window.is_playing)
        self.assertTrue(window.replay_controller.is_playing)
        self.assertEqual(window.btn_play.text(), "Pause")

        window.toggle_play()
        self.assertFalse(window.is_playing)
        self.assertFalse(window.replay_controller.is_playing)
        self.assertEqual(window.btn_play.text(), "Play")

    def test_on_timer_tick_stops_playback_when_replay_reaches_end(self):
        window = self.create_window()
        index = self.attach_ticks(window)
        window.replay_engine.tick_pos = len(index) - 1

        def advance_to(target_time):
            window.replay_controller.current_time = index[-1]
            return index[-1]

        with patch.object(window, "refresh_all_charts"), patch.object(window, "_ensure_replay_engine"):
            window.chk_replay.setChecked(True)

        window.is_playing = True
        window.replay_controller.is_playing = True

        with patch.object(window, "refresh_all_charts"), patch.object(window, "_ensure_replay_engine"), patch.object(
            window.replay_controller,
            "advance_to",
            side_effect=advance_to,
        ):
            window.on_timer_tick()

        self.assertFalse(window.is_playing)
        self.assertFalse(window.replay_controller.is_playing)
        self.assertEqual(window.btn_play.text(), "Play")
        self.assertEqual(window.current_time, index[-1])


if __name__ == "__main__":
    unittest.main()
