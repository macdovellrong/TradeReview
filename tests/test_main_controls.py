import datetime
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt6.QtWidgets import QApplication

from ui.main_controls import MainControls


class MainControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def create_controls(self):
        controls = MainControls(current_time=datetime.datetime(2026, 4, 24, 9, 30))
        self.addCleanup(controls.close)
        return controls

    def test_button_clicks_emit_semantic_request_signals(self):
        controls = self.create_controls()
        events = []

        controls.load_requested.connect(lambda: events.append("load"))
        controls.reset_requested.connect(lambda: events.append("reset"))
        controls.save_view_requested.connect(lambda: events.append("save"))
        controls.pop_layout_requested.connect(lambda: events.append("pop"))
        controls.play_requested.connect(lambda: events.append("play"))
        controls.step_back_requested.connect(lambda: events.append("back"))
        controls.step_forward_requested.connect(lambda: events.append("forward"))
        controls.date_edit_finished.connect(lambda: events.append("date"))

        controls.btn_load.click()
        controls.btn_reset.click()
        controls.btn_save_view.click()
        controls.btn_detach_layout.click()
        controls.btn_play.setEnabled(True)
        controls.btn_play.click()
        controls.btn_step_back.click()
        controls.btn_step_forward.click()
        controls.date_edit.editingFinished.emit()

        self.assertEqual(
            events,
            ["load", "reset", "save", "pop", "play", "back", "forward", "date"],
        )

    def test_value_controls_emit_selected_values(self):
        controls = self.create_controls()
        events = {}

        controls.layout_changed.connect(lambda value: events.setdefault("layout", []).append(value))
        controls.chart_count_changed.connect(lambda value: events.setdefault("count", []).append(value))
        controls.replay_mode_changed.connect(lambda value: events.setdefault("replay", []).append(value))
        controls.speed_changed.connect(lambda value: events.setdefault("speed", []).append(value))

        controls.combo_layout.setCurrentText("Grid 2x2")
        controls.combo_chart_count.setCurrentText("2")
        controls.chk_replay.setChecked(True)
        next(button for button in controls.speed_btn_group.buttons() if button.text() == "120x").click()

        self.assertEqual(events["layout"][-1], "Grid 2x2")
        self.assertEqual(events["count"][-1], "2")
        self.assertTrue(events["replay"][-1] > 0)
        self.assertEqual(events["speed"][-1], 120)


if __name__ == "__main__":
    unittest.main()
