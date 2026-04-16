import os
import unittest
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.dialogs import FibConfigDialog
from ui.drawings.fib_config import FibLevelsConfig, FibSettings


APP = QApplication.instance() or QApplication([])


class FibConfigDialogTests(unittest.TestCase):
    def test_dialog_preserves_full_presets_and_builds_settings(self):
        dialog = FibConfigDialog(
            FibSettings(
                retracement=FibLevelsConfig(enabled_levels=[0.5, 0.618], custom_levels_text="0.786"),
                extension=FibLevelsConfig(enabled_levels=[1.0], custom_levels_text="1.618"),
            )
        )

        self.assertIn(0.382, dialog.retracement_checkboxes)
        dialog.retracement_checkboxes[0.382].setChecked(True)
        dialog.retracement_custom_edit.setText("0.786, 0.8")

        updated = dialog.build_settings()

        self.assertEqual(updated.retracement.effective_levels, [0.382, 0.5, 0.618, 0.786, 0.8])
