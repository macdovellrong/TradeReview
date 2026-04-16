import tempfile
import unittest
from pathlib import Path
import sys

from PyQt6.QtCore import QSettings


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.fib_config import (
    DEFAULT_EXTENSION_PRESETS,
    DEFAULT_RETRACEMENT_PRESETS,
    FibLevelsConfig,
    FibSettings,
    load_fib_settings,
    merge_fib_levels,
    save_fib_settings,
)


class FibConfigTests(unittest.TestCase):
    def test_merge_fib_levels_merges_presets_and_custom_values(self):
        levels = merge_fib_levels([0.5, 0.618], "0.618, 0.786, 0.8")

        self.assertEqual(levels, [0.5, 0.618, 0.786, 0.8])

    def test_merge_fib_levels_rejects_invalid_tokens(self):
        with self.assertRaises(ValueError):
            merge_fib_levels([0.5], "0.7,abc")

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = QSettings(str(Path(tmpdir) / "fib.ini"), QSettings.Format.IniFormat)
            state = FibSettings(
                retracement=FibLevelsConfig(enabled_levels=[0.5, 0.618], custom_levels_text="0.786"),
                extension=FibLevelsConfig(enabled_levels=[1.0, 1.618], custom_levels_text="2.0"),
            )

            save_fib_settings(settings, state)
            loaded = load_fib_settings(settings)

        self.assertEqual(loaded.retracement.effective_levels, [0.5, 0.618, 0.786])
        self.assertEqual(loaded.extension.effective_levels, [1.0, 1.618, 2.0])

    def test_defaults_match_requested_levels(self):
        self.assertEqual(DEFAULT_RETRACEMENT_PRESETS, [0.236, 0.382, 0.5, 0.618, 0.7, 0.786, 0.8])
        self.assertEqual(DEFAULT_EXTENSION_PRESETS, [0.618, 1.0, 1.272, 1.618, 2.0])


if __name__ == "__main__":
    unittest.main()
