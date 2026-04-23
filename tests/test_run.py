import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run


class ConfigureQtPlatformTests(unittest.TestCase):
    def test_windows_defaults_to_system_dpi_awareness(self):
        env = {}

        value = run.configure_qt_platform(env=env, platform_name="nt")

        self.assertEqual("windows:dpiawareness=1", value)
        self.assertEqual("windows:dpiawareness=1", env["QT_QPA_PLATFORM"])

    def test_existing_qt_platform_is_preserved(self):
        env = {"QT_QPA_PLATFORM": "offscreen"}

        value = run.configure_qt_platform(env=env, platform_name="nt")

        self.assertEqual("offscreen", value)
        self.assertEqual("offscreen", env["QT_QPA_PLATFORM"])

    def test_non_windows_does_not_override_platform(self):
        env = {}

        value = run.configure_qt_platform(env=env, platform_name="posix")

        self.assertIsNone(value)
        self.assertNotIn("QT_QPA_PLATFORM", env)


if __name__ == "__main__":
    unittest.main()
