import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd
from PyQt6.QtCore import QSettings


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.session_state import SessionState, load_session_state, save_session_state


class SessionStateTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = QSettings(str(Path(tmpdir) / "session.ini"), QSettings.Format.IniFormat)
            state = SessionState(
                db_path=r"\\10.0.0.23\code\gold\TradeReview\data\sample.duckdb",
                center_time=pd.Timestamp("2026-03-15 16:59:00", tz="America/New_York"),
            )

            save_session_state(settings, state)
            loaded = load_session_state(settings)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.db_path, state.db_path)
        self.assertEqual(loaded.center_time, state.center_time)

    def test_load_returns_none_when_state_is_missing_or_invalid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = QSettings(str(Path(tmpdir) / "session.ini"), QSettings.Format.IniFormat)
            self.assertIsNone(load_session_state(settings))

            settings.setValue("session/db_path", "x.duckdb")
            settings.setValue("session/center_time", "not-a-timestamp")
            settings.sync()

            self.assertIsNone(load_session_state(settings))


if __name__ == "__main__":
    unittest.main()
