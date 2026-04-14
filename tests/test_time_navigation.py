import unittest
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.time_navigation import clamp_timestamp, normalize_jump_timestamp, resolve_chart_target


class TimeNavigationTests(unittest.TestCase):
    def test_normalize_jump_timestamp_rounds_to_minute(self):
        ts = pd.Timestamp("2026-03-15 16:59:57.914000", tz="America/New_York")

        normalized = normalize_jump_timestamp(ts)

        self.assertEqual(normalized, pd.Timestamp("2026-03-15 16:59:00", tz="America/New_York"))

    def test_clamp_timestamp_limits_to_available_range(self):
        start = pd.Timestamp("2026-03-15 16:00:00", tz="America/New_York")
        end = pd.Timestamp("2026-03-15 17:00:00", tz="America/New_York")

        self.assertEqual(clamp_timestamp(pd.Timestamp("2026-03-15 15:50:00", tz="America/New_York"), start, end), start)
        self.assertEqual(clamp_timestamp(pd.Timestamp("2026-03-15 17:10:00", tz="America/New_York"), start, end), end)

    def test_resolve_chart_target_uses_right_side_bar_and_close_price(self):
        index = pd.to_datetime(
            [
                "2026-03-15 16:58:00",
                "2026-03-15 16:59:00",
                "2026-03-15 17:00:00",
            ]
        )
        df = pd.DataFrame(
            {"close": [3000.1, 3001.2, 3002.3]},
            index=index,
        )

        idx, close = resolve_chart_target(df, pd.Timestamp("2026-03-15 16:58:30", tz="America/New_York"))

        self.assertEqual(idx, 1)
        self.assertEqual(close, 3001.2)


if __name__ == "__main__":
    unittest.main()
