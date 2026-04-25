import unittest

import pandas as pd

from ui.chart_lod import choose_lod_period


class ChartLODTests(unittest.TestCase):
    def test_short_intraday_range_keeps_current_low_period(self):
        period = choose_lod_period(
            requested_period="1min",
            view_start=pd.Timestamp("2026-04-01 09:00:00"),
            view_end=pd.Timestamp("2026-04-01 15:00:00"),
            pixel_width=1600,
        )

        self.assertEqual(period, "1min")

    def test_multi_month_range_uses_hourly_period(self):
        period = choose_lod_period(
            requested_period="1min",
            view_start=pd.Timestamp("2026-01-01"),
            view_end=pd.Timestamp("2026-04-01"),
            pixel_width=1600,
        )

        self.assertEqual(period, "1h")

    def test_multi_year_range_uses_daily_period(self):
        period = choose_lod_period(
            requested_period="30s",
            view_start=pd.Timestamp("2021-01-01"),
            view_end=pd.Timestamp("2026-01-01"),
            pixel_width=1600,
        )

        self.assertEqual(period, "1D")


if __name__ == "__main__":
    unittest.main()
