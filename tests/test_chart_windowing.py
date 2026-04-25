import unittest

import pandas as pd

from ui.chart_windowing import (
    build_query_window,
    is_view_inside_loaded_window,
    should_prefetch_window,
)


class ChartWindowingTests(unittest.TestCase):
    def test_build_query_window_adds_buffer_on_both_sides(self):
        start = pd.Timestamp("2026-04-01 10:00:00")
        end = pd.Timestamp("2026-04-01 11:00:00")

        query_start, query_end = build_query_window(start, end, buffer_multiplier=2)

        self.assertEqual(query_start, pd.Timestamp("2026-04-01 08:00:00"))
        self.assertEqual(query_end, pd.Timestamp("2026-04-01 13:00:00"))

    def test_view_inside_loaded_window_returns_true(self):
        self.assertTrue(
            is_view_inside_loaded_window(
                pd.Timestamp("2026-04-01 10:00:00"),
                pd.Timestamp("2026-04-01 11:00:00"),
                pd.Timestamp("2026-04-01 08:00:00"),
                pd.Timestamp("2026-04-01 13:00:00"),
            )
        )

    def test_prefetch_when_view_nears_right_edge(self):
        self.assertTrue(
            should_prefetch_window(
                pd.Timestamp("2026-04-01 11:30:00"),
                pd.Timestamp("2026-04-01 12:30:00"),
                pd.Timestamp("2026-04-01 08:00:00"),
                pd.Timestamp("2026-04-01 13:00:00"),
                edge_fraction=0.5,
            )
        )


if __name__ == "__main__":
    unittest.main()
