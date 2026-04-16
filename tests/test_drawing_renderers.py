import datetime
import os
import unittest
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.renderers import build_render_plan, render_spec_items


APP = QApplication.instance() or QApplication([])


class FakePlotItem:
    def __init__(self):
        self.items = []

    def addItem(self, item, ignoreBounds=False):
        self.items.append((item, ignoreBounds))


class DrawingRenderPlanTests(unittest.TestCase):
    def test_build_render_plan_for_fib_uses_snapshot_levels(self):
        plan = build_render_plan(
            {
                "id": 1,
                "type": "fib",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 16, 9, 30), "price": 100.0},
                    {"dt": datetime.datetime(2026, 4, 16, 10, 0), "price": 120.0},
                ],
                "config_snapshot": {"levels": [0.5, 0.618]},
            }
        )

        self.assertEqual(plan["levels"], [0.5, 0.618])
        self.assertEqual([round(row["price"], 3) for row in plan["rows"]], [110.0, 107.64])

    def test_build_render_plan_for_fib_extension_uses_three_points(self):
        plan = build_render_plan(
            {
                "id": 2,
                "type": "fib_ext",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 16, 9, 30), "price": 100.0},
                    {"dt": datetime.datetime(2026, 4, 16, 10, 0), "price": 120.0},
                    {"dt": datetime.datetime(2026, 4, 16, 10, 30), "price": 110.0},
                ],
                "config_snapshot": {"levels": [1.0]},
            }
        )

        self.assertEqual([round(row["price"], 3) for row in plan["rows"]], [130.0])

    def test_render_spec_items_for_fib_creates_boundaries_levels_and_labels(self):
        plot = FakePlotItem()

        items = render_spec_items(
            plot,
            {
                "id": 7,
                "type": "fib",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 16, 9, 30), "price": 100.0},
                    {"dt": datetime.datetime(2026, 4, 16, 10, 0), "price": 120.0},
                ],
                "config_snapshot": {"levels": [0.5, 0.618]},
            },
            x_from_datetime=lambda dt: 10 if dt.minute == 30 else 20,
        )

        self.assertEqual(len(items), 6)
        self.assertEqual(len(plot.items), 6)
        self.assertTrue(all(getattr(item, "_drawing_id", None) == 7 for item in items))

    def test_render_spec_items_for_fib_ext_includes_guides_level_and_label(self):
        plot = FakePlotItem()

        items = render_spec_items(
            plot,
            {
                "id": 9,
                "type": "fib_ext",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 16, 9, 30), "price": 100.0},
                    {"dt": datetime.datetime(2026, 4, 16, 10, 0), "price": 120.0},
                    {"dt": datetime.datetime(2026, 4, 16, 10, 30), "price": 110.0},
                ],
                "config_snapshot": {"levels": [1.0]},
            },
            x_from_datetime=lambda dt: {30: 10, 0: 20}[dt.minute],
        )

        self.assertEqual(len(items), 4)
        self.assertEqual(len(plot.items), 4)
        self.assertTrue(all(getattr(item, "_drawing_id", None) == 9 for item in items))


if __name__ == "__main__":
    unittest.main()
