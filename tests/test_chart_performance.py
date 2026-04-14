import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.chart_performance import (
    build_visible_slice_window,
    get_crosshair_sync_targets,
    should_refresh_visible_slice,
)


class DummyChart:
    def __init__(self, name, is_detached=False):
        self.name = name
        self.is_detached = is_detached


class ChartPerformanceTests(unittest.TestCase):
    def test_small_pan_within_buffer_does_not_require_refresh(self):
        should_refresh = should_refresh_visible_slice(
            view_min=101,
            view_max=201,
            total_len=10_000,
            last_slice_start=0,
            last_slice_end=1_200,
            padding=1_000,
        )

        self.assertFalse(should_refresh)

    def test_refresh_is_required_when_view_nears_slice_edge(self):
        should_refresh = should_refresh_visible_slice(
            view_min=760,
            view_max=951,
            total_len=10_000,
            last_slice_start=0,
            last_slice_end=1_200,
            padding=1_000,
        )

        self.assertTrue(should_refresh)

    def test_slice_window_clamps_to_available_data(self):
        slice_start, slice_end = build_visible_slice_window(
            view_min=50,
            view_max=150,
            total_len=500,
            padding=1_000,
        )

        self.assertEqual((slice_start, slice_end), (0, 500))

    def test_crosshair_sync_targets_skip_source_and_detached_charts(self):
        source = DummyChart("source")
        attached = DummyChart("attached")
        detached = DummyChart("detached", is_detached=True)

        targets = get_crosshair_sync_targets([source, attached, detached], source)

        self.assertEqual(targets, [attached])


if __name__ == "__main__":
    unittest.main()
