import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.crosshair_sync import CrosshairSyncController


class DummyChart:
    def __init__(self, name, is_detached=False):
        self.name = name
        self.is_detached = is_detached
        self.sync_calls = []

    def sync_crosshair(self, timestamp, price):
        self.sync_calls.append((timestamp, price))


class CrosshairSyncControllerTests(unittest.TestCase):
    def test_sync_from_skips_source_but_includes_detached_targets(self):
        controller = CrosshairSyncController()
        source = DummyChart("source")
        attached = DummyChart("attached")
        detached = DummyChart("detached", is_detached=True)

        controller.register_chart(source)
        controller.register_chart(attached)
        controller.register_chart(detached)

        controller.sync_from(source, "2026-04-19T09:30:00", 123.45)

        self.assertEqual(source.sync_calls, [])
        self.assertEqual(attached.sync_calls, [("2026-04-19T09:30:00", 123.45)])
        self.assertEqual(detached.sync_calls, [("2026-04-19T09:30:00", 123.45)])

    def test_unregister_chart_removes_sync_target(self):
        controller = CrosshairSyncController()
        source = DummyChart("source")
        target = DummyChart("target")

        controller.register_chart(source)
        controller.register_chart(target)
        controller.unregister_chart(target)

        controller.sync_from(source, "2026-04-19T09:30:00", 123.45)

        self.assertEqual(source.sync_calls, [])
        self.assertEqual(target.sync_calls, [])

    def test_register_chart_deduplicates_sync_targets(self):
        controller = CrosshairSyncController()
        source = DummyChart("source")
        target = DummyChart("target")

        controller.register_chart(source)
        controller.register_chart(target)
        controller.register_chart(target)

        controller.sync_from(source, "2026-04-19T09:30:00", 123.45)

        self.assertEqual(target.sync_calls, [("2026-04-19T09:30:00", 123.45)])

    def test_iter_charts_returns_registered_charts_in_order(self):
        controller = CrosshairSyncController()
        first = DummyChart("first")
        second = DummyChart("second")

        controller.register_chart(first)
        controller.register_chart(second)

        self.assertEqual(list(controller.iter_charts()), [first, second])


if __name__ == "__main__":
    unittest.main()
