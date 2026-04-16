import datetime
import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.specs import normalize_drawing_spec
from ui.drawings.tools import DrawingSession, TOOL_DEFINITIONS


class DrawingToolsTests(unittest.TestCase):
    def test_normalize_drawing_spec_supports_legacy_two_point_payload(self):
        spec = normalize_drawing_spec(
            {
                "type": "fib",
                "p1_dt": datetime.datetime(2026, 4, 16, 9, 30),
                "p1_price": 100.0,
                "p2_dt": datetime.datetime(2026, 4, 16, 10, 0),
                "p2_price": 120.0,
            }
        )

        self.assertEqual(len(spec["points"]), 2)
        self.assertEqual(spec["points"][0]["price"], 100.0)
        self.assertEqual(spec["points"][1]["price"], 120.0)

    def test_line_session_completes_after_two_points(self):
        session = DrawingSession(TOOL_DEFINITIONS["line"])

        self.assertIsNone(session.add_point(datetime.datetime(2026, 4, 16, 9, 30), 100.0))
        spec = session.add_point(datetime.datetime(2026, 4, 16, 10, 0), 110.0)

        self.assertEqual(spec["type"], "line")
        self.assertEqual(len(spec["points"]), 2)

    def test_fib_extension_session_snapshots_levels_after_third_point(self):
        session = DrawingSession(TOOL_DEFINITIONS["fib_ext"], config_snapshot={"levels": [1.0, 1.618]})
        session.add_point(datetime.datetime(2026, 4, 16, 9, 30), 100.0)
        session.add_point(datetime.datetime(2026, 4, 16, 10, 0), 120.0)

        spec = session.add_point(datetime.datetime(2026, 4, 16, 10, 30), 110.0)

        self.assertEqual(spec["config_snapshot"]["levels"], [1.0, 1.618])
        self.assertEqual(len(spec["points"]), 3)

    def test_fib_extension_preview_builds_projection_after_second_point(self):
        session = DrawingSession(TOOL_DEFINITIONS["fib_ext"], config_snapshot={"levels": [1.0, 1.618]})
        session.add_point(datetime.datetime(2026, 4, 16, 9, 30), 100.0)
        session.add_point(datetime.datetime(2026, 4, 16, 10, 0), 120.0)

        spec = session.build_preview_spec(datetime.datetime(2026, 4, 16, 10, 30), 110.0)

        self.assertEqual(spec["type"], "fib_ext")
        self.assertEqual(spec["config_snapshot"]["levels"], [1.0, 1.618])
        self.assertEqual(len(spec["points"]), 3)


if __name__ == "__main__":
    unittest.main()
