import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.fib_math import build_extension_levels, build_retracement_levels


class FibMathTests(unittest.TestCase):
    def test_build_retracement_levels_uses_requested_ratios(self):
        rows = build_retracement_levels(100.0, 120.0, [0.5, 0.618, 0.786])

        self.assertEqual([row.ratio for row in rows], [0.5, 0.618, 0.786])
        self.assertEqual([round(row.price, 3) for row in rows], [110.0, 107.64, 104.28])

    def test_build_extension_levels_projects_upward(self):
        rows = build_extension_levels(100.0, 120.0, 110.0, [1.0, 1.618])

        self.assertEqual([round(row.price, 3) for row in rows], [130.0, 142.36])

    def test_build_extension_levels_projects_downward(self):
        rows = build_extension_levels(120.0, 100.0, 110.0, [1.0, 1.618])

        self.assertEqual([round(row.price, 3) for row in rows], [90.0, 77.64])


if __name__ == "__main__":
    unittest.main()
