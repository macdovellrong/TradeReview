import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class UIModuleBoundaryTests(unittest.TestCase):
    def test_chart_classes_have_dedicated_modules_and_compat_exports(self):
        from ui.chart_primitives import CandlestickItem, MockYScale, TimeAxisItem
        from ui.chart_widget import ChartWidget
        from ui.chart_window import FloatingChartWindow
        from ui.main_window import (
            CandlestickItem as CompatCandlestickItem,
            ChartWidget as CompatChartWidget,
            FloatingChartWindow as CompatFloatingChartWindow,
            MockYScale as CompatMockYScale,
            TimeAxisItem as CompatTimeAxisItem,
        )

        self.assertIs(CompatCandlestickItem, CandlestickItem)
        self.assertIs(CompatTimeAxisItem, TimeAxisItem)
        self.assertIs(CompatMockYScale, MockYScale)
        self.assertIs(CompatChartWidget, ChartWidget)
        self.assertIs(CompatFloatingChartWindow, FloatingChartWindow)
        self.assertEqual(FloatingChartWindow.__module__, "ui.chart_window")


if __name__ == "__main__":
    unittest.main()
