# Crosshair Sync Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把主窗口标题改为 `TradeReview`，并引入独立的十字同步控制器，让主界面图表与 `Pop` 浮动图表之间可以双向同步十字虚线。

**Architecture:** 新增一个轻量的 `CrosshairSyncController` 作为十字同步唯一入口，`MainWindow` 负责持有并为每个 `ChartWidget` 注册它。`ChartWidget` 继续发出已有的 `sig_mouse_moved_with_price`，但不再依赖布局派生的目标筛选逻辑；controller 只排除事件源自身，不根据 `is_detached` 做过滤。

**Tech Stack:** Python, PyQt6, pyqtgraph, unittest

---

### Task 1: Add a Dedicated Crosshair Sync Controller

**Files:**
- Create: `ui/crosshair_sync.py`
- Create: `tests/test_crosshair_sync.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_crosshair_sync.py
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
        self.calls = []

    def sync_crosshair(self, timestamp, price):
        self.calls.append((timestamp, price))


class CrosshairSyncControllerTests(unittest.TestCase):
    def test_sync_from_skips_source_but_includes_detached_targets(self):
        source = DummyChart("source")
        attached = DummyChart("attached")
        detached = DummyChart("detached", is_detached=True)
        controller = CrosshairSyncController()

        for chart in (source, attached, detached):
            controller.register_chart(chart)

        controller.sync_from(source, 123.0, 456.0)

        self.assertEqual(source.calls, [])
        self.assertEqual(attached.calls, [(123.0, 456.0)])
        self.assertEqual(detached.calls, [(123.0, 456.0)])

    def test_unregister_chart_removes_sync_target(self):
        source = DummyChart("source")
        target = DummyChart("target")
        controller = CrosshairSyncController()

        controller.register_chart(source)
        controller.register_chart(target)
        controller.unregister_chart(target)
        controller.sync_from(source, 1.0, 2.0)

        self.assertEqual(target.calls, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_crosshair_sync.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'ui.crosshair_sync'`

- [ ] **Step 3: Write the minimal implementation**

```python
# ui/crosshair_sync.py
from __future__ import annotations


class CrosshairSyncController:
    def __init__(self) -> None:
        self._charts: list[object] = []

    def register_chart(self, chart) -> None:
        if chart not in self._charts:
            self._charts.append(chart)

    def unregister_chart(self, chart) -> None:
        self._charts = [existing for existing in self._charts if existing is not chart]

    def sync_from(self, source_chart, timestamp: float, price: float) -> None:
        for chart in list(self._charts):
            if chart is source_chart:
                continue
            if chart is None or not hasattr(chart, "sync_crosshair"):
                continue
            chart.sync_crosshair(timestamp, price)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_crosshair_sync.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_crosshair_sync.py ui/crosshair_sync.py
git commit -m "feat: add crosshair sync controller"
```

### Task 2: Integrate the Controller into MainWindow and Fix the Window Title

**Files:**
- Modify: `ui/main_window.py:1-27`
- Modify: `ui/main_window.py:1259-1291`
- Modify: `ui/main_window.py:1523-1551`
- Modify: `ui/main_window.py:1636-1638`
- Create: `tests/test_main_window_crosshair_sync.py`

- [ ] **Step 1: Write the failing integration tests**

```python
# tests/test_main_window_crosshair_sync.py
import os
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.main_window import MainWindow


APP = QApplication.instance() or QApplication([])


class MainWindowCrosshairSyncTests(unittest.TestCase):
    def _make_window(self):
        with patch("ui.main_window.QTimer.singleShot", lambda *_args, **_kwargs: None):
            window = MainWindow()
        window.timer.stop()
        self.addCleanup(window.close)
        self.addCleanup(self._close_floating_windows, window)
        return window

    def _close_floating_windows(self, window):
        for floating in list(window.floating_windows):
            floating.close()

    def test_window_title_is_tradereview(self):
        window = self._make_window()

        self.assertEqual(window.windowTitle(), "TradeReview")

    def test_attached_chart_emits_crosshair_to_detached_chart(self):
        window = self._make_window()
        source = window.charts[0]
        detached_target = window.charts[1]
        window.detach_chart(detached_target)

        with patch.object(detached_target, "sync_crosshair") as sync_crosshair:
            source.sig_mouse_moved_with_price.emit(123.0, 456.0)
            APP.processEvents()

        sync_crosshair.assert_called_once_with(123.0, 456.0)

    def test_detached_chart_emits_crosshair_back_to_attached_chart(self):
        window = self._make_window()
        detached_source = window.charts[0]
        attached_target = window.charts[1]
        window.detach_chart(detached_source)

        with patch.object(attached_target, "sync_crosshair") as sync_crosshair:
            detached_source.sig_mouse_moved_with_price.emit(321.0, 654.0)
            APP.processEvents()

        sync_crosshair.assert_called_once_with(321.0, 654.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_main_window_crosshair_sync.py -q`

Expected: FAIL because:
- `MainWindow.windowTitle()` still returns `Gemini Trade Review`
- detached charts are excluded by the current crosshair target filtering path

- [ ] **Step 3: Write the minimal integration**

```python
# ui/main_window.py imports
from ui.crosshair_sync import CrosshairSyncController
from ui.chart_performance import build_visible_slice_window, should_refresh_visible_slice
```

```python
# ui/main_window.py inside MainWindow.__init__
        self.setWindowTitle("TradeReview")
        self.resize(1400, 950)

        self.engine = DataEngine(parquet_file=None)
        self.replay_engine = ReplayEngine(self.engine)
        self.settings = QSettings("TradeReview", "TradeReview")
        self.fib_settings = load_fib_settings(self.settings)
        self.crosshair_sync_controller = CrosshairSyncController()
        self.current_time = datetime.datetime.now()
```

```python
# ui/main_window.py inside init_charts
            chart.sig_set_replay_start.connect(self.set_replay_start_time)

            self.crosshair_sync_controller.register_chart(chart)
            chart.sig_mouse_moved_with_price.connect(
                partial(self.crosshair_sync_controller.sync_from, chart)
            )
            chart.sig_drawing_request.connect(self.on_drawing_request)
```

```python
# ui/main_window.py keep a thin compatibility wrapper
    def sync_all_charts_crosshair(self, source_chart, timestamp, price):
        self.crosshair_sync_controller.sync_from(source_chart, timestamp, price)
```

- [ ] **Step 4: Run the integration tests**

Run: `python -m pytest tests/test_crosshair_sync.py tests/test_main_window_crosshair_sync.py -q`

Expected: PASS

- [ ] **Step 5: Run the related regression suite**

Run: `python -m pytest tests/test_chart_performance.py tests/test_main_window_crosshair_sync.py tests/test_session_state.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_main_window_crosshair_sync.py ui/main_window.py
git commit -m "fix: sync detached chart crosshairs"
```

## Self-Review

- Spec coverage:
  - 主窗口标题改为 `TradeReview` 由 Task 2 覆盖
  - 独立 `CrosshairSyncController` 由 Task 1 覆盖
  - 主界面图表到 `Pop` 图表同步由 Task 2 覆盖
  - `Pop` 图表回同步到主界面图表由 Task 2 覆盖
  - 同步目标不再依赖 `is_detached` 由 Task 1 和 Task 2 共同覆盖
- Placeholder scan:
  - 无 `TODO`、`TBD`、模糊测试步骤或缺失命令
- Type consistency:
  - 统一使用 `CrosshairSyncController.register_chart()`、`unregister_chart()`、`sync_from()`
  - `MainWindow` 中统一命名为 `self.crosshair_sync_controller`
