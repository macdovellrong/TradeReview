# UI Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `ui/main_window.py` into focused UI modules while preserving existing TradeReview behavior.

**Architecture:** First perform safe file-level extraction with compatibility exports. Then introduce thin UI controllers and services for replay, workspace layout, and data loading. `MainWindow` remains the composition root and should shrink into widget assembly, signal wiring, and dialog presentation.

**Tech Stack:** Python, PyQt6, pyqtgraph, pandas, pytest

---

## File Structure

Create these modules:

- `ui/chart_primitives.py`: `MockYScale`, `TimeAxisItem`, `CandlestickItem`
- `ui/chart_widget.py`: `ChartWidget`
- `ui/chart_window.py`: `FloatingChartWindow`
- `ui/main_controls.py`: main toolbar widget and semantic control signals
- `ui/controllers/__init__.py`: controller package marker
- `ui/controllers/replay_controller.py`: thin UI replay coordinator
- `ui/controllers/workspace_layout_manager.py`: chart layout and floating-window coordinator
- `ui/services/__init__.py`: service package marker
- `ui/services/data_loading.py`: structured data-load result facade

Modify these modules:

- `ui/main_window.py`: remove moved classes, import focused modules, keep compatibility exports during migration
- `ui/chart_performance.py`: keep only visible-slice helpers or make crosshair helper part of production path
- `tests/test_drawing_chart_widget.py`: update imports after compatibility stage
- `tests/test_main_window_crosshair_sync.py`: split layout/crosshair assertions into clearer tests

Add these tests:

- `tests/test_ui_module_boundaries.py`
- `tests/test_chart_widget.py`
- `tests/test_main_window_time_navigation.py`
- `tests/test_main_window_layout_and_crosshair.py`
- `tests/test_replay_controller.py`
- `tests/test_data_loading_facade.py`

## Task 1: Lock Module Boundaries

**Files:**
- Create: `tests/test_ui_module_boundaries.py`

- [ ] **Step 1: Write the failing boundary test**

```python
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
        from ui.chart_primitives import CandlestickItem, TimeAxisItem
        from ui.chart_widget import ChartWidget
        from ui.chart_window import FloatingChartWindow
        from ui.main_window import (
            CandlestickItem as CompatCandlestickItem,
            ChartWidget as CompatChartWidget,
            FloatingChartWindow as CompatFloatingChartWindow,
            TimeAxisItem as CompatTimeAxisItem,
        )

        self.assertIs(CompatCandlestickItem, CandlestickItem)
        self.assertIs(CompatTimeAxisItem, TimeAxisItem)
        self.assertIs(CompatChartWidget, ChartWidget)
        self.assertIs(CompatFloatingChartWindow, FloatingChartWindow)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_module_boundaries.py
```

Expected: FAIL because `ui.chart_primitives`, `ui.chart_widget`, and `ui.chart_window` do not exist yet.

- [ ] **Step 3: Keep the failing test uncommitted until the modules exist**

Do not commit this test while it fails. It will be committed together with the first module extraction in Task 2.

## Task 2: Extract Chart Primitives

**Files:**
- Create: `ui/chart_primitives.py`
- Create: `ui/chart_widget.py` (temporary compatibility shim if missing)
- Create: `ui/chart_window.py` (temporary compatibility shim if missing)
- Modify: `ui/main_window.py`
- Test: `tests/test_ui_module_boundaries.py`

- [ ] **Step 1: Create `ui/chart_primitives.py`**

Move these classes from `ui/main_window.py` into `ui/chart_primitives.py` without changing method bodies:

- `MockYScale`
- `TimeAxisItem`
- `CandlestickItem`

The new file starts with these imports:

```python
import datetime

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QPainter, QPicture
```

Then paste the three existing class definitions immediately below the imports. The class bodies must remain byte-for-byte equivalent except for quote style changes introduced by formatting tools.

- [ ] **Step 2: Update `ui/main_window.py` imports**

Add:

```python
from ui.chart_primitives import CandlestickItem, MockYScale, TimeAxisItem
```

Remove the original `MockYScale`, `TimeAxisItem`, and `CandlestickItem` class definitions from `ui/main_window.py`.

- [ ] **Step 2.5: Add temporary compatibility shims if required by the boundary test**

If `ui/chart_widget.py` and `ui/chart_window.py` do not exist yet, create the smallest possible re-export shims so `tests/test_ui_module_boundaries.py` can pass before Task 3 and Task 4 move the real implementations:

`ui/chart_widget.py`

```python
from ui.main_window import ChartWidget
```

`ui/chart_window.py`

```python
from ui.main_window import FloatingChartWindow
```

Do not move any implementation into these two files during Task 2. They are temporary import targets only.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_module_boundaries.py tests/test_drawing_chart_widget.py
```

Expected: PASS for chart primitive imports and existing `ChartWidget` behavior.

- [ ] **Step 4: Commit primitive extraction**

```powershell
git add tests/test_ui_module_boundaries.py ui/chart_primitives.py ui/chart_widget.py ui/chart_window.py ui/main_window.py docs/superpowers/plans/2026-04-24-ui-architecture-refactor.md
git commit -m "refactor: 拆分图表底层绘制组件"
```

## Task 3: Extract Floating Chart Window

**Files:**
- Create: `ui/chart_window.py`
- Modify: `ui/main_window.py`
- Test: `tests/test_ui_module_boundaries.py`, `tests/test_main_window_crosshair_sync.py`

- [ ] **Step 1: Create `ui/chart_window.py`**

Move `FloatingChartWindow` from `ui/main_window.py` into:

```python
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class FloatingChartWindow(QWidget):
    sig_window_closed = pyqtSignal(object)

    def __init__(self, chart_widget, parent=None):
        super().__init__(parent)
        self.chart_widget = chart_widget
        self.setWindowTitle(f"Chart - {chart_widget.current_period}")
        self.resize(800, 600)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        layout.addWidget(self.chart_widget)
        self.chart_widget.show()

        self.chart_widget.sig_period_changed.connect(self.update_title)

    def update_title(self, period_display):
        self.setWindowTitle(f"Chart - {period_display}")

    def closeEvent(self, event):
        self.sig_window_closed.emit(self.chart_widget)
        event.accept()
```

- [ ] **Step 2: Update `ui/main_window.py`**

Add:

```python
from ui.chart_window import FloatingChartWindow
```

Remove the original `FloatingChartWindow` class definition from `ui/main_window.py`.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_module_boundaries.py tests/test_main_window_crosshair_sync.py
```

Expected: PASS. Detached chart tests must still pass.

- [ ] **Step 4: Commit floating window extraction**

```powershell
git add ui/chart_window.py ui/main_window.py
git commit -m "refactor: 拆分图表浮窗组件"
```

## Task 4: Extract ChartWidget

**Files:**
- Create: `ui/chart_widget.py`
- Modify: `ui/main_window.py`
- Modify: `tests/test_drawing_chart_widget.py`
- Test: `tests/test_ui_module_boundaries.py`, `tests/test_drawing_chart_widget.py`, `tests/test_main_window_crosshair_sync.py`

- [ ] **Step 1: Create `ui/chart_widget.py`**

Move `ChartWidget` from `ui/main_window.py` into `ui/chart_widget.py`.

Use these imports at the top of the new file:

```python
import datetime

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.chart_performance import build_visible_slice_window, should_refresh_visible_slice
from ui.chart_primitives import CandlestickItem, MockYScale, TimeAxisItem
from ui.drawings.fib_config import default_fib_settings
from ui.drawings.renderers import render_spec_items
from ui.drawings.specs import normalize_drawing_spec
from ui.drawings.tools import DrawingSession, TOOL_DEFINITIONS
```

- [ ] **Step 2: Update `ui/main_window.py`**

Add:

```python
from ui.chart_widget import ChartWidget
```

Remove the original `ChartWidget` class definition from `ui/main_window.py`.

Keep compatibility exports by leaving these imported names available from `ui.main_window`:

```python
from ui.chart_primitives import CandlestickItem, MockYScale, TimeAxisItem
from ui.chart_widget import ChartWidget
from ui.chart_window import FloatingChartWindow
```

- [ ] **Step 3: Update direct test imports**

Change `tests/test_drawing_chart_widget.py` from:

```python
from ui.main_window import ChartWidget
```

to:

```python
from ui.chart_widget import ChartWidget
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_ui_module_boundaries.py tests/test_drawing_chart_widget.py tests/test_main_window_crosshair_sync.py
```

Expected: PASS. Existing signal names and constructor behavior must remain unchanged.

- [ ] **Step 5: Commit ChartWidget extraction**

```powershell
git add ui/chart_widget.py ui/main_window.py tests/test_drawing_chart_widget.py
git commit -m "refactor: 拆分单图表组件"
```

## Task 5: Add MainWindow Time Navigation Characterization Tests

**Files:**
- Create: `tests/test_main_window_time_navigation.py`
- Modify: none

- [ ] **Step 1: Write tests for visible MainWindow time behavior**

```python
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


class MainWindowTimeNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def create_window(self):
        with patch("ui.main_window.QTimer.singleShot", lambda *args, **kwargs: None):
            window = MainWindow()
        window.timer.stop()
        self.addCleanup(window.close)
        return window

    def attach_ticks(self, window):
        index = pd.date_range(
            "2026-04-24 09:30:00",
            periods=5,
            freq="min",
            tz="America/New_York",
        )
        window.engine.df_ticks = pd.DataFrame(
            {"price": [100.0, 101.0, 102.0, 103.0, 104.0], "volume": [1, 1, 1, 1, 1]},
            index=index,
        )
        window.engine.parquet_file = "sample.duckdb"
        window.current_time = index[0]
        window._update_date_edit_bounds()
        return index

    def test_jump_to_time_clamps_to_loaded_tick_range(self):
        window = self.create_window()
        index = self.attach_ticks(window)

        with patch.object(window, "refresh_all_charts"), patch.object(window, "_center_charts_on_time"):
            window.jump_to_time(pd.Timestamp("2026-04-24 09:40:00", tz="America/New_York"))

        self.assertEqual(window.current_time, index[-1].floor("min"))

    def test_step_forward_uses_selected_step_size(self):
        window = self.create_window()
        index = self.attach_ticks(window)
        window.combo_step.setCurrentText("1m")

        with patch.object(window, "refresh_all_charts"):
            window.on_step_forward()

        self.assertEqual(window.current_time, index[1])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests before refactoring**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_main_window_time_navigation.py
```

Expected: PASS before extracting replay/time controller. These are characterization tests.

- [ ] **Step 3: Commit time navigation tests**

```powershell
git add tests/test_main_window_time_navigation.py
git commit -m "test: 补充主窗口时间导航行为测试"
```

## Task 6: Add MainWindow Layout and Crosshair Characterization Tests

**Files:**
- Create: `tests/test_main_window_layout_and_crosshair.py`

- [ ] **Step 1: Write layout and sync tests**

```python
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


class MainWindowLayoutAndCrosshairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def create_window(self):
        with patch("ui.main_window.QTimer.singleShot", lambda *args, **kwargs: None):
            window = MainWindow()
        window.timer.stop()
        self.addCleanup(window.close)
        return window

    def test_chart_count_limits_enabled_crosshair_targets(self):
        window = self.create_window()
        window.combo_chart_count.setCurrentText("2")
        window.on_chart_count_changed("2")

        registered = list(window.crosshair_sync_controller.iter_charts())

        self.assertEqual(registered, window.charts[:2])

    def test_detach_and_close_window_round_trip(self):
        window = self.create_window()
        chart = window.charts[0]

        window.detach_chart(chart, refresh_layout=False)

        self.assertTrue(chart.is_detached)
        self.assertEqual(len(window.floating_windows), 1)

        window.floating_windows[0].close()

        self.assertFalse(chart.is_detached)
        self.assertEqual(window.floating_windows, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run layout tests before extracting layout manager**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_main_window_layout_and_crosshair.py tests/test_main_window_crosshair_sync.py
```

Expected: PASS before layout refactor.

- [ ] **Step 3: Commit layout characterization tests**

```powershell
git add tests/test_main_window_layout_and_crosshair.py
git commit -m "test: 补充主窗口布局和同步行为测试"
```

## Task 7: Extract MainControls

**Files:**
- Create: `ui/main_controls.py`
- Modify: `ui/main_window.py`
- Test: `tests/test_main_window_time_navigation.py`, `tests/test_main_window_layout_and_crosshair.py`

- [ ] **Step 1: Create `MainControls` with semantic signals**

```python
import datetime

from PyQt6.QtCore import QDateTime, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class MainControls(QWidget):
    load_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    save_view_requested = pyqtSignal()
    layout_changed = pyqtSignal(str)
    pop_layout_requested = pyqtSignal()
    chart_count_changed = pyqtSignal(str)
    replay_mode_changed = pyqtSignal(int)
    play_requested = pyqtSignal()
    step_back_requested = pyqtSignal()
    step_forward_requested = pyqtSignal()
    speed_changed = pyqtSignal(int)
    date_edit_finished = pyqtSignal()

    def __init__(self, current_time=None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(self.load_requested)
        layout.addWidget(self.btn_load)

        self.btn_reset = QPushButton("Reset View")
        self.btn_reset.clicked.connect(self.reset_requested)
        layout.addWidget(self.btn_reset)

        self.btn_save_view = QPushButton("Save View")
        self.btn_save_view.clicked.connect(self.save_view_requested)
        layout.addWidget(self.btn_save_view)

        layout.addWidget(QLabel("Layout:"))
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["Tabs", "Dual Vertical", "Grid 2x2", "Vertical"])
        self.combo_layout.currentTextChanged.connect(self.layout_changed)
        layout.addWidget(self.combo_layout)

        self.btn_detach_layout = QPushButton("Pop Layout")
        self.btn_detach_layout.clicked.connect(self.pop_layout_requested)
        layout.addWidget(self.btn_detach_layout)

        layout.addWidget(QLabel("Charts:"))
        self.combo_chart_count = QComboBox()
        self.combo_chart_count.addItems(["1", "2", "3", "4"])
        self.combo_chart_count.setCurrentText("4")
        self.combo_chart_count.currentTextChanged.connect(self.chart_count_changed)
        layout.addWidget(self.combo_chart_count)

        self.chk_replay = QCheckBox("Replay Mode")
        self.chk_replay.stateChanged.connect(self.replay_mode_changed)
        layout.addWidget(self.chk_replay)

        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self.play_requested)
        layout.addWidget(self.btn_play)

        self.btn_step_back = QPushButton("Back")
        self.btn_step_back.clicked.connect(self.step_back_requested)
        layout.addWidget(self.btn_step_back)

        self.btn_step_forward = QPushButton("Forward")
        self.btn_step_forward.clicked.connect(self.step_forward_requested)
        layout.addWidget(self.btn_step_forward)

        self.combo_step = QComboBox()
        self.combo_step.addItems(["30s", "1m", "5m", "15m", "30m", "1h", "2h", "4h", "1D"])
        self.combo_step.setCurrentText("1h")
        layout.addWidget(self.combo_step)

        layout.addWidget(QLabel("Speed:"))
        self.speed_btn_group = QButtonGroup(self)
        self.speed_btn_group.setExclusive(True)
        for speed in [1, 10, 60, 120, 300, 600]:
            btn = QPushButton(f"{speed}x")
            btn.setCheckable(True)
            btn.setFixedSize(40, 25)
            if speed == 60:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, value=speed: self.speed_changed.emit(value))
            self.speed_btn_group.addButton(btn)
            layout.addWidget(btn)

        self.date_edit = QDateTimeEdit()
        self.date_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setKeyboardTracking(False)
        self.date_edit.editingFinished.connect(self.date_edit_finished)
        dt = current_time or datetime.datetime.now()
        self.date_edit.setDateTime(QDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0))
        layout.addWidget(self.date_edit)

        layout.addStretch()
```

- [ ] **Step 2: Replace `create_control_panel()` internals**

In `ui/main_window.py`, import `MainControls` and create it:

```python
from ui.main_controls import MainControls
```

Replace `create_control_panel()` body with:

```python
def create_control_panel(self):
    self.controls = MainControls(current_time=self.current_time, parent=self)
    self.controls.load_requested.connect(self.open_file_dialog)
    self.controls.reset_requested.connect(self.reset_charts_view)
    self.controls.save_view_requested.connect(self.on_save_view)
    self.controls.layout_changed.connect(self.switch_layout)
    self.controls.pop_layout_requested.connect(self.detach_layout_charts)
    self.controls.chart_count_changed.connect(self.on_chart_count_changed)
    self.controls.replay_mode_changed.connect(self.on_mode_change)
    self.controls.play_requested.connect(self.toggle_play)
    self.controls.step_back_requested.connect(self.on_step_back)
    self.controls.step_forward_requested.connect(self.on_step_forward)
    self.controls.speed_changed.connect(self.set_speed)
    self.controls.date_edit_finished.connect(self.on_date_edit_finished)

    self.combo_layout = self.controls.combo_layout
    self.btn_detach_layout = self.controls.btn_detach_layout
    self.combo_chart_count = self.controls.combo_chart_count
    self.chk_replay = self.controls.chk_replay
    self.btn_play = self.controls.btn_play
    self.combo_step = self.controls.combo_step
    self.date_edit = self.controls.date_edit

    self.main_layout.addWidget(self.controls)
```

- [ ] **Step 3: Run focused tests**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_main_window_time_navigation.py tests/test_main_window_layout_and_crosshair.py tests/test_main_window_crosshair_sync.py
```

Expected: PASS. Existing `MainWindow` attributes must remain available through compatibility assignments.

- [ ] **Step 4: Commit control panel extraction**

```powershell
git add ui/main_controls.py ui/main_window.py
git commit -m "refactor: 拆分主控制面板"
```

## Task 8: Introduce ReplayController

**Files:**
- Create: `ui/controllers/__init__.py`
- Create: `ui/controllers/replay_controller.py`
- Create: `tests/test_replay_controller.py`
- Modify: `ui/main_window.py`

- [ ] **Step 1: Write `ReplayController` tests**

```python
import unittest


class FakeReplayEngine:
    def __init__(self):
        self.initialized_with = None
        self.reset_with = None
        self.advanced_to = None

    def initialize(self, periods, start_time, max_count_map=None):
        self.initialized_with = (list(periods), start_time, dict(max_count_map or {}))

    def reset(self, start_time):
        self.reset_with = start_time

    def advance_to(self, end_time):
        self.advanced_to = end_time
        return end_time

    def get_view(self, period, count=300, with_indicators=True):
        return {"period": period, "count": count, "with_indicators": with_indicators}


class ReplayControllerTests(unittest.TestCase):
    def test_initialize_forwards_to_engine(self):
        from ui.controllers.replay_controller import ReplayController

        engine = FakeReplayEngine()
        controller = ReplayController(engine)

        controller.initialize(["1min"], "2026-04-24 09:30", {"1min": 800})

        self.assertEqual(engine.initialized_with, (["1min"], "2026-04-24 09:30", {"1min": 800}))

    def test_advance_returns_actual_time_and_tracks_current_time(self):
        from ui.controllers.replay_controller import ReplayController

        engine = FakeReplayEngine()
        controller = ReplayController(engine)

        actual = controller.advance_to("2026-04-24 09:31")

        self.assertEqual(actual, "2026-04-24 09:31")
        self.assertEqual(controller.current_time, "2026-04-24 09:31")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the controller tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_replay_controller.py
```

Expected: FAIL because `ui.controllers.replay_controller` does not exist.

- [ ] **Step 3: Implement `ReplayController`**

```python
class ReplayController:
    def __init__(self, replay_engine):
        self.replay_engine = replay_engine
        self.enabled = False
        self.is_playing = False
        self.speed = 60
        self.current_time = None

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)
        if not self.enabled:
            self.is_playing = False

    def set_speed(self, speed: int):
        self.speed = int(speed)

    def toggle_playing(self):
        self.is_playing = not self.is_playing
        return self.is_playing

    def initialize(self, periods, start_time, max_count_map=None):
        self.current_time = start_time
        self.replay_engine.initialize(periods, start_time, max_count_map=max_count_map)

    def reset(self, start_time):
        self.current_time = start_time
        self.replay_engine.reset(start_time)

    def advance_to(self, end_time):
        actual_time = self.replay_engine.advance_to(end_time)
        if actual_time is not None:
            self.current_time = actual_time
        return actual_time

    def get_view(self, period, count=300, with_indicators=True):
        return self.replay_engine.get_view(period, count=count, with_indicators=with_indicators)
```

- [ ] **Step 4: Wire `MainWindow` gradually**

Add in `MainWindow.__init__` after `self.replay_engine = ReplayEngine(self.engine)`:

```python
from ui.controllers.replay_controller import ReplayController

self.replay_controller = ReplayController(self.replay_engine)
```

Only replace local replay state when tests can stay green. Keep `self.is_playing` and `self.replay_speed` aliases until a later cleanup.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_replay_controller.py tests/test_main_window_time_navigation.py tests/test_main_window_crosshair_sync.py
```

Expected: PASS.

- [ ] **Step 6: Commit replay controller**

```powershell
git add ui/controllers/__init__.py ui/controllers/replay_controller.py tests/test_replay_controller.py ui/main_window.py
git commit -m "refactor: 引入回放控制器"
```

## Task 9: Introduce Data Loading Facade

**Files:**
- Create: `ui/services/__init__.py`
- Create: `ui/services/data_loading.py`
- Create: `tests/test_data_loading_facade.py`

- [ ] **Step 1: Write data loading facade tests**

```python
import unittest

import pandas as pd


class FakeDataEngine:
    def __init__(self, df_ticks=None, error=None, warnings=None):
        self.parquet_file = None
        self.df_ticks = df_ticks
        self.last_load_error = error
        self.last_load_warnings = warnings or []
        self.loaded = False

    def load_data(self):
        self.loaded = True


class DataLoadingFacadeTests(unittest.TestCase):
    def test_load_returns_initial_time_after_100000_ticks_for_large_dataset(self):
        from ui.services.data_loading import DataLoadingFacade

        index = pd.date_range("2026-04-24", periods=100001, freq="s", tz="America/New_York")
        engine = FakeDataEngine(pd.DataFrame({"price": range(len(index)), "volume": 1}, index=index))
        facade = DataLoadingFacade(engine)

        result = facade.load("sample.duckdb")

        self.assertTrue(result.success)
        self.assertEqual(result.initial_time, index[100000])
        self.assertEqual(result.warnings, [])

    def test_load_returns_error_when_engine_has_no_ticks(self):
        from ui.services.data_loading import DataLoadingFacade

        engine = FakeDataEngine(df_ticks=None, error="bad file")
        facade = DataLoadingFacade(engine)

        result = facade.load("bad.duckdb")

        self.assertFalse(result.success)
        self.assertEqual(result.error, "bad file")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement facade**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoadResult:
    success: bool
    file_path: str
    initial_time: object = None
    error: str | None = None
    warnings: tuple[str, ...] = ()


class DataLoadingFacade:
    def __init__(self, engine):
        self.engine = engine

    def load(self, file_path: str) -> DataLoadResult:
        self.engine.parquet_file = file_path
        self.engine.load_data()
        df_ticks = self.engine.df_ticks
        if df_ticks is None or df_ticks.empty:
            return DataLoadResult(
                success=False,
                file_path=file_path,
                error=self.engine.last_load_error or "Failed to load the selected data file.",
            )

        total_ticks = len(df_ticks)
        if total_ticks > 100000:
            initial_time = df_ticks.index[100000]
        else:
            initial_time = df_ticks.index[0]

        return DataLoadResult(
            success=True,
            file_path=file_path,
            initial_time=initial_time,
            warnings=tuple(self.engine.last_load_warnings or ()),
        )
```

- [ ] **Step 3: Run facade tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_data_loading_facade.py
```

Expected: PASS.

- [ ] **Step 4: Wire facade into `MainWindow.load_data_file()`**

In `MainWindow.__init__`, create:

```python
from ui.services.data_loading import DataLoadingFacade

self.data_loading = DataLoadingFacade(self.engine)
```

Update `load_data_file()` to call:

```python
result = self.data_loading.load(file_path)
if not result.success:
    QMessageBox.critical(self, "Load Data Failed", result.error)
    return
self.current_time = result.initial_time
```

Keep the existing chart refresh, replay initialize, restore jump, and warning dialog behavior unchanged.

- [ ] **Step 5: Run data and window tests**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_data_loading_facade.py tests/test_main_window_time_navigation.py tests/test_main_window_crosshair_sync.py
```

Expected: PASS.

- [ ] **Step 6: Commit data loading facade**

```powershell
git add ui/services/__init__.py ui/services/data_loading.py tests/test_data_loading_facade.py ui/main_window.py
git commit -m "refactor: 引入数据加载门面"
```

## Task 10: Full Verification and Cleanup

**Files:**
- Modify: imports and tests touched by previous tasks

- [ ] **Step 1: Run full test suite**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Check for stale imports from old locations**

Run:

```powershell
rg -n "from ui\.main_window import ChartWidget|from ui\.main_window import FloatingChartWindow|from ui\.main_window import TimeAxisItem|from ui\.main_window import CandlestickItem" tests ui
```

Expected: no test imports from old class locations. Compatibility exports may remain in `ui/main_window.py` for one release cycle.

- [ ] **Step 3: Check `ui/main_window.py` size trend**

Run:

```powershell
(Get-Content ui/main_window.py).Count
```

Expected: lower than the pre-refactor baseline of 2145 lines.

- [ ] **Step 4: Commit final cleanup**

```powershell
git add ui tests
git commit -m "refactor: 收敛界面架构拆分"
```

## Execution Notes

- Execute tasks in order.
- Do not combine ChartWidget extraction with replay changes.
- Do not combine data loading facade with time navigation changes.
- Keep compatibility exports in `ui/main_window.py` until all internal imports have moved.
- Every commit message must be Chinese when using `git commit`.
