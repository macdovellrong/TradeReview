# Drawing System Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the chart drawing system so Fibonacci retracement levels are configurable, add a standard three-point Fibonacci extension tool, and leave the codebase ready for more drawing tools later.

**Architecture:** Move Fibonacci configuration, drawing specs, point-collection state, and renderer logic into focused modules under `ui/`, then keep `ui/main_window.py` as the integration layer that wires toolbar actions, dialog opening, chart events, and cross-chart broadcasts. Snapshot Fibonacci settings into each created drawing so new settings only affect new objects.

**Tech Stack:** Python, PyQt6, pyqtgraph, unittest, QSettings

---

### Task 1: Add Fibonacci Config and Math Modules

**Files:**
- Create: `ui/drawings/__init__.py`
- Create: `ui/drawings/fib_config.py`
- Create: `ui/drawings/fib_math.py`
- Test: `tests/test_drawing_fib_config.py`
- Test: `tests/test_drawing_fib_math.py`

- [ ] **Step 1: Write the failing Fibonacci config tests**

```python
import tempfile
import unittest
from pathlib import Path

from PyQt6.QtCore import QSettings

from ui.drawings.fib_config import (
    DEFAULT_EXTENSION_LEVELS,
    DEFAULT_RETRACEMENT_LEVELS,
    FibLevelsConfig,
    FibSettings,
    load_fib_settings,
    normalize_level_text,
    save_fib_settings,
)


class FibConfigTests(unittest.TestCase):
    def test_normalize_level_text_merges_presets_and_custom_values(self):
        levels = normalize_level_text([0.5, 0.618], "0.618, 0.786, 0.8")
        self.assertEqual(levels, [0.5, 0.618, 0.786, 0.8])

    def test_normalize_level_text_rejects_invalid_tokens(self):
        with self.assertRaises(ValueError):
            normalize_level_text([0.5], "0.7,abc")

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = QSettings(str(Path(tmpdir) / "fib.ini"), QSettings.Format.IniFormat)
            state = FibSettings(
                retracement=FibLevelsConfig(selected_levels=[0.5, 0.618], custom_text="0.786"),
                extension=FibLevelsConfig(selected_levels=[1.0, 1.618], custom_text="2.0"),
            )
            save_fib_settings(settings, state)
            loaded = load_fib_settings(settings)
        self.assertEqual(loaded.retracement.effective_levels, [0.5, 0.618, 0.786])
        self.assertEqual(loaded.extension.effective_levels, [1.0, 1.618, 2.0])

    def test_defaults_match_requested_levels(self):
        self.assertEqual(DEFAULT_RETRACEMENT_LEVELS, [0.236, 0.382, 0.5, 0.618, 0.7, 0.786, 0.8])
        self.assertEqual(DEFAULT_EXTENSION_LEVELS, [0.618, 1.0, 1.272, 1.618, 2.0])
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `\\10.0.0.23\code\gold\TradeReview\.venv\Scripts\python.exe -m unittest tests.test_drawing_fib_config -v`

Expected: `ModuleNotFoundError` for `ui.drawings.fib_config`

- [ ] **Step 3: Write the failing Fibonacci math tests**

```python
import unittest

from ui.drawings.fib_math import (
    build_extension_levels,
    build_retracement_levels,
)


class FibMathTests(unittest.TestCase):
    def test_build_retracement_levels_uses_requested_ratios(self):
        rows = build_retracement_levels(120.0, 100.0, [0.5, 0.618, 0.786])
        self.assertEqual([row.ratio for row in rows], [0.5, 0.618, 0.786])
        self.assertEqual([round(row.price, 3) for row in rows], [110.0, 107.64, 104.28])

    def test_build_extension_levels_projects_upward(self):
        rows = build_extension_levels(100.0, 120.0, 110.0, [1.0, 1.618])
        self.assertEqual([round(row.price, 3) for row in rows], [130.0, 142.36])

    def test_build_extension_levels_projects_downward(self):
        rows = build_extension_levels(120.0, 100.0, 110.0, [1.0, 1.618])
        self.assertEqual([round(row.price, 3) for row in rows], [90.0, 77.64])
```

- [ ] **Step 4: Run the math tests to verify they fail**

Run: `\\10.0.0.23\code\gold\TradeReview\.venv\Scripts\python.exe -m unittest tests.test_drawing_fib_math -v`

Expected: `ModuleNotFoundError` for `ui.drawings.fib_math`

- [ ] **Step 5: Write the minimal config and math implementation**

```python
# ui/drawings/fib_math.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FibLevelRow:
    ratio: float
    price: float


def build_retracement_levels(start_price: float, end_price: float, levels: list[float]) -> list[FibLevelRow]:
    return [
        FibLevelRow(ratio=ratio, price=end_price + (start_price - end_price) * ratio)
        for ratio in levels
    ]


def build_extension_levels(a_price: float, b_price: float, c_price: float, levels: list[float]) -> list[FibLevelRow]:
    delta = b_price - a_price
    return [FibLevelRow(ratio=ratio, price=c_price + delta * ratio) for ratio in levels]
```

```python
# ui/drawings/fib_config.py
from __future__ import annotations

from dataclasses import dataclass


DEFAULT_RETRACEMENT_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.7, 0.786, 0.8]
DEFAULT_EXTENSION_LEVELS = [0.618, 1.0, 1.272, 1.618, 2.0]
KEY_RETRACEMENT_LEVELS = "drawing/fib/retracement_selected"
KEY_RETRACEMENT_CUSTOM = "drawing/fib/retracement_custom"
KEY_EXTENSION_LEVELS = "drawing/fib/extension_selected"
KEY_EXTENSION_CUSTOM = "drawing/fib/extension_custom"


def _normalize_levels(values: list[float]) -> list[float]:
    cleaned = []
    for value in values:
        value = float(value)
        if value < 0:
            raise ValueError("Fibonacci levels must be non-negative")
        cleaned.append(value)
    return sorted(set(cleaned))


def normalize_level_text(selected_levels: list[float], custom_text: str) -> list[float]:
    custom_levels = []
    for raw in custom_text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            custom_levels.append(float(raw))
        except ValueError as exc:
            raise ValueError(f"Invalid Fibonacci level: {raw}") from exc
    return _normalize_levels([*selected_levels, *custom_levels])


@dataclass(frozen=True)
class FibLevelsConfig:
    selected_levels: list[float]
    custom_text: str = ""

    @property
    def effective_levels(self) -> list[float]:
        return normalize_level_text(self.selected_levels, self.custom_text)


@dataclass(frozen=True)
class FibSettings:
    retracement: FibLevelsConfig
    extension: FibLevelsConfig


def default_fib_settings() -> FibSettings:
    return FibSettings(
        retracement=FibLevelsConfig(selected_levels=DEFAULT_RETRACEMENT_LEVELS),
        extension=FibLevelsConfig(selected_levels=DEFAULT_EXTENSION_LEVELS),
    )


def save_fib_settings(settings, state: FibSettings) -> None:
    settings.setValue(KEY_RETRACEMENT_LEVELS, state.retracement.selected_levels)
    settings.setValue(KEY_RETRACEMENT_CUSTOM, state.retracement.custom_text)
    settings.setValue(KEY_EXTENSION_LEVELS, state.extension.selected_levels)
    settings.setValue(KEY_EXTENSION_CUSTOM, state.extension.custom_text)
    settings.sync()


def load_fib_settings(settings) -> FibSettings:
    defaults = default_fib_settings()
    retracement_levels = settings.value(KEY_RETRACEMENT_LEVELS, defaults.retracement.selected_levels)
    extension_levels = settings.value(KEY_EXTENSION_LEVELS, defaults.extension.selected_levels)
    return FibSettings(
        retracement=FibLevelsConfig(
            selected_levels=[float(value) for value in retracement_levels],
            custom_text=settings.value(KEY_RETRACEMENT_CUSTOM, "", type=str),
        ),
        extension=FibLevelsConfig(
            selected_levels=[float(value) for value in extension_levels],
            custom_text=settings.value(KEY_EXTENSION_CUSTOM, "", type=str),
        ),
    )
```

- [ ] **Step 6: Run the new tests and verify they pass**

Run: `\\10.0.0.23\code\gold\TradeReview\.venv\Scripts\python.exe -m unittest tests.test_drawing_fib_config tests.test_drawing_fib_math -v`

Expected: all tests in both modules pass

- [ ] **Step 7: Commit**

```bash
git add ui/drawings/__init__.py ui/drawings/fib_config.py ui/drawings/fib_math.py tests/test_drawing_fib_config.py tests/test_drawing_fib_math.py
git commit -m "feat: add fibonacci drawing config and math helpers"
```

### Task 2: Add Drawing Specs, Compatibility Helpers, and Tool State

**Files:**
- Create: `ui/drawings/specs.py`
- Create: `ui/drawings/tools.py`
- Test: `tests/test_drawing_tools.py`

- [ ] **Step 1: Write the failing drawing state tests**

```python
import datetime
import unittest

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
```

- [ ] **Step 2: Run the state tests to verify they fail**

Run: `\\10.0.0.23\code\gold\TradeReview\.venv\Scripts\python.exe -m unittest tests.test_drawing_tools -v`

Expected: `ModuleNotFoundError` for `ui.drawings.specs` or `ui.drawings.tools`

- [ ] **Step 3: Write the minimal normalized spec and tool state implementation**

```python
# ui/drawings/specs.py
from __future__ import annotations


def normalize_drawing_spec(spec: dict) -> dict:
    if "points" in spec:
        return dict(spec)
    points = []
    if "p1_dt" in spec:
        points.append({"dt": spec.get("p1_dt"), "price": spec.get("p1_price")})
    if "p2_dt" in spec:
        points.append({"dt": spec.get("p2_dt"), "price": spec.get("p2_price")})
    if "p3_dt" in spec:
        points.append({"dt": spec.get("p3_dt"), "price": spec.get("p3_price")})
    normalized = dict(spec)
    normalized["points"] = points
    return normalized
```

```python
# ui/drawings/tools.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolDefinition:
    tool_id: str
    point_count: int


TOOL_DEFINITIONS = {
    "hline": ToolDefinition(tool_id="hline", point_count=1),
    "vline": ToolDefinition(tool_id="vline", point_count=1),
    "line": ToolDefinition(tool_id="line", point_count=2),
    "fib": ToolDefinition(tool_id="fib", point_count=2),
    "fib_ext": ToolDefinition(tool_id="fib_ext", point_count=3),
}


@dataclass
class DrawingSession:
    tool: ToolDefinition
    config_snapshot: dict | None = None
    points: list[dict] = field(default_factory=list)

    def add_point(self, dt, price: float):
        self.points.append({"dt": dt, "price": float(price)})
        if len(self.points) < self.tool.point_count:
            return None
        spec = {"type": self.tool.tool_id, "points": list(self.points)}
        if self.config_snapshot is not None:
            spec["config_snapshot"] = self.config_snapshot
        return spec
```

- [ ] **Step 4: Run the state tests to verify they pass**

Run: `\\10.0.0.23\code\gold\TradeReview\.venv\Scripts\python.exe -m unittest tests.test_drawing_tools -v`

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add ui/drawings/specs.py ui/drawings/tools.py tests/test_drawing_tools.py
git commit -m "feat: add drawing spec normalization and tool sessions"
```

### Task 3: Add Renderer and Fibonacci Settings Dialog

**Files:**
- Create: `ui/drawings/renderers.py`
- Create: `ui/drawings/dialogs.py`
- Test: `tests/test_drawing_renderers.py`

- [ ] **Step 1: Write the failing renderer tests**

```python
import datetime
import unittest

from ui.drawings.renderers import build_render_plan


class DrawingRenderPlanTests(unittest.TestCase):
    def test_build_render_plan_for_fib_uses_snapshot_levels(self):
        plan = build_render_plan(
            {
                "id": 1,
                "type": "fib",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 16, 9, 30), "price": 120.0},
                    {"dt": datetime.datetime(2026, 4, 16, 10, 0), "price": 100.0},
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
```

- [ ] **Step 2: Run the renderer tests to verify they fail**

Run: `\\10.0.0.23\code\gold\TradeReview\.venv\Scripts\python.exe -m unittest tests.test_drawing_renderers -v`

Expected: `ModuleNotFoundError` for `ui.drawings.renderers`

- [ ] **Step 3: Write the minimal render-plan and dialog implementation**

```python
# ui/drawings/renderers.py
from __future__ import annotations

from ui.drawings.fib_math import build_extension_levels, build_retracement_levels
from ui.drawings.specs import normalize_drawing_spec


def build_render_plan(spec: dict) -> dict:
    spec = normalize_drawing_spec(spec)
    levels = list(spec.get("config_snapshot", {}).get("levels", []))
    points = spec["points"]
    if spec["type"] == "fib":
        rows = build_retracement_levels(points[0]["price"], points[1]["price"], levels)
    elif spec["type"] == "fib_ext":
        rows = build_extension_levels(points[0]["price"], points[1]["price"], points[2]["price"], levels)
    else:
        rows = []
    return {"type": spec["type"], "levels": levels, "rows": [row.__dict__ for row in rows], "points": points}
```

```python
# ui/drawings/dialogs.py
from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QLineEdit, QMessageBox, QVBoxLayout

from ui.drawings.fib_config import FibLevelsConfig, FibSettings, normalize_level_text


class FibConfigDialog(QDialog):
    def __init__(self, fib_settings: FibSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fib Config")
        self._retracement_boxes = []
        self._extension_boxes = []
        self._retracement_edit = QLineEdit(fib_settings.retracement.custom_text)
        self._extension_edit = QLineEdit(fib_settings.extension.custom_text)
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_group("Retracement Levels", fib_settings.retracement.selected_levels, self._retracement_boxes, self._retracement_edit))
        layout.addWidget(self._build_group("Extension Levels", fib_settings.extension.selected_levels, self._extension_boxes, self._extension_edit))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_group(self, title, selected_levels, box_store, line_edit):
        group = QGroupBox(title)
        form = QFormLayout(group)
        for value in selected_levels:
            box = QCheckBox(f"{value:g}")
            box.setChecked(True)
            box_store.append(box)
            form.addRow(box)
        form.addRow("Custom", line_edit)
        return group

    def build_settings(self) -> FibSettings:
        try:
            retracement_selected = [float(box.text()) for box in self._retracement_boxes if box.isChecked()]
            extension_selected = [float(box.text()) for box in self._extension_boxes if box.isChecked()]
            normalize_level_text(retracement_selected, self._retracement_edit.text())
            normalize_level_text(extension_selected, self._extension_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Fib Config", str(exc))
            raise
        return FibSettings(
            retracement=FibLevelsConfig(retracement_selected, self._retracement_edit.text()),
            extension=FibLevelsConfig(extension_selected, self._extension_edit.text()),
        )
```

- [ ] **Step 4: Run the renderer tests to verify they pass**

Run: `\\10.0.0.23\code\gold\TradeReview\.venv\Scripts\python.exe -m unittest tests.test_drawing_renderers -v`

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add ui/drawings/renderers.py ui/drawings/dialogs.py tests/test_drawing_renderers.py
git commit -m "feat: add drawing render plans and fib config dialog"
```

### Task 4: Integrate the New Drawing System into Main Window

**Files:**
- Modify: `ui/main_window.py`
- Test: `tests/test_chart_performance.py`
- Test: `tests/test_drawing_fib_config.py`
- Test: `tests/test_drawing_fib_math.py`
- Test: `tests/test_drawing_tools.py`
- Test: `tests/test_drawing_renderers.py`

- [ ] **Step 1: Add one failing integration test for fib-level defaults staying in config**

```python
import unittest

from ui.drawings.fib_config import default_fib_settings


class FibIntegrationDefaultsTests(unittest.TestCase):
    def test_default_fib_settings_include_requested_retracement_and_extension_levels(self):
        settings = default_fib_settings()
        self.assertEqual(settings.retracement.effective_levels, [0.236, 0.382, 0.5, 0.618, 0.7, 0.786, 0.8])
        self.assertEqual(settings.extension.effective_levels, [0.618, 1.0, 1.272, 1.618, 2.0])
```

- [ ] **Step 2: Run the targeted integration test to verify it passes after earlier tasks**

Run: `\\10.0.0.23\code\gold\TradeReview\.venv\Scripts\python.exe -m unittest tests.test_drawing_fib_config.FibConfigTests.test_defaults_match_requested_levels -v`

Expected: PASS

- [ ] **Step 3: Refactor `ui/main_window.py` to use the new modules**

```python
from ui.drawings.dialogs import FibConfigDialog
from ui.drawings.fib_config import default_fib_settings, load_fib_settings, save_fib_settings
from ui.drawings.renderers import add_spec_to_chart, build_preview_items
from ui.drawings.specs import normalize_drawing_spec
from ui.drawings.tools import DrawingSession, TOOL_DEFINITIONS
```

```python
self.fib_settings = load_fib_settings(self.settings)
self.active_drawing_session = None
self.btn_draw_fib_ext = QPushButton("Fib Ext")
self.btn_draw_fib_config = QPushButton("Fib Config")
self.btn_draw_fib_ext.clicked.connect(lambda: self.set_draw_mode("fib_ext"))
self.btn_draw_fib_config.clicked.connect(self.on_open_fib_config)
```

```python
def set_draw_mode(self, mode):
    self.draw_mode = mode
    self.active_drawing_session = None if mode is None else DrawingSession(
        TOOL_DEFINITIONS[mode],
        config_snapshot=self._snapshot_for_tool(mode),
    )
    self._clear_preview()
```

```python
def _snapshot_for_tool(self, mode):
    if mode == "fib":
        return {"levels": self.fib_settings.retracement.effective_levels}
    if mode == "fib_ext":
        return {"levels": self.fib_settings.extension.effective_levels}
    return None
```

```python
def _handle_draw_click(self, scene_pos):
    if self.draw_mode is None:
        return
    mouse_point = self.ax.vb.mapSceneToView(scene_pos)
    dt = self.get_datetime_from_x(mouse_point.x())
    spec = self.active_drawing_session.add_point(dt, float(mouse_point.y()))
    if spec is not None:
        self.sig_drawing_request.emit(spec)
        self.set_draw_mode(None)
```

```python
def add_drawing(self, spec):
    spec = normalize_drawing_spec(spec)
    items = add_spec_to_chart(self.ax, spec, self._x_from_datetime)
    if items:
        self.drawings[spec["id"]] = items
```

- [ ] **Step 4: Run the focused drawing tests**

Run: `\\10.0.0.23\code\gold\TradeReview\.venv\Scripts\python.exe -m unittest tests.test_drawing_fib_config tests.test_drawing_fib_math tests.test_drawing_tools tests.test_drawing_renderers -v`

Expected: all drawing tests pass

- [ ] **Step 5: Run the full test suite**

Run: `\\10.0.0.23\code\gold\TradeReview\.venv\Scripts\python.exe -m unittest discover -s tests -v`

Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add ui/main_window.py tests/test_drawing_fib_config.py tests/test_drawing_fib_math.py tests/test_drawing_tools.py tests/test_drawing_renderers.py
git commit -m "feat: refactor drawing system for configurable fibonacci tools"
```
