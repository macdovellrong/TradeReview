# Editable Global Drawings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement globally persisted, directly editable drawings with rectangle support and cross-chart synchronization across all periods.

**Architecture:** Introduce a dedicated drawing subsystem built around `GlobalDrawingStore`, `DrawingController`, `ChartDrawingLayer`, and tool-specific `Editable*Item` classes. `ChartWidget` delegates drawing interaction to the new layer/controller while `MainWindow` owns the shared store, persistence wiring, and chart registration.

**Tech Stack:** Python 3, PyQt6, pyqtgraph, pandas, unittest executed through `python -m pytest`

---

## File Structure

- Create: `ui/drawings/store.py`
  Stores canonical drawing specs, allocates ids, and preserves draw order.
- Create: `ui/drawings/persistence.py`
  Saves and loads global drawings with `QSettings`.
- Create: `ui/drawings/coords.py`
  Converts `datetime <-> plot x` with left/right extrapolation.
- Create: `ui/drawings/editable_items.py`
  Implements `EditableHLineItem`, `EditableVLineItem`, `EditableLineItem`, `EditableFibItem`, `EditableFibExtensionItem`, and `EditableRectItem`.
- Create: `ui/drawings/controller.py`
  Owns selection state, drag state, store mutations, and chart broadcasts.
- Create: `ui/drawings/layer.py`
  Bridges one `ChartWidget` plot to the shared controller/store.
- Create: `tests/test_drawing_specs.py`
  Covers spec serialization and rect normalization.
- Create: `tests/test_drawing_store.py`
  Covers store mutations and persistence recovery.
- Create: `tests/test_drawing_coords.py`
  Covers in-range and extrapolated x-coordinate mapping.
- Create: `tests/test_editable_drawing_items.py`
  Covers handle dragging, body dragging, and selected-state handle visibility.
- Create: `tests/test_chart_drawing_layer.py`
  Covers per-chart item syncing and selection propagation.
- Create: `tests/test_drawing_persistence.py`
  Covers QSettings round-trip for global drawings.
- Modify: `ui/drawings/specs.py:1-25`
  Add supported drawing types plus serialize/deserialize helpers.
- Modify: `ui/drawings/tools.py:1-43`
  Add `rect` tool definition and two-point preview behavior.
- Modify: `ui/drawings/renderers.py:1-152`
  Add `rect` rendering and selected-style support.
- Modify: `ui/main_window.py:160-1045`
  Integrate `ChartDrawingLayer`, selection/drag handling, `Rect` toolbar button, and delete-key routing.
- Modify: `ui/main_window.py:1259-1655`
  Create and restore the shared store/controller/persistence wiring in `MainWindow`.
- Modify: `tests/test_drawing_tools.py:1-62`
  Add rectangle drawing session tests.
- Modify: `tests/test_drawing_renderers.py:1-104`
  Add rectangle render coverage.
- Modify: `tests/test_drawing_chart_widget.py:1-56`
  Add chart-level creation, selection, and delete integration checks.

### Task 1: Normalize Persisted Specs and Add Rect Sessions

**Files:**
- Create: `tests/test_drawing_specs.py`
- Modify: `tests/test_drawing_tools.py:1-62`
- Modify: `ui/drawings/specs.py:1-25`
- Modify: `ui/drawings/tools.py:1-43`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_drawing_specs.py
import datetime
import unittest
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.specs import deserialize_drawing_spec, serialize_drawing_spec


class DrawingSpecSerializationTests(unittest.TestCase):
    def test_serialize_and_deserialize_rect_round_trip(self):
        spec = {
            "id": 17,
            "type": "rect",
            "points": [
                {"dt": pd.Timestamp("2026-04-17 09:30:00", tz="America/New_York"), "price": 3301.25},
                {"dt": pd.Timestamp("2026-04-17 11:00:00", tz="America/New_York"), "price": 3320.5},
            ],
        }

        payload = serialize_drawing_spec(spec)
        restored = deserialize_drawing_spec(payload)

        self.assertEqual(restored["id"], 17)
        self.assertEqual(restored["type"], "rect")
        self.assertEqual(restored["points"][0]["dt"], spec["points"][0]["dt"])
        self.assertEqual(restored["points"][1]["price"], 3320.5)

    def test_deserialize_rejects_unknown_type(self):
        with self.assertRaises(ValueError):
            deserialize_drawing_spec({"id": 1, "type": "ray", "points": []})


if __name__ == "__main__":
    unittest.main()
```

```python
# tests/test_drawing_tools.py
    def test_rect_session_completes_after_two_points(self):
        session = DrawingSession(TOOL_DEFINITIONS["rect"])

        self.assertIsNone(session.add_point(datetime.datetime(2026, 4, 17, 9, 30), 3301.25))
        spec = session.add_point(datetime.datetime(2026, 4, 17, 11, 0), 3320.50)

        self.assertEqual(spec["type"], "rect")
        self.assertEqual(len(spec["points"]), 2)

    def test_rect_preview_uses_cursor_as_second_point(self):
        session = DrawingSession(TOOL_DEFINITIONS["rect"])
        session.add_point(datetime.datetime(2026, 4, 17, 9, 30), 3301.25)

        spec = session.build_preview_spec(datetime.datetime(2026, 4, 17, 11, 0), 3320.50)

        self.assertEqual(spec["type"], "rect")
        self.assertEqual(spec["points"][1]["price"], 3320.50)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_drawing_specs.py tests/test_drawing_tools.py -q`

Expected: FAIL with an import error for `serialize_drawing_spec` / `deserialize_drawing_spec` and a `KeyError: 'rect'` from `TOOL_DEFINITIONS`.

- [ ] **Step 3: Write the minimal implementation**

```python
# ui/drawings/specs.py
from __future__ import annotations

import pandas as pd


SUPPORTED_DRAWING_TYPES = {"hline", "vline", "line", "fib", "fib_ext", "rect"}


def _normalize_point(point: dict) -> dict:
    dt = point.get("dt")
    return {
        "dt": None if dt is None else pd.Timestamp(dt),
        "price": float(point.get("price")),
    }


def _point_from_legacy(spec: dict, prefix: str) -> dict | None:
    dt_key = f"{prefix}_dt"
    price_key = f"{prefix}_price"
    if dt_key not in spec or price_key not in spec:
        return None
    return {"dt": spec.get(dt_key), "price": float(spec.get(price_key))}


def normalize_drawing_spec(spec: dict) -> dict:
    if "points" in spec:
        normalized = dict(spec)
        normalized["points"] = [_normalize_point(point) for point in spec["points"]]
        return normalized

    points = []
    for prefix in ("p1", "p2", "p3"):
        point = _point_from_legacy(spec, prefix)
        if point is not None:
            points.append(_normalize_point(point))

    normalized = dict(spec)
    normalized["points"] = points
    return normalized


def serialize_drawing_spec(spec: dict) -> dict:
    normalized = normalize_drawing_spec(spec)
    drawing_type = normalized.get("type")
    if drawing_type not in SUPPORTED_DRAWING_TYPES:
        raise ValueError(f"Unsupported drawing type: {drawing_type}")
    payload = {
        "id": int(normalized["id"]),
        "type": drawing_type,
        "points": [
            {
                "dt": None if point["dt"] is None else pd.Timestamp(point["dt"]).isoformat(),
                "price": float(point["price"]),
            }
            for point in normalized["points"]
        ],
    }
    if "config_snapshot" in normalized:
        payload["config_snapshot"] = dict(normalized["config_snapshot"])
    return payload


def deserialize_drawing_spec(payload: dict) -> dict:
    drawing_type = payload.get("type")
    if drawing_type not in SUPPORTED_DRAWING_TYPES:
        raise ValueError(f"Unsupported drawing type: {drawing_type}")
    restored = {
        "id": int(payload["id"]),
        "type": drawing_type,
        "points": [
            {
                "dt": None if point.get("dt") is None else pd.Timestamp(point["dt"]),
                "price": float(point["price"]),
            }
            for point in payload.get("points", [])
        ],
    }
    if "config_snapshot" in payload:
        restored["config_snapshot"] = dict(payload["config_snapshot"])
    return restored
```

```python
# ui/drawings/tools.py
TOOL_DEFINITIONS = {
    "hline": ToolDefinition(tool_id="hline", point_count=1),
    "vline": ToolDefinition(tool_id="vline", point_count=1),
    "line": ToolDefinition(tool_id="line", point_count=2),
    "fib": ToolDefinition(tool_id="fib", point_count=2),
    "fib_ext": ToolDefinition(tool_id="fib_ext", point_count=3),
    "rect": ToolDefinition(tool_id="rect", point_count=2),
}

    def build_preview_spec(self, dt, price: float) -> dict | None:
        if not self.points:
            return None

        preview_points = [*self.points, {"dt": dt, "price": float(price)}]
        if self.tool.tool_id in {"line", "fib", "rect"} and len(self.points) == 1:
            spec = {"type": self.tool.tool_id, "points": preview_points}
        elif self.tool.tool_id == "fib_ext" and len(self.points) == 1:
            spec = {"type": "line", "points": preview_points}
        elif self.tool.tool_id == "fib_ext" and len(self.points) == 2:
            spec = {"type": "fib_ext", "points": preview_points}
        else:
            return None

        if self.config_snapshot is not None:
            spec["config_snapshot"] = dict(self.config_snapshot)
        return spec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_drawing_specs.py tests/test_drawing_tools.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_drawing_specs.py tests/test_drawing_tools.py ui/drawings/specs.py ui/drawings/tools.py
git commit -m "feat: add rect drawing specs and sessions"
```

### Task 2: Add Global Drawing Store and QSettings Persistence

**Files:**
- Create: `ui/drawings/store.py`
- Create: `ui/drawings/persistence.py`
- Create: `tests/test_drawing_store.py`
- Create: `tests/test_drawing_persistence.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_drawing_store.py
import datetime
import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.store import GlobalDrawingStore


class GlobalDrawingStoreTests(unittest.TestCase):
    def test_add_update_delete_and_clear_drawings(self):
        store = GlobalDrawingStore()

        line = store.add_drawing(
            {
                "type": "line",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 17, 9, 30), "price": 3300.0},
                    {"dt": datetime.datetime(2026, 4, 17, 10, 0), "price": 3310.0},
                ],
            }
        )
        rect = store.add_drawing(
            {
                "type": "rect",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 17, 10, 0), "price": 3301.0},
                    {"dt": datetime.datetime(2026, 4, 17, 11, 0), "price": 3320.0},
                ],
            }
        )
        updated = store.upsert_drawing(
            {
                "id": line["id"],
                "type": "line",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 17, 9, 30), "price": 3298.0},
                    {"dt": datetime.datetime(2026, 4, 17, 10, 0), "price": 3310.0},
                ],
            }
        )

        self.assertEqual(line["id"], 1)
        self.assertEqual(rect["id"], 2)
        self.assertEqual(updated["points"][0]["price"], 3298.0)
        self.assertEqual(len(list(store.iter_drawings())), 2)

        store.remove_drawing(1)
        self.assertEqual([spec["id"] for spec in store.iter_drawings()], [2])

        store.clear()
        self.assertEqual(list(store.iter_drawings()), [])
```

```python
# tests/test_drawing_persistence.py
import json
import tempfile
import unittest
from pathlib import Path
import sys

from PyQt6.QtCore import QSettings


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.persistence import load_drawings, save_drawings


class DrawingPersistenceTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        drawings = [
            {
                "id": 7,
                "type": "rect",
                "points": [
                    {"dt": "2026-04-17T09:30:00-04:00", "price": 3301.25},
                    {"dt": "2026-04-17T11:00:00-04:00", "price": 3320.50},
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = QSettings(str(Path(tmpdir) / "drawings.ini"), QSettings.Format.IniFormat)
            save_drawings(settings, drawings)
            restored = load_drawings(settings)

        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["id"], 7)
        self.assertEqual(restored[0]["type"], "rect")

    def test_load_skips_malformed_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = QSettings(str(Path(tmpdir) / "drawings.ini"), QSettings.Format.IniFormat)
            settings.setValue(
                "drawings/items",
                json.dumps(
                    [
                        {"id": 1, "type": "line", "points": [{"dt": "2026-04-17T09:30:00", "price": 1.0}]},
                        {"id": 2, "type": "ray", "points": []},
                    ]
                ),
            )
            settings.sync()

            restored = load_drawings(settings)

        self.assertEqual([spec["id"] for spec in restored], [1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_drawing_store.py tests/test_drawing_persistence.py -q`

Expected: FAIL with `ModuleNotFoundError` for `ui.drawings.store` and `ui.drawings.persistence`.

- [ ] **Step 3: Write the minimal implementation**

```python
# ui/drawings/store.py
from __future__ import annotations

from collections import OrderedDict

from ui.drawings.specs import normalize_drawing_spec


class GlobalDrawingStore:
    def __init__(self, drawings: list[dict] | None = None):
        self._drawings: "OrderedDict[int, dict]" = OrderedDict()
        self._next_id = 1
        for spec in drawings or []:
            normalized = normalize_drawing_spec(spec)
            drawing_id = int(normalized["id"])
            self._drawings[drawing_id] = normalized
            self._next_id = max(self._next_id, drawing_id + 1)

    def add_drawing(self, spec: dict) -> dict:
        normalized = normalize_drawing_spec(spec)
        drawing_id = int(normalized.get("id") or self._next_id)
        normalized["id"] = drawing_id
        self._drawings[drawing_id] = normalized
        self._next_id = max(self._next_id, drawing_id + 1)
        return normalized

    def upsert_drawing(self, spec: dict) -> dict:
        normalized = normalize_drawing_spec(spec)
        drawing_id = int(normalized["id"])
        self._drawings[drawing_id] = normalized
        self._next_id = max(self._next_id, drawing_id + 1)
        return normalized

    def remove_drawing(self, drawing_id: int) -> None:
        self._drawings.pop(int(drawing_id), None)

    def clear(self) -> None:
        self._drawings.clear()

    def get_drawing(self, drawing_id: int) -> dict | None:
        return self._drawings.get(int(drawing_id))

    def iter_drawings(self):
        return list(self._drawings.values())
```

```python
# ui/drawings/persistence.py
from __future__ import annotations

import json

from ui.drawings.specs import deserialize_drawing_spec, serialize_drawing_spec


KEY_DRAWINGS_VERSION = "drawings/version"
KEY_DRAWINGS_ITEMS = "drawings/items"


def save_drawings(settings, drawings: list[dict]) -> None:
    payload = [serialize_drawing_spec(spec) for spec in drawings]
    settings.setValue(KEY_DRAWINGS_VERSION, 1)
    settings.setValue(KEY_DRAWINGS_ITEMS, json.dumps(payload))
    settings.sync()


def load_drawings(settings) -> list[dict]:
    raw = settings.value(KEY_DRAWINGS_ITEMS, "[]", type=str)
    try:
        payload = json.loads(raw)
    except Exception:
        return []

    restored: list[dict] = []
    for item in payload:
        try:
            restored.append(deserialize_drawing_spec(item))
        except Exception:
            continue
    return restored
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_drawing_store.py tests/test_drawing_persistence.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_drawing_store.py tests/test_drawing_persistence.py ui/drawings/store.py ui/drawings/persistence.py
git commit -m "feat: add global drawing store persistence"
```

### Task 3: Add Time-to-X Extrapolation Helpers

**Files:**
- Create: `ui/drawings/coords.py`
- Create: `tests/test_drawing_coords.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_drawing_coords.py
import unittest
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.coords import datetime_to_plot_x


class DrawingCoordinateTests(unittest.TestCase):
    def test_returns_exact_index_for_in_range_timestamp(self):
        index = pd.date_range("2026-04-17 09:30:00", periods=4, freq="30min", tz="America/New_York")

        x_value = datetime_to_plot_x(index, pd.Timedelta(minutes=30), pd.Timestamp("2026-04-17 10:30:00", tz="America/New_York"))

        self.assertEqual(x_value, 2.0)

    def test_extrapolates_left_and_right_without_clamping(self):
        index = pd.date_range("2026-04-17 09:30:00", periods=3, freq="30min", tz="America/New_York")

        left = datetime_to_plot_x(index, pd.Timedelta(minutes=30), pd.Timestamp("2026-04-17 09:00:00", tz="America/New_York"))
        right = datetime_to_plot_x(index, pd.Timedelta(minutes=30), pd.Timestamp("2026-04-17 11:00:00", tz="America/New_York"))

        self.assertEqual(left, -1.0)
        self.assertEqual(right, 3.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_drawing_coords.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'ui.drawings.coords'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# ui/drawings/coords.py
from __future__ import annotations

import pandas as pd


def _align_timestamp(index: pd.Index, dt) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if getattr(index, "tz", None) is None and ts.tzinfo is not None:
        return ts.tz_convert("America/New_York").tz_localize(None)
    if getattr(index, "tz", None) is not None and ts.tzinfo is None:
        return ts.tz_localize(index.tz)
    return ts


def datetime_to_plot_x(index: pd.Index, default_delta: pd.Timedelta, dt) -> float | None:
    if dt is None or index is None or len(index) == 0:
        return None

    ts = _align_timestamp(index, dt)
    delta = default_delta or pd.Timedelta(minutes=1)

    exact_idx = int(index.searchsorted(ts))
    if exact_idx < len(index) and index[exact_idx] == ts:
        return float(exact_idx)

    if ts < index[0]:
        return float((ts - index[0]) / delta)

    if ts > index[-1]:
        return float((len(index) - 1) + ((ts - index[-1]) / delta))

    right_idx = int(index.searchsorted(ts))
    left_idx = max(0, right_idx - 1)
    left = index[left_idx]
    right = index[right_idx]
    span = right - left
    if span == pd.Timedelta(0):
        return float(left_idx)
    return float(left_idx + ((ts - left) / span))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_drawing_coords.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_drawing_coords.py ui/drawings/coords.py
git commit -m "feat: add drawing coordinate extrapolation"
```

### Task 4: Add Editable Item Classes and Rectangle Rendering

**Files:**
- Create: `ui/drawings/editable_items.py`
- Modify: `ui/drawings/renderers.py:1-152`
- Modify: `tests/test_drawing_renderers.py:1-104`
- Create: `tests/test_editable_drawing_items.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_drawing_renderers.py
    def test_render_spec_items_for_rect_creates_outline(self):
        plot = FakePlotItem()

        items = render_spec_items(
            plot,
            {
                "id": 12,
                "type": "rect",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 17, 9, 30), "price": 3301.25},
                    {"dt": datetime.datetime(2026, 4, 17, 11, 0), "price": 3320.50},
                ],
            },
            x_from_datetime=lambda dt: 10 if dt.hour == 9 else 20,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(len(plot.items), 1)
```

```python
# tests/test_editable_drawing_items.py
import datetime
import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.editable_items import EditableLineItem, EditableRectItem


class FakePlotItem:
    def __init__(self):
        self.items = []

    def addItem(self, item, ignoreBounds=False):
        self.items.append((item, ignoreBounds))

    def removeItem(self, item):
        self.items = [(existing, flag) for existing, flag in self.items if existing is not item]


class EditableDrawingItemTests(unittest.TestCase):
    def test_line_drag_handle_updates_only_target_point(self):
        spec = {
            "id": 1,
            "type": "line",
            "points": [
                {"dt": datetime.datetime(2026, 4, 17, 9, 30), "price": 3300.0},
                {"dt": datetime.datetime(2026, 4, 17, 10, 0), "price": 3310.0},
            ],
        }
        item = EditableLineItem(FakePlotItem(), lambda dt: 10.0, spec)

        updated = item.drag_handle("p1", datetime.datetime(2026, 4, 17, 10, 15), 3315.0)

        self.assertEqual(updated["points"][0]["price"], 3300.0)
        self.assertEqual(updated["points"][1]["price"], 3315.0)

    def test_rect_drag_body_translates_both_points(self):
        spec = {
            "id": 2,
            "type": "rect",
            "points": [
                {"dt": datetime.datetime(2026, 4, 17, 9, 30), "price": 3301.0},
                {"dt": datetime.datetime(2026, 4, 17, 11, 0), "price": 3320.0},
            ],
        }
        item = EditableRectItem(FakePlotItem(), lambda dt: 10.0, spec)

        updated = item.drag_body(datetime.timedelta(minutes=30), 5.0)

        self.assertEqual(updated["points"][0]["dt"], datetime.datetime(2026, 4, 17, 10, 0))
        self.assertEqual(updated["points"][1]["price"], 3325.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_drawing_renderers.py tests/test_editable_drawing_items.py -q`

Expected: FAIL because `rect` is not rendered yet and `ui.drawings.editable_items` does not exist.

- [ ] **Step 3: Write the minimal implementation**

```python
# ui/drawings/renderers.py
from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import Qt

from ui.drawings.fib_math import build_extension_levels, build_retracement_levels
from ui.drawings.specs import normalize_drawing_spec


def _tag_drawing_item(item, drawing_id: int | None) -> None:
    item._is_drawing = True
    item._drawing_id = drawing_id


def _line_pen(preview: bool = False, dashed: bool = False, color: str | None = None, width: float = 2.0):
    pen_color = color or ("#00E5FF" if preview else "#FFD54A")
    style = Qt.PenStyle.DashLine if dashed or preview else Qt.PenStyle.SolidLine
    return pg.mkPen(pen_color, width=width, style=style)


def build_render_plan(spec: dict) -> dict:
    normalized = normalize_drawing_spec(spec)
    levels = list(normalized.get("config_snapshot", {}).get("levels", []))
    points = normalized["points"]
    rows = []

    if normalized["type"] == "fib" and len(points) >= 2:
        rows = [{"ratio": level.ratio, "price": level.price} for level in build_retracement_levels(points[0]["price"], points[1]["price"], levels)]
    elif normalized["type"] == "fib_ext" and len(points) >= 3:
        rows = [{"ratio": level.ratio, "price": level.price} for level in build_extension_levels(points[0]["price"], points[1]["price"], points[2]["price"], levels)]

    return {"id": normalized.get("id"), "type": normalized["type"], "points": points, "levels": levels, "rows": rows}


def render_spec_items(plot_item, spec: dict, x_from_datetime, preview: bool = False, selected: bool = False) -> list[object]:
    normalized = normalize_drawing_spec(spec)
    drawing_id = normalized.get("id")
    points = normalized["points"]
    dtype = normalized["type"]
    items: list[object] = []
    body_width = 3.0 if selected and not preview else 2.0

    def add_item(item, ignore_bounds: bool = False):
        _tag_drawing_item(item, drawing_id)
        plot_item.addItem(item, ignoreBounds=ignore_bounds)
        items.append(item)

    if dtype == "rect" and len(points) >= 2:
        x1 = x_from_datetime(points[0]["dt"])
        x2 = x_from_datetime(points[1]["dt"])
        if x1 is None or x2 is None:
            return []
        left, right = sorted([x1, x2])
        low, high = sorted([points[0]["price"], points[1]["price"]])
        rect = pg.PlotCurveItem(
            x=[left, right, right, left, left],
            y=[high, high, low, low, high],
            pen=_line_pen(preview=preview, color="#4DD0E1", width=body_width),
        )
        add_item(rect)
        return items

    if dtype == "hline" and len(points) >= 1:
        line = pg.InfiniteLine(angle=0, pos=points[0]["price"], pen=_line_pen(preview=preview, color="#FF4444", width=body_width))
        add_item(line)
        return items

    if dtype == "vline" and len(points) >= 1:
        x = x_from_datetime(points[0]["dt"])
        if x is None:
            return []
        line = pg.InfiniteLine(angle=90, pos=x, pen=_line_pen(preview=preview, color="#FF4444", width=body_width))
        add_item(line)
        return items

    if dtype == "line" and len(points) >= 2:
        x_values = [x_from_datetime(point["dt"]) for point in points[:2]]
        if any(x is None for x in x_values):
            return []
        curve = pg.PlotCurveItem(
            x=x_values,
            y=[points[0]["price"], points[1]["price"]],
            pen=_line_pen(preview=preview, color="#00E5FF", width=body_width),
        )
        add_item(curve)
        return items

    plan = build_render_plan(normalized)

    if dtype == "fib" and len(points) >= 2:
        x_values = [x_from_datetime(point["dt"]) for point in points[:2]]
        if any(x is None for x in x_values):
            return []
        x_left = min(x_values)
        x_right = max(x_values)

        for point in points[:2]:
            edge = pg.PlotCurveItem(
                x=[x_left, x_right],
                y=[point["price"], point["price"]],
                pen=_line_pen(preview=preview, dashed=False, width=body_width),
            )
            add_item(edge)

        for row in plan["rows"]:
            level = pg.PlotCurveItem(
                x=[x_left, x_right],
                y=[row["price"], row["price"]],
                pen=_line_pen(preview=preview, dashed=True, width=body_width),
            )
            add_item(level)
            if preview:
                continue
            label = pg.TextItem(
                text=f'{row["ratio"]:.3f}  {row["price"]:.3f}',
                color="#FFD54A",
                fill=pg.mkBrush(20, 20, 20, 220),
                anchor=(0, 0.5),
            )
            label.setPos(x_right, row["price"])
            label.setZValue(21)
            add_item(label, ignore_bounds=True)
        return items

    if dtype == "fib_ext" and len(points) >= 3:
        x_a = x_from_datetime(points[0]["dt"])
        x_b = x_from_datetime(points[1]["dt"])
        x_c = x_from_datetime(points[2]["dt"])
        if any(x is None for x in (x_a, x_b, x_c)):
            return []

        guide_pen = _line_pen(preview=preview, dashed=True, color="#B38F00", width=1.6 if not selected else 2.0)
        ab = pg.PlotCurveItem(x=[x_a, x_b], y=[points[0]["price"], points[1]["price"]], pen=guide_pen)
        bc = pg.PlotCurveItem(x=[x_b, x_c], y=[points[1]["price"], points[2]["price"]], pen=guide_pen)
        add_item(ab)
        add_item(bc)

        projection_span = max(abs(x_b - x_a), 1.0)
        x_right = x_c + projection_span
        for row in plan["rows"]:
            level = pg.PlotCurveItem(
                x=[x_c, x_right],
                y=[row["price"], row["price"]],
                pen=_line_pen(preview=preview, dashed=True, width=body_width),
            )
            add_item(level)
            if preview:
                continue
            label = pg.TextItem(
                text=f'{row["ratio"]:.3f}  {row["price"]:.3f}',
                color="#FFD54A",
                fill=pg.mkBrush(20, 20, 20, 220),
                anchor=(0, 0.5),
            )
            label.setPos(x_right, row["price"])
            label.setZValue(21)
            add_item(label, ignore_bounds=True)
        return items

    return items
```

```python
# ui/drawings/editable_items.py
from __future__ import annotations

import datetime
from dataclasses import dataclass

import pandas as pd
import pyqtgraph as pg

from ui.drawings.renderers import render_spec_items


def _shift_point(point: dict, delta_dt: datetime.timedelta, delta_price: float) -> dict:
    return {
        "dt": pd.Timestamp(point["dt"]).to_pydatetime() + delta_dt,
        "price": float(point["price"]) + float(delta_price),
    }


@dataclass(frozen=True)
class HandlePoint:
    handle_id: str
    dt
    price: float


class EditableDrawingItem:
    def __init__(self, plot_item, x_from_datetime, spec: dict):
        self.plot_item = plot_item
        self.x_from_datetime = x_from_datetime
        self.spec = spec
        self.selected = False
        self.body_items: list[object] = []
        self.handle_item = None
        self.refresh()

    def set_spec(self, spec: dict) -> None:
        self.spec = spec
        self.refresh()

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self.refresh()

    def refresh(self) -> None:
        self.remove_from_plot()
        self.body_items = render_spec_items(self.plot_item, self.spec, self.x_from_datetime, selected=self.selected)
        handle_points = self.handle_points()
        if self.selected and handle_points:
            self.handle_item = pg.ScatterPlotItem(
                x=[self.x_from_datetime(point.dt) for point in handle_points],
                y=[point.price for point in handle_points],
                size=10,
                brush=pg.mkBrush("#FFFFFF"),
                pen=pg.mkPen("#000000", width=1),
            )
            self.plot_item.addItem(self.handle_item, ignoreBounds=True)

    def remove_from_plot(self) -> None:
        for item in self.body_items:
            try:
                self.plot_item.removeItem(item)
            except Exception:
                pass
        self.body_items = []
        if self.handle_item is not None:
            try:
                self.plot_item.removeItem(self.handle_item)
            except Exception:
                pass
            self.handle_item = None

    def handle_points(self) -> list[HandlePoint]:
        return []

    def drag_body(self, delta_dt: datetime.timedelta, delta_price: float) -> dict:
        return {**self.spec, "points": [_shift_point(point, delta_dt, delta_price) for point in self.spec["points"]]}


class EditableLineItem(EditableDrawingItem):
    def handle_points(self) -> list[HandlePoint]:
        return [
            HandlePoint("p0", self.spec["points"][0]["dt"], self.spec["points"][0]["price"]),
            HandlePoint("p1", self.spec["points"][1]["dt"], self.spec["points"][1]["price"]),
        ]

    def drag_handle(self, handle_id: str, dt, price: float) -> dict:
        points = [dict(point) for point in self.spec["points"]]
        index = 0 if handle_id == "p0" else 1
        points[index] = {"dt": dt, "price": float(price)}
        return {**self.spec, "points": points}


class EditableFibItem(EditableLineItem):
    pass


class EditableFibExtensionItem(EditableDrawingItem):
    def handle_points(self) -> list[HandlePoint]:
        return [
            HandlePoint("p0", self.spec["points"][0]["dt"], self.spec["points"][0]["price"]),
            HandlePoint("p1", self.spec["points"][1]["dt"], self.spec["points"][1]["price"]),
            HandlePoint("p2", self.spec["points"][2]["dt"], self.spec["points"][2]["price"]),
        ]

    def drag_handle(self, handle_id: str, dt, price: float) -> dict:
        index = {"p0": 0, "p1": 1, "p2": 2}[handle_id]
        points = [dict(point) for point in self.spec["points"]]
        points[index] = {"dt": dt, "price": float(price)}
        return {**self.spec, "points": points}


class EditableRectItem(EditableDrawingItem):
    def drag_handle(self, handle_id: str, dt, price: float) -> dict:
        first = dict(self.spec["points"][0])
        second = dict(self.spec["points"][1])
        if handle_id in {"p0", "top_left", "bottom_left"}:
            first["dt"] = dt
        else:
            second["dt"] = dt
        if handle_id in {"p0", "top_left", "top_right"}:
            first["price"] = float(price)
        else:
            second["price"] = float(price)
        return {**self.spec, "points": [first, second]}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_drawing_renderers.py tests/test_editable_drawing_items.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_drawing_renderers.py tests/test_editable_drawing_items.py ui/drawings/renderers.py ui/drawings/editable_items.py
git commit -m "feat: add editable drawing items"
```

### Task 5: Add Controller and Per-Chart Drawing Layers

**Files:**
- Create: `ui/drawings/controller.py`
- Create: `ui/drawings/layer.py`
- Create: `tests/test_chart_drawing_layer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chart_drawing_layer.py
import datetime
import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ui.drawings.controller import DrawingController
from ui.drawings.layer import ChartDrawingLayer
from ui.drawings.store import GlobalDrawingStore


class FakePlotItem:
    def __init__(self):
        self.items = []

    def addItem(self, item, ignoreBounds=False):
        self.items.append((item, ignoreBounds))

    def removeItem(self, item):
        self.items = [(existing, flag) for existing, flag in self.items if existing is not item]


class ChartDrawingLayerTests(unittest.TestCase):
    def test_upsert_updates_only_target_drawing(self):
        store = GlobalDrawingStore()
        controller = DrawingController(store, persist_callback=lambda drawings: None)
        layer = ChartDrawingLayer(FakePlotItem(), lambda dt: 10.0, controller)
        controller.register_layer(layer)

        line = controller.add_drawing(
            {
                "type": "line",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 17, 9, 30), "price": 3300.0},
                    {"dt": datetime.datetime(2026, 4, 17, 10, 0), "price": 3310.0},
                ],
            }
        )
        rect = controller.add_drawing(
            {
                "type": "rect",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 17, 10, 0), "price": 3301.0},
                    {"dt": datetime.datetime(2026, 4, 17, 11, 0), "price": 3320.0},
                ],
            }
        )

        controller.update_drawing(
            {
                "id": rect["id"],
                "type": "rect",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 17, 10, 30), "price": 3301.0},
                    {"dt": datetime.datetime(2026, 4, 17, 11, 30), "price": 3320.0},
                ],
            }
        )

        self.assertEqual(sorted(layer.items.keys()), [line["id"], rect["id"]])
        self.assertEqual(layer.items[rect["id"]].spec["points"][0]["dt"], datetime.datetime(2026, 4, 17, 10, 30))

    def test_select_drawing_propagates_selected_id(self):
        store = GlobalDrawingStore()
        controller = DrawingController(store, persist_callback=lambda drawings: None)
        layer = ChartDrawingLayer(FakePlotItem(), lambda dt: 10.0, controller)
        controller.register_layer(layer)

        line = controller.add_drawing(
            {
                "type": "line",
                "points": [
                    {"dt": datetime.datetime(2026, 4, 17, 9, 30), "price": 3300.0},
                    {"dt": datetime.datetime(2026, 4, 17, 10, 0), "price": 3310.0},
                ],
            }
        )

        controller.select_drawing(line["id"])

        self.assertEqual(layer.selected_drawing_id, line["id"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_chart_drawing_layer.py -q`

Expected: FAIL with `ModuleNotFoundError` for `ui.drawings.controller` and `ui.drawings.layer`.

- [ ] **Step 3: Write the minimal implementation**

```python
# ui/drawings/layer.py
from __future__ import annotations

from ui.drawings.editable_items import (
    EditableFibExtensionItem,
    EditableFibItem,
    EditableLineItem,
    EditableRectItem,
)


ITEM_TYPES = {
    "line": EditableLineItem,
    "fib": EditableFibItem,
    "fib_ext": EditableFibExtensionItem,
    "rect": EditableRectItem,
}


class ChartDrawingLayer:
    def __init__(self, plot_item, x_from_datetime, controller):
        self.plot_item = plot_item
        self.x_from_datetime = x_from_datetime
        self.controller = controller
        self.items: dict[int, object] = {}
        self.selected_drawing_id: int | None = None

    def upsert_drawing(self, spec: dict) -> None:
        drawing_id = int(spec["id"])
        item = self.items.get(drawing_id)
        if item is None:
            item = ITEM_TYPES[spec["type"]](self.plot_item, self.x_from_datetime, spec)
            self.items[drawing_id] = item
        else:
            item.set_spec(spec)
        item.set_selected(drawing_id == self.selected_drawing_id)

    def remove_drawing(self, drawing_id: int) -> None:
        item = self.items.pop(int(drawing_id), None)
        if item is not None:
            item.remove_from_plot()

    def clear(self) -> None:
        for drawing_id in list(self.items.keys()):
            self.remove_drawing(drawing_id)

    def set_selected_drawing(self, drawing_id: int | None) -> None:
        self.selected_drawing_id = drawing_id
        for existing_id, item in self.items.items():
            item.set_selected(existing_id == drawing_id)
```

```python
# ui/drawings/controller.py
from __future__ import annotations


class DrawingController:
    def __init__(self, store, persist_callback):
        self.store = store
        self.persist_callback = persist_callback
        self.layers: list[object] = []
        self.selected_drawing_id: int | None = None

    def register_layer(self, layer) -> None:
        self.layers.append(layer)
        for spec in self.store.iter_drawings():
            layer.upsert_drawing(spec)
        layer.set_selected_drawing(self.selected_drawing_id)

    def _persist(self) -> None:
        self.persist_callback(self.store.iter_drawings())

    def _broadcast_upsert(self, spec: dict) -> None:
        for layer in self.layers:
            layer.upsert_drawing(spec)

    def _broadcast_remove(self, drawing_id: int) -> None:
        for layer in self.layers:
            layer.remove_drawing(drawing_id)

    def add_drawing(self, spec: dict) -> dict:
        stored = self.store.add_drawing(spec)
        self._broadcast_upsert(stored)
        self._persist()
        return stored

    def update_drawing(self, spec: dict) -> dict:
        stored = self.store.upsert_drawing(spec)
        self._broadcast_upsert(stored)
        self._persist()
        return stored

    def delete_drawing(self, drawing_id: int) -> None:
        self.store.remove_drawing(drawing_id)
        self._broadcast_remove(drawing_id)
        if self.selected_drawing_id == drawing_id:
            self.select_drawing(None)
        self._persist()

    def clear_drawings(self) -> None:
        self.store.clear()
        for layer in self.layers:
            layer.clear()
            layer.set_selected_drawing(None)
        self.selected_drawing_id = None
        self._persist()

    def select_drawing(self, drawing_id: int | None) -> None:
        self.selected_drawing_id = drawing_id
        for layer in self.layers:
            layer.set_selected_drawing(drawing_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_chart_drawing_layer.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_chart_drawing_layer.py ui/drawings/controller.py ui/drawings/layer.py
git commit -m "feat: add drawing controller and layers"
```

### Task 6: Integrate the Shared Drawing System into ChartWidget and MainWindow

**Files:**
- Modify: `ui/main_window.py:160-1045`
- Modify: `ui/main_window.py:1259-1655`
- Modify: `tests/test_drawing_chart_widget.py:1-56`
- Run: `tests/test_session_state.py`

- [ ] **Step 1: Write the failing integration tests**

```python
# tests/test_drawing_chart_widget.py
    def test_set_draw_mode_supports_rect(self):
        chart = ChartWidget("1min")

        chart.set_draw_mode("rect")

        self.assertEqual(chart.active_drawing_session.tool.tool_id, "rect")

    def test_x_from_datetime_extrapolates_beyond_loaded_data(self):
        chart = ChartWidget("1min")
        index = pd.to_datetime(["2026-04-17 09:30:00", "2026-04-17 10:00:00", "2026-04-17 10:30:00"])
        chart.full_df = pd.DataFrame({"close": [3300.0, 3310.0, 3320.0]}, index=index)
        chart.time_axis.set_datetime_index(index)

        left = chart._x_from_datetime(pd.Timestamp("2026-04-17 09:00:00"))
        right = chart._x_from_datetime(pd.Timestamp("2026-04-17 11:00:00"))

        self.assertEqual(left, -1.0)
        self.assertEqual(right, 3.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_drawing_chart_widget.py tests/test_session_state.py -q`

Expected: FAIL because `rect` mode is not wired into `ChartWidget` and `_x_from_datetime()` still clamps to the first/last bar.

- [ ] **Step 3: Write the minimal implementation**

```python
# ui/main_window.py imports
from ui.drawings.controller import DrawingController
from ui.drawings.coords import datetime_to_plot_x
from ui.drawings.layer import ChartDrawingLayer
from ui.drawings.persistence import load_drawings, save_drawings
from ui.drawings.store import GlobalDrawingStore
```

```python
# ui/main_window.py inside ChartWidget.__init__
        self.btn_draw_rect = QPushButton("Rect")
        for btn in [
            self.btn_draw_select,
            self.btn_draw_hline,
            self.btn_draw_vline,
            self.btn_draw_line,
            self.btn_draw_fib,
            self.btn_draw_fib_ext,
            self.btn_draw_rect,
            self.btn_draw_fib_config,
            self.btn_draw_clear,
        ]:
            width = 44
            if btn in (self.btn_draw_fib_ext, self.btn_draw_fib_config):
                width = 76
            if btn is self.btn_draw_rect:
                width = 52
            btn.setFixedSize(width, 30)
            btn.setStyleSheet(
                """
                QPushButton {
                    border: 1px solid #444;
                    background-color: #222;
                    color: #AAA;
                    border-radius: 2px;
                }
                QPushButton:hover {
                    background-color: #333;
                    color: white;
                }
                """
            )

        self.btn_draw_rect.clicked.connect(lambda: self.set_draw_mode("rect"))
        self.toolbar_layout.addWidget(self.btn_draw_rect)
        self.drawing_layer = None
```

```python
# ui/main_window.py ChartWidget methods
    def attach_drawing_controller(self, controller):
        self.drawing_layer = ChartDrawingLayer(self.ax, self._x_from_datetime, controller)
        controller.register_layer(self.drawing_layer)

    def _x_from_datetime(self, dt):
        if dt is None or self.full_df is None or self.full_df.empty:
            return None
        default_delta = pd.Timedelta(self.time_axis._delta) if self.time_axis._delta is not None else pd.Timedelta(minutes=1)
        return datetime_to_plot_x(self.full_df.index, default_delta, dt)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.drawing_layer is not None:
            self.parent().drawing_controller.delete_drawing(self.parent().drawing_controller.selected_drawing_id)
            event.accept()
            return
        super().keyPressEvent(event)
```

```python
# ui/main_window.py inside MainWindow.__init__
        self.settings = QSettings("TradeReview", "TradeReview")
        persisted_drawings = load_drawings(self.settings)
        self.drawing_store = GlobalDrawingStore(persisted_drawings)
        self.drawing_controller = DrawingController(
            self.drawing_store,
            persist_callback=lambda drawings: save_drawings(self.settings, drawings),
        )
```

```python
# ui/main_window.py inside init_charts
            chart.attach_drawing_controller(self.drawing_controller)
```

```python
# ui/main_window.py drawing handlers
    def on_drawing_request(self, spec):
        self.drawing_controller.add_drawing(spec)

    def on_drawing_delete(self, draw_id):
        self.drawing_controller.delete_drawing(draw_id)

    def on_drawing_clear(self):
        self.drawing_controller.clear_drawings()
```

- [ ] **Step 4: Run the targeted regression suite**

Run: `python -m pytest tests/test_drawing_specs.py tests/test_drawing_tools.py tests/test_drawing_store.py tests/test_drawing_persistence.py tests/test_drawing_coords.py tests/test_drawing_renderers.py tests/test_editable_drawing_items.py tests/test_chart_drawing_layer.py tests/test_drawing_chart_widget.py tests/test_session_state.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/main_window.py tests/test_drawing_chart_widget.py
git commit -m "feat: integrate editable persisted drawings"
```

## Self-Review

- Spec coverage:
  - direct editing is covered by Tasks 4, 5, and 6
  - rectangle creation is covered by Tasks 1, 4, and 6
  - global persistence is covered by Tasks 2 and 6
  - all-period synchronization is covered by Tasks 5 and 6
  - time-based extrapolated coordinates are covered by Task 3 and Task 6
- Placeholder scan:
  - no unresolved markers remain
  - each task includes exact files, test code, commands, and commit messages
- Type consistency:
  - the plan consistently uses `GlobalDrawingStore`, `DrawingController`, `ChartDrawingLayer`, `Editable*Item`, `serialize_drawing_spec`, and `datetime_to_plot_x`
