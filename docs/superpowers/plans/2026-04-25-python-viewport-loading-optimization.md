# Python Viewport Loading Performance Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current Python/PyQtGraph TradeReview UI browse multi-year DuckDB data smoothly by loading chart data by viewport instead of by full dataset.

**Architecture:** Keep the existing PyQt6/PyQtGraph UI, but move normal browsing to a DuckDB-backed window query model. `DataEngine` exposes metadata and range queries, `ChartWidget` renders only the active window plus buffer, and a small LOD resolver chooses an appropriate candle period for wide time ranges.

**Tech Stack:** Python, PyQt6, PyQtGraph, pandas, DuckDB, pytest

---

## File Structure

Create:

- `ui/chart_lod.py`: LOD period selection based on visible time span and target render density.
- `ui/chart_windowing.py`: pure helpers for loaded-window range math, buffer thresholds, and stale generation checks.
- `tests/test_chart_lod.py`: unit tests for period selection.
- `tests/test_chart_windowing.py`: unit tests for buffer and reload decisions.
- `tests/test_data_engine_window_queries.py`: data-layer tests using a temporary DuckDB file.

Modify:

- `engine/data_engine.py`: split DuckDB metadata loading from full tick loading; add range query methods.
- `ui/services/data_loading.py`: derive initial time from metadata when available.
- `ui/chart_widget.py`: accept and render windowed data; keep full-data mode until migration is complete.
- `ui/main_window.py`: request chart windows by visible time range instead of passing full candle DataFrames in normal browsing mode.
- `tests/test_data_loading_facade.py`: cover metadata-backed initial time.
- `tests/test_chart_performance.py`: keep existing visible-slice tests and add compatibility checks where useful.

## Task 1: Add Pure Windowing Helpers

**Files:**
- Create: `ui/chart_windowing.py`
- Create: `tests/test_chart_windowing.py`

- [ ] **Step 1: Write tests for buffer behavior**

```python
import unittest

import pandas as pd

from ui.chart_windowing import (
    build_query_window,
    is_view_inside_loaded_window,
    should_prefetch_window,
)


class ChartWindowingTests(unittest.TestCase):
    def test_build_query_window_adds_buffer_on_both_sides(self):
        start = pd.Timestamp("2026-04-01 10:00:00")
        end = pd.Timestamp("2026-04-01 11:00:00")

        query_start, query_end = build_query_window(start, end, buffer_multiplier=2)

        self.assertEqual(query_start, pd.Timestamp("2026-04-01 08:00:00"))
        self.assertEqual(query_end, pd.Timestamp("2026-04-01 13:00:00"))

    def test_view_inside_loaded_window_returns_true(self):
        self.assertTrue(
            is_view_inside_loaded_window(
                pd.Timestamp("2026-04-01 10:00:00"),
                pd.Timestamp("2026-04-01 11:00:00"),
                pd.Timestamp("2026-04-01 08:00:00"),
                pd.Timestamp("2026-04-01 13:00:00"),
            )
        )

    def test_prefetch_when_view_nears_right_edge(self):
        self.assertTrue(
            should_prefetch_window(
                pd.Timestamp("2026-04-01 11:30:00"),
                pd.Timestamp("2026-04-01 12:30:00"),
                pd.Timestamp("2026-04-01 08:00:00"),
                pd.Timestamp("2026-04-01 13:00:00"),
                edge_fraction=0.5,
            )
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_chart_windowing.py
```

Expected: FAIL because `ui.chart_windowing` does not exist.

- [ ] **Step 3: Implement the helpers**

```python
import pandas as pd


def _span(view_start, view_end):
    span = pd.Timestamp(view_end) - pd.Timestamp(view_start)
    if span <= pd.Timedelta(0):
        return pd.Timedelta(minutes=1)
    return span


def build_query_window(view_start, view_end, buffer_multiplier=2):
    view_start = pd.Timestamp(view_start)
    view_end = pd.Timestamp(view_end)
    buffer = _span(view_start, view_end) * buffer_multiplier
    return view_start - buffer, view_end + buffer


def is_view_inside_loaded_window(view_start, view_end, loaded_start, loaded_end):
    return pd.Timestamp(loaded_start) <= pd.Timestamp(view_start) and pd.Timestamp(view_end) <= pd.Timestamp(loaded_end)


def should_prefetch_window(view_start, view_end, loaded_start, loaded_end, edge_fraction=0.5):
    view_start = pd.Timestamp(view_start)
    view_end = pd.Timestamp(view_end)
    loaded_start = pd.Timestamp(loaded_start)
    loaded_end = pd.Timestamp(loaded_end)
    margin = _span(view_start, view_end) * edge_fraction
    return view_start <= loaded_start + margin or view_end >= loaded_end - margin
```

- [ ] **Step 4: Run tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_chart_windowing.py
```

Expected: PASS.

Commit:

```powershell
git add ui/chart_windowing.py tests/test_chart_windowing.py
git commit -m "test: 补充图表窗口缓存判断"
```

## Task 2: Add LOD Period Resolver

**Files:**
- Create: `ui/chart_lod.py`
- Create: `tests/test_chart_lod.py`

- [ ] **Step 1: Write LOD tests**

```python
import unittest

import pandas as pd

from ui.chart_lod import choose_lod_period


class ChartLODTests(unittest.TestCase):
    def test_short_intraday_range_keeps_current_low_period(self):
        period = choose_lod_period(
            requested_period="1min",
            view_start=pd.Timestamp("2026-04-01 09:00:00"),
            view_end=pd.Timestamp("2026-04-01 15:00:00"),
            pixel_width=1600,
        )

        self.assertEqual(period, "1min")

    def test_multi_month_range_uses_hourly_period(self):
        period = choose_lod_period(
            requested_period="1min",
            view_start=pd.Timestamp("2026-01-01"),
            view_end=pd.Timestamp("2026-04-01"),
            pixel_width=1600,
        )

        self.assertEqual(period, "1h")

    def test_multi_year_range_uses_daily_period(self):
        period = choose_lod_period(
            requested_period="30s",
            view_start=pd.Timestamp("2021-01-01"),
            view_end=pd.Timestamp("2026-01-01"),
            pixel_width=1600,
        )

        self.assertEqual(period, "1D")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement the resolver**

```python
import pandas as pd


def choose_lod_period(requested_period, view_start, view_end, pixel_width=1600, max_points_per_pixel=2.0):
    span = pd.Timestamp(view_end) - pd.Timestamp(view_start)
    if span <= pd.Timedelta(days=2):
        return requested_period
    if span <= pd.Timedelta(days=30):
        return "5min"
    if span <= pd.Timedelta(days=180):
        return "1h"
    if span <= pd.Timedelta(days=730):
        return "4h"
    return "1D"
```

- [ ] **Step 3: Run tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_chart_lod.py
```

Expected: PASS.

Commit:

```powershell
git add ui/chart_lod.py tests/test_chart_lod.py
git commit -m "feat: 增加图表LOD周期选择"
```

## Task 3: Stop Full Tick Loading for DuckDB Metadata

**Files:**
- Modify: `engine/data_engine.py`
- Modify: `ui/services/data_loading.py`
- Modify: `tests/test_data_loading_facade.py`
- Create: `tests/test_data_engine_window_queries.py`

- [ ] **Step 1: Write DuckDB metadata test**

Create a temporary DuckDB with `ticks` and one candle table. The test should assert that loading metadata exposes start/end/count without materializing all tick rows into `df_ticks`.

```python
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import duckdb
import pandas as pd

from engine.data_engine import DataEngine


class DataEngineWindowQueryTests(unittest.TestCase):
    def test_duckdb_load_keeps_tick_dataframe_unloaded_and_exposes_metadata(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "sample.duckdb"
            con = duckdb.connect(str(db_path))
            con.execute(
                "CREATE TABLE ticks AS SELECT * FROM (VALUES "
                "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 1.0),"
                "(TIMESTAMP '2026-04-01 09:01:00', 101.0, 2.0)"
                ") AS t(timestamp, price, volume)"
            )
            con.execute(
                "CREATE TABLE candles_1m AS SELECT * FROM (VALUES "
                "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 101.0, 99.0, 100.5, 1.0),"
                "(TIMESTAMP '2026-04-01 09:01:00', 100.5, 102.0, 100.0, 101.5, 2.0)"
                ") AS t(timestamp, open, high, low, close, volume)"
            )
            con.close()

            engine = DataEngine(parquet_file=str(db_path))

            self.assertIsNone(engine.df_ticks)
            self.assertEqual(engine.tick_count, 2)
            self.assertEqual(engine.tick_start, pd.Timestamp("2026-04-01 09:00:00"))
            self.assertEqual(engine.tick_end, pd.Timestamp("2026-04-01 09:01:00"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Add metadata fields to `DataEngine`**

Add these instance attributes in `DataEngine.__init__`:

```python
self.tick_start = None
self.tick_end = None
self.tick_count = 0
```

In `load_data()`, when `ext == ".duckdb"`, replace the eager `SELECT * FROM ticks ORDER BY timestamp` path with a metadata query:

```python
row = con.execute(
    "SELECT count(*) AS row_count, min(timestamp) AS min_ts, max(timestamp) AS max_ts FROM ticks"
).fetchone()
self.tick_count = int(row[0] or 0)
self.tick_start = pd.Timestamp(row[1]) if row[1] is not None else None
self.tick_end = pd.Timestamp(row[2]) if row[2] is not None else None
self.df_ticks = None
```

Keep Parquet behavior unchanged in this task.

- [ ] **Step 3: Update `DataLoadingFacade`**

Allow DuckDB metadata to count as successful load:

```python
if df_ticks is None or df_ticks.empty:
    if getattr(self.engine, "_duckdb_path", None) and getattr(self.engine, "tick_count", 0) > 0:
        initial_time = self.engine.tick_start
        return DataLoadResult(
            success=True,
            file_path=file_path,
            initial_time=initial_time,
            warnings=tuple(self.engine.last_load_warnings or ()),
        )
    return DataLoadResult(...)
```

- [ ] **Step 4: Run focused tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_data_engine_window_queries.py tests/test_data_loading_facade.py
```

Expected: PASS.

Commit:

```powershell
git add engine/data_engine.py ui/services/data_loading.py tests/test_data_engine_window_queries.py tests/test_data_loading_facade.py
git commit -m "refactor: 使用DuckDB元数据加载行情范围"
```

## Task 4: Add DuckDB Candle Window Queries

**Files:**
- Modify: `engine/data_engine.py`
- Modify: `tests/test_data_engine_window_queries.py`

- [ ] **Step 1: Add a failing range-query test**

```python
def test_get_candles_window_reads_only_requested_time_range(self):
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "sample.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute(
            "CREATE TABLE ticks AS SELECT * FROM (VALUES "
            "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 1.0)"
            ") AS t(timestamp, price, volume)"
        )
        con.execute(
            "CREATE TABLE candles_1m AS SELECT * FROM (VALUES "
            "(TIMESTAMP '2026-04-01 09:00:00', 100.0, 101.0, 99.0, 100.5, 1.0),"
            "(TIMESTAMP '2026-04-01 09:01:00', 100.5, 102.0, 100.0, 101.5, 2.0),"
            "(TIMESTAMP '2026-04-01 09:02:00', 101.5, 103.0, 101.0, 102.5, 3.0)"
            ") AS t(timestamp, open, high, low, close, volume)"
        )
        con.close()

        engine = DataEngine(parquet_file=str(db_path))
        df = engine.get_candles_window(
            "1min",
            pd.Timestamp("2026-04-01 09:01:00"),
            pd.Timestamp("2026-04-01 09:02:00"),
        )

        self.assertEqual(list(df["close"]), [101.5, 102.5])
```

- [ ] **Step 2: Implement `get_candles_window()`**

```python
def get_candles_window(self, timeframe, start_time, end_time):
    if not self._duckdb_path:
        df_full = self.get_candles(timeframe)
        if df_full is None or df_full.empty:
            return None
        start_ts = pd.Timestamp(start_time)
        end_ts = pd.Timestamp(end_time)
        if df_full.index.tz is not None and start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(df_full.index.tz)
            end_ts = end_ts.tz_localize(df_full.index.tz)
        if df_full.index.tz is None and start_ts.tzinfo is not None:
            start_ts = start_ts.tz_localize(None)
            end_ts = end_ts.tz_localize(None)
        return df_full.loc[start_ts:end_ts]

    table = self._duckdb_table_for_timeframe(timeframe)
    if table not in self._duckdb_candles_tables:
        return None

    import duckdb

    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_localize(None)

    con = duckdb.connect(self._duckdb_path, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT *
            FROM {table}
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            """,
            [start_ts.to_pydatetime(), end_ts.to_pydatetime()],
        ).df()
    finally:
        con.close()

    return normalize_candle_dataframe(
        df,
        f"{self._duckdb_path}::{table}",
        allow_gap_rows=self._table_allows_gap_rows(table),
    )
```

- [ ] **Step 3: Run tests and commit**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_data_engine_window_queries.py
```

Expected: PASS.

Commit:

```powershell
git add engine/data_engine.py tests/test_data_engine_window_queries.py
git commit -m "feat: 支持按时间窗口查询K线"
```

## Task 5: Route Normal Browsing Through Window Queries

**Files:**
- Modify: `ui/main_window.py`
- Modify: `ui/chart_widget.py`
- Test: `tests/test_main_window_time_navigation.py`, `tests/test_drawing_chart_widget.py`

- [ ] **Step 1: Add chart viewport time helpers**

In `ChartWidget`, add methods that expose visible time range based on current X range and loaded window index:

```python
def get_visible_time_range(self):
    if self.current_df is None or self.current_df.empty:
        return None
    min_x, max_x = self.ax.vb.viewRange()[0]
    start_idx = max(0, min(len(self.current_df) - 1, int(min_x)))
    end_idx = max(0, min(len(self.current_df) - 1, int(max_x)))
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    return self.current_df.index[start_idx], self.current_df.index[end_idx]
```

- [ ] **Step 2: Add `update_chart_window()` to `ChartWidget`**

Add a method that sets `current_df` to a window DataFrame and calls existing `update_plot_items()`:

```python
def update_chart_window(self, df, auto_scale=False, highlight_idx=None):
    if df is None or df.empty:
        return
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    self.full_df = df
    self.current_df = df
    self.current_x = np.arange(len(df), dtype=np.float64)
    self.current_time_values = np.asarray(df.index.view("int64"), dtype=np.float64)
    self.time_axis.set_datetime_index(df.index)
    self._last_slice_start = -1
    self._last_slice_end = -1
    self.update_plot_items(df, offset_x=0)
    if auto_scale:
        idx = highlight_idx if highlight_idx is not None else len(df) - 1
        x_start = max(0, idx - 150)
        x_end = min(len(df) - 1, idx + 20)
        visible_slice = df.iloc[int(x_start):int(x_end) + 1]
        if not visible_slice.empty:
            y_min = visible_slice["low"].min()
            y_max = visible_slice["high"].max()
            y_pad = (y_max - y_min) * 0.1
            self.ax.setYRange(y_min - y_pad, y_max + y_pad, padding=0)
        self.ax.setXRange(x_start, x_end, padding=0)
```

- [ ] **Step 3: Use window query for DuckDB normal browsing**

In `MainWindow.refresh_single_chart()`, when not replaying and `self.engine._duckdb_path` is set:

```python
period = chart.current_period
target_time = self._normalize_time(self.current_time)
view_count = self._get_view_count_for_period(period)
window_start = target_time - pd.Timedelta(minutes=max(view_count, 300))
window_end = target_time + pd.Timedelta(minutes=max(view_count // 4, 100))
df = self.engine.get_candles_window(period, window_start, window_end)
if df is not None and not df.empty:
    search_time = target_time
    if df.index.tz is None and search_time.tzinfo is not None:
        search_time = search_time.replace(tzinfo=None)
    target_idx = df.index.searchsorted(search_time)
    chart.update_chart_window(df, auto_scale=auto_scale, highlight_idx=target_idx)
    return
```

Keep the existing `get_candles()` path for Parquet and for DuckDB fallback.

- [ ] **Step 4: Run focused tests and commit**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_main_window_time_navigation.py tests/test_drawing_chart_widget.py tests/test_data_engine_window_queries.py
```

Expected: PASS.

Commit:

```powershell
git add ui/main_window.py ui/chart_widget.py
git commit -m "refactor: 浏览模式改用窗口K线数据"
```

## Task 6: Add Buffer-Aware Refresh and Prefetch Hooks

**Files:**
- Modify: `ui/chart_widget.py`
- Modify: `ui/main_window.py`
- Modify: `tests/test_chart_windowing.py`

- [ ] **Step 1: Track loaded window bounds in `ChartWidget`**

Add fields:

```python
self.loaded_start_time = None
self.loaded_end_time = None
self.window_generation = 0
```

Set them in `update_chart_window()`:

```python
self.loaded_start_time = df.index[0]
self.loaded_end_time = df.index[-1]
self.window_generation += 1
```

- [ ] **Step 2: Emit a signal when visible range needs reload**

Add signal:

```python
sig_window_reload_requested = pyqtSignal(object, object)
```

In `on_range_changed()`, if a chart is in window mode and the visible time range is outside loaded bounds, emit the signal instead of forcing a full redraw.

- [ ] **Step 3: Handle reload request in `MainWindow`**

Connect each chart signal to a method:

```python
chart.sig_window_reload_requested.connect(partial(self.reload_chart_window, chart))
```

Implement `reload_chart_window(chart, view_start, view_end)` using `build_query_window()` and `engine.get_candles_window()`.

- [ ] **Step 4: Run focused tests and commit**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_chart_windowing.py tests/test_main_window_time_navigation.py tests/test_drawing_chart_widget.py
```

Expected: PASS.

Commit:

```powershell
git add ui/chart_widget.py ui/main_window.py tests/test_chart_windowing.py
git commit -m "feat: 支持图表窗口边缘重载"
```

## Task 7: Integrate LOD for Wide Ranges

**Files:**
- Modify: `ui/main_window.py`
- Modify: `ui/chart_widget.py`
- Modify: `tests/test_chart_lod.py`

- [ ] **Step 1: Resolve actual display period before querying**

In window reload paths, call:

```python
actual_period = choose_lod_period(
    requested_period=chart.current_period,
    view_start=view_start,
    view_end=view_end,
    pixel_width=max(chart.width(), 800),
)
```

Use `actual_period` for `get_candles_window()`.

- [ ] **Step 2: Make actual period visible to the chart**

Add `chart.active_display_period = actual_period` so later UI can show whether the chart is rendering a coarser LOD period.

- [ ] **Step 3: Run focused tests and commit**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q tests/test_chart_lod.py tests/test_main_window_time_navigation.py
```

Expected: PASS.

Commit:

```powershell
git add ui/main_window.py ui/chart_widget.py ui/chart_lod.py tests/test_chart_lod.py
git commit -m "feat: 浏览大范围时自动选择LOD周期"
```

## Task 8: Full Verification and Manual Performance Check

**Files:**
- No planned code changes unless verification reveals defects.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Manual large-data smoke test**

Run the app, load:

```text
data/2026.3.15xauusd-tick-no session.duckdb
```

Verify:

- Initial load does not pause to read all `ticks`.
- Jumping to early, middle, and late dates reloads the chart window.
- Dragging inside the loaded buffer is smooth.
- Dragging past the buffer reloads once, not continuously.
- Zooming out switches to a coarser display period.

- [ ] **Step 3: Check Git state**

Run:

```powershell
git status --short --branch
```

Expected: clean working tree after commits.

## Notes

- Keep commit messages in Chinese.
- Do not combine this data-windowing work with a C++ / OpenGL rewrite.
- Do not remove existing full-DataFrame code paths until DuckDB window mode is stable.
- Keep Replay mode separate. Replay can continue to use `ReplayEngine` until normal browsing performance is fixed.
