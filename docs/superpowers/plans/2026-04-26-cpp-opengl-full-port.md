# C++/Qt/OpenGL Full Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Python TradeReview application into the native C++/Qt/OpenGL application while preserving the Python version's product semantics and replacing full-data UI loading with viewport-windowed data flow.

**Architecture:** Treat Python as the reference behavior, not as code to translate line-by-line. Build the native app around DuckDB window queries, per-chart scene models, OpenGL batched rendering, independent indicator panels, timestamp-based sync, canonical drawing storage, and chunked replay.

**Tech Stack:** C++20, Qt 6 Widgets, QOpenGLWidget/OpenGL, CMake, DuckDB C API boundary, CTest-native test harness, existing Python tests as behavior references.

---

## Execution Rules

- Work from `V:\gold\TradeReview\.worktrees\cpp-opengl-native-m0m1` unless explicitly redirected.
- Keep commits small and use Chinese commit messages.
- Do not proactively compile or run the C++ native app. The user will compile manually. Static verification is allowed.
- Ignore untracked build directories such as `native-build-*`, `native/build/`, and `native/.qtcreator/`.
- Complete tasks in order unless a blocker requires splitting a task.
- After every completed task:
  - update this plan's checkbox state;
  - add a short entry under `Progress Log`;
  - commit the code and plan update together;
  - summarize changed files, verification, risks, and next task to the user.
- Subagents may be used for implementation and review. Prefer medium/standard models for bounded implementation tasks, and reserve higher reasoning for architecture, integration, or difficult review.

## Current Baseline

- Branch: `cpp-opengl-native-m0m1`
- Latest known commit before this plan: `d52ef65 增加C++原生占位工具栏`
- Native foundation already exists:
  - `native/core`: period and time range helpers.
  - `native/data`: `CandleWindow`, `DataSetInfo`, `IDataStore`.
  - `native/chart`: `ChartSceneModel`, `ChartIndexMapper`, LOD/window helpers, `ChartViewWidget`, placeholder workspace toolbar.
  - `native/app`: Qt main window and placeholder controls.
- Existing Python reference modules:
  - Data: `engine/data_engine.py`, `engine/data_validation.py`, `ui/services/data_loading.py`
  - Replay: `engine/replay_engine.py`, `ui/controllers/replay_controller.py`
  - Chart: `ui/chart_widget.py`, `ui/chart_lod.py`, `ui/chart_windowing.py`, `ui/chart_performance.py`
  - Sync: `ui/crosshair_sync.py`, `ui/time_navigation.py`
  - Drawing: `ui/drawings/*`
  - Session: `ui/session_state.py`

## Product Semantics To Preserve

- Opening a DuckDB source must read metadata first and must not materialize full `ticks`.
- Candle queries are time-window based and include a buffer around the visible range.
- LOD never selects a finer period than the user requested.
- Multi-chart sync uses canonical timestamp/price, not local dense x values.
- Drawings persist canonical `{timestamp, price}` points.
- Fib settings are snapshotted into each Fib drawing.
- EMA and Bollinger are price-pane overlays.
- MACD/RSI are independent indicator panes with shared X axis and independent Y scales.
- Replay must be chunked and must not require full tick history in memory.
- The final candle should have right-side visual padding.

---

## Task List

### Task 1: Data Contract Hardening

**Goal:** Make native data contracts expressive enough for real DuckDB metadata, candle windows, indicator columns, and user-visible load failures without adding a DuckDB dependency yet.

**Files:**
- Modify: `native/data/include/tradereview/data/DataSetInfo.h`
- Modify: `native/data/include/tradereview/data/CandleWindow.h`
- Modify: `native/data/src/CandleWindow.cpp`
- Modify: `native/data/include/tradereview/data/IDataStore.h`
- Create: `native/data/include/tradereview/data/DataError.h`
- Create: `native/data/include/tradereview/data/IndicatorColumns.h`
- Create: `native/tests/data/test_data_contracts.cpp`
- Modify: `native/tests/CMakeLists.txt`

- [x] Add `DataError` with a stable `DataErrorCode` enum and message/path/table fields.
- [x] Add canonical indicator names for `EMA20`, `EMA30`, `EMA40`, `EMA50`, `EMA60`, `EMA100`, `EMA240`, `BB_Upper`, `BB_Lower`, `MACD`, `MACD_Signal`, `MACD_Hist`, `RSI`.
- [x] Extend `DataSetInfo` with dataset path, tick count, tick time range, available periods, available indicators, schema version, indicator version, and metadata-only flag.
- [x] Extend `CandleWindow` with helpers for required OHLCV consistency, indicator consistency, and loaded/visible range presence.
- [x] Extend `CandleWindowRequest` with requested indicator names and right-padding/warmup fields needed by chart and replay tasks.
- [x] Add native tests for metadata-only source state, indicator consistency, inconsistent columns, and request defaults.
- [x] Static verification: `rg -n "DataError|IndicatorColumns|metadata_only|requested_indicators|has_consistent" native`.
- [x] Commit with message: `强化C++数据契约`.

### Task 2: DuckDB Repository Boundary

**Goal:** Add the real repository boundary for opening a DuckDB file, inspecting metadata/schema, and translating period names to candle tables.

**Files:**
- Create: `native/data/include/tradereview/data/DuckDbRepository.h`
- Create: `native/data/src/DuckDbRepository.cpp`
- Create: `native/data/include/tradereview/data/DuckDbSchema.h`
- Create: `native/data/src/DuckDbSchema.cpp`
- Modify: `native/data/CMakeLists.txt`
- Create: `native/tests/data/test_duckdb_schema.cpp`
- Modify: `native/tests/CMakeLists.txt`

- [ ] Add schema inspection types independent of the concrete DuckDB C API handle.
- [ ] Implement table name mapping for existing period semantics including `1min`/`1m` and `1M`/month separation.
- [ ] Implement pure schema validation helpers for `ticks`, candle OHLCV columns, optional indicator columns, and gap-row allowance.
- [ ] Add a `DuckDbRepository` class with methods matching `IDataStore`, with C API calls isolated in this class.
- [ ] Gate DuckDB linkage behind a CMake option if local headers/libs are not always present.
- [ ] Static verification: `rg -n "DuckDbRepository|DuckDbSchema|validate_.*candle|candles_1mo|TRADEREVIEW_NATIVE_WITH_DUCKDB" native`.
- [ ] Commit with message: `增加C++ DuckDB仓储边界`.

### Task 3: Single Chart Data Loading Path

**Goal:** Wire `Load Data` to a real data-loading controller and show one chart's candle window from the selected dataset.

**Files:**
- Create: `native/app/include/tradereview/app/DataLoadController.h`
- Create: `native/app/src/DataLoadController.cpp`
- Modify: `native/app/include/tradereview/app/MainControlsBar.h`
- Modify: `native/app/src/MainControlsBar.cpp`
- Modify: `native/app/src/MainWindow.cpp`
- Modify: `native/chart/include/tradereview/chart/ChartWorkspaceWidget.h`
- Modify: `native/chart/src/ChartWorkspaceWidget.cpp`

- [ ] Replace the `Load Data` placeholder with a signal/callback path that opens a file dialog.
- [ ] Add a controller that opens the repository, reads metadata, selects an initial center time, and requests the first candle window.
- [ ] Pass the returned `CandleWindow` into the first `ChartViewWidget`.
- [ ] Keep the UI responsive by keeping the synchronous path small; full async arrives in Task 6.
- [ ] Static verification: `rg -n "DataLoadController|open_readonly|query_candles|Load Data is not wired" native`.
- [ ] Commit with message: `接入C++单图数据加载路径`.

### Task 4: Real OpenGL Candle Layer

**Goal:** Replace the blank OpenGL clear pass with batched candlestick rendering for the current `CandleWindow`.

**Files:**
- Create: `native/chart/rendering/include/tradereview/chart/rendering/GLResources.h`
- Create: `native/chart/rendering/src/GLResources.cpp`
- Create: `native/chart/rendering/include/tradereview/chart/rendering/CandleLayer.h`
- Create: `native/chart/rendering/src/CandleLayer.cpp`
- Modify: `native/chart/rendering/include/tradereview/chart/rendering/GLChartRenderer.h`
- Modify: `native/chart/rendering/src/GLChartRenderer.cpp`
- Modify: `native/chart/CMakeLists.txt`

- [ ] Add shader/program and buffer lifetime wrappers owned by the GL thread.
- [ ] Build candle body and wick geometry from the visible `CandleWindow`.
- [ ] Render up/down candles with distinct colors and stable background/grid color.
- [ ] Keep GPU buffers scoped to the current window/generation.
- [ ] Static verification: `rg -n "CandleLayer|glBufferData|glDraw|generation|wick|body" native/chart`.
- [ ] Commit with message: `实现C++ OpenGL K线图层`.

### Task 5: Chart View Interaction

**Goal:** Add pan, zoom, visible range tracking, right-edge padding, and window reload requests without re-uploading full history during small viewport changes.

**Files:**
- Create: `native/chart/include/tradereview/chart/ChartInteractionController.h`
- Create: `native/chart/src/ChartInteractionController.cpp`
- Modify: `native/chart/include/tradereview/chart/ChartViewWidget.h`
- Modify: `native/chart/src/ChartViewWidget.cpp`
- Modify: `native/chart/include/tradereview/chart/ChartSceneModel.h`
- Modify: `native/chart/src/ChartSceneModel.cpp`
- Create: `native/tests/chart/test_chart_interaction.cpp`
- Modify: `native/tests/CMakeLists.txt`

- [ ] Track visible dense-x range separately from loaded window data.
- [ ] Add mouse drag panning and wheel zoom around cursor.
- [ ] Add visual right padding after the last candle.
- [ ] Emit/request reload when the visible time range leaves or nears the loaded range.
- [ ] Preserve Python behavior from `ui/chart_windowing.py` and right-padding tests.
- [ ] Static verification: `rg -n "ChartInteractionController|right.*padding|wheelEvent|mouseMoveEvent|reload" native/chart`.
- [ ] Commit with message: `增加C++图表视口交互`.

### Task 6: Async Window Scheduler and Cache

**Goal:** Move DuckDB window queries off the UI thread and discard stale generation results.

**Files:**
- Create: `native/data/include/tradereview/data/DataScheduler.h`
- Create: `native/data/src/DataScheduler.cpp`
- Create: `native/data/include/tradereview/data/WindowCache.h`
- Create: `native/data/src/WindowCache.cpp`
- Modify: `native/data/CMakeLists.txt`
- Modify: `native/app/src/DataLoadController.cpp`
- Create: `native/tests/data/test_window_cache.cpp`
- Modify: `native/tests/CMakeLists.txt`

- [ ] Add request coalescing by chart id, generation, period, and visible range.
- [ ] Add a small LRU window cache keyed by dataset, period, range, and indicator version.
- [ ] Return results to the UI thread through Qt queued callbacks.
- [ ] Drop stale results whose generation no longer matches the chart.
- [ ] Static verification: `rg -n "DataScheduler|WindowCache|generation|queued|stale|LRU" native`.
- [ ] Commit with message: `增加C++异步窗口加载`.

### Task 7: Indicator Columns and Panels

**Goal:** Render EMA/BB overlays on the price pane and MACD/RSI as independent panes.

**Files:**
- Create: `native/chart/include/tradereview/chart/PaneLayout.h`
- Create: `native/chart/src/PaneLayout.cpp`
- Create: `native/chart/rendering/include/tradereview/chart/rendering/IndicatorLayer.h`
- Create: `native/chart/rendering/src/IndicatorLayer.cpp`
- Create: `native/chart/rendering/include/tradereview/chart/rendering/HistogramLayer.h`
- Create: `native/chart/rendering/src/HistogramLayer.cpp`
- Modify: `native/chart/src/ChartToolbarWidget.cpp`
- Modify: `native/chart/rendering/src/GLChartRenderer.cpp`

- [ ] Add pane rectangles for price, MACD, and RSI.
- [ ] Draw EMA and BB using price pane scale.
- [ ] Draw MACD lines/histogram and RSI line in separate panes with independent Y scaling.
- [ ] Connect toolbar indicator toggles to scene state rather than placeholders.
- [ ] Static verification: `rg -n "PaneLayout|IndicatorLayer|HistogramLayer|MACD|RSI|EMA20|BB" native/chart`.
- [ ] Commit with message: `实现C++指标图层和窗格`.

### Task 8: Multi-Chart Workspace and Layouts

**Goal:** Restore the Python workspace behavior: chart count, tabs, vertical, dual vertical, grid 2x2, period switching, and layout detach placeholders progressing toward real detach.

**Files:**
- Create: `native/chart/include/tradereview/chart/ChartPanelWidget.h`
- Create: `native/chart/src/ChartPanelWidget.cpp`
- Modify: `native/chart/include/tradereview/chart/ChartWorkspaceWidget.h`
- Modify: `native/chart/src/ChartWorkspaceWidget.cpp`
- Modify: `native/app/src/MainControlsBar.cpp`
- Modify: `native/app/src/MainWindow.cpp`

- [ ] Replace one chart with managed chart panels.
- [ ] Support 1-4 enabled charts.
- [ ] Support Tabs, Vertical, Dual Vertical, and Grid 2x2 layouts.
- [ ] Preserve each chart's selected period.
- [ ] Use shared dataset metadata and independent chart generations.
- [ ] Static verification: `rg -n "ChartPanelWidget|Grid 2x2|Dual Vertical|setChartCount|setLayoutMode|period" native`.
- [ ] Commit with message: `实现C++多图工作区布局`.

### Task 9: Crosshair and Time Sync

**Goal:** Implement timestamp/price crosshair sync and chart-center sync across attached and detached chart panels.

**Files:**
- Create: `native/sync/include/tradereview/sync/CrosshairSyncController.h`
- Create: `native/sync/src/CrosshairSyncController.cpp`
- Modify: `native/sync/CMakeLists.txt`
- Modify: `native/chart/include/tradereview/chart/ChartViewWidget.h`
- Modify: `native/chart/src/ChartViewWidget.cpp`
- Modify: `native/chart/src/ChartWorkspaceWidget.cpp`
- Create: `native/tests/sync/test_crosshair_sync.cpp`
- Modify: `native/tests/CMakeLists.txt`

- [ ] Emit crosshair updates as canonical timestamp/price.
- [ ] Convert incoming timestamp to each chart's local dense x.
- [ ] Skip disabled source charts and avoid feedback loops.
- [ ] Add center-time sync and optional Y-center sync.
- [ ] Static verification: `rg -n "CrosshairSyncController|timestamp_ns|price|sync_crosshair|center" native`.
- [ ] Commit with message: `实现C++多图十字线同步`.

### Task 10: Drawing Store and Fib Math

**Goal:** Port drawing object semantics before UI interaction: canonical points, normalization, Fib retracement/extension math, and settings snapshot.

**Files:**
- Create: `native/drawing/include/tradereview/drawing/DrawingSpec.h`
- Create: `native/drawing/src/DrawingSpec.cpp`
- Create: `native/drawing/include/tradereview/drawing/FibMath.h`
- Create: `native/drawing/src/FibMath.cpp`
- Create: `native/drawing/include/tradereview/drawing/FibSettings.h`
- Create: `native/drawing/src/FibSettings.cpp`
- Modify: `native/drawing/CMakeLists.txt`
- Create: `native/tests/drawing/test_drawing_spec.cpp`
- Create: `native/tests/drawing/test_fib_math.cpp`
- Modify: `native/tests/CMakeLists.txt`

- [ ] Add drawing point/spec types using timestamp/price canonical coordinates.
- [ ] Normalize line, horizontal, vertical, Fib, and Fib extension specs.
- [ ] Implement Fib retracement and extension level generation matching Python tests.
- [ ] Snapshot Fib settings into created drawing specs.
- [ ] Static verification: `rg -n "DrawingSpec|FibMath|FibSettings|retracement|extension|timestamp_ns" native`.
- [ ] Commit with message: `移植C++绘图规格和Fib计算`.

### Task 11: Drawing Interaction and Rendering

**Goal:** Add drawing toolbar behavior, drawing creation, preview, clear/delete, and OpenGL/overlay rendering.

**Files:**
- Create: `native/drawing/include/tradereview/drawing/DrawingSession.h`
- Create: `native/drawing/src/DrawingSession.cpp`
- Create: `native/chart/rendering/include/tradereview/chart/rendering/DrawingLayer.h`
- Create: `native/chart/rendering/src/DrawingLayer.cpp`
- Modify: `native/chart/src/ChartToolbarWidget.cpp`
- Modify: `native/chart/src/ChartViewWidget.cpp`
- Modify: `native/chart/rendering/src/GLChartRenderer.cpp`

- [ ] Activate tools from toolbar: Sel, H, V, Line, Fib, Fib Ext.
- [ ] Convert mouse clicks to canonical drawing points.
- [ ] Render preview while drawing.
- [ ] Store completed drawings and replay them across period changes through timestamp mapping.
- [ ] Implement Clear and selected drawing delete.
- [ ] Static verification: `rg -n "DrawingSession|DrawingLayer|Fib Ext|Clear|preview|delete" native`.
- [ ] Commit with message: `实现C++图表绘图交互`.

### Task 12: Chunked Replay

**Goal:** Port replay without full tick materialization by querying tick chunks and incrementally building active candles.

**Files:**
- Create: `native/replay/include/tradereview/replay/ReplaySession.h`
- Create: `native/replay/src/ReplaySession.cpp`
- Create: `native/replay/include/tradereview/replay/BarBuilder.h`
- Create: `native/replay/src/BarBuilder.cpp`
- Modify: `native/replay/CMakeLists.txt`
- Modify: `native/app/src/MainControlsBar.cpp`
- Modify: `native/app/src/DataLoadController.cpp`
- Create: `native/tests/replay/test_bar_builder.cpp`
- Create: `native/tests/replay/test_replay_session.cpp`
- Modify: `native/tests/CMakeLists.txt`

- [ ] Add replay enabled/play/pause/speed/step state.
- [ ] Query tick chunks with a per-frame tick cap.
- [ ] Maintain per-period bar builders.
- [ ] Update the active candle incrementally.
- [ ] Stop at dataset end and update controls.
- [ ] Static verification: `rg -n "ReplaySession|BarBuilder|max_ticks|advance|Replay Mode|Speed" native`.
- [ ] Commit with message: `实现C++分块回放基础`.

### Task 13: Time Navigation and Session State

**Goal:** Restore date jump, step back/forward, reset view, save/restore view, and session state.

**Files:**
- Create: `native/app/include/tradereview/app/SessionState.h`
- Create: `native/app/src/SessionState.cpp`
- Create: `native/app/include/tradereview/app/TimeNavigation.h`
- Create: `native/app/src/TimeNavigation.cpp`
- Modify: `native/app/src/MainControlsBar.cpp`
- Modify: `native/app/src/MainWindow.cpp`
- Create: `native/tests/app/test_time_navigation.cpp`
- Create: `native/tests/app/test_session_state.cpp`
- Modify: `native/tests/CMakeLists.txt`

- [ ] Normalize jump timestamps to minute precision.
- [ ] Clamp jumps to loaded dataset range.
- [ ] Resolve target chart row from right-side bar behavior.
- [ ] Save and restore dataset path, center time, chart count, layout, and periods.
- [ ] Static verification: `rg -n "SessionState|TimeNavigation|clamp|Save View|Reset View|QSettings" native`.
- [ ] Commit with message: `移植C++时间导航和会话状态`.

### Task 14: Error Handling and Product Polish

**Goal:** Replace placeholder status messages with explicit user-visible errors, loading states, and robust stale-result handling.

**Files:**
- Modify: `native/app/src/MainWindow.cpp`
- Modify: `native/app/src/DataLoadController.cpp`
- Modify: `native/chart/src/ChartViewWidget.cpp`
- Modify: `native/data/src/DuckDbRepository.cpp`
- Create: `native/app/include/tradereview/app/ErrorPresenter.h`
- Create: `native/app/src/ErrorPresenter.cpp`

- [ ] Present file/schema/query errors with clear messages.
- [ ] Show lightweight chart loading state during async window requests.
- [ ] Silently discard stale generation results.
- [ ] Keep UI usable when one chart fails to load.
- [ ] Static verification: `rg -n "ErrorPresenter|loading|stale|DataError|QMessageBox" native`.
- [ ] Commit with message: `完善C++错误提示和加载状态`.

### Task 15: Verification Checklist and Manual Build Notes

**Goal:** Prepare a final manual verification checklist for the user without automatically compiling/running native code.

**Files:**
- Create: `docs/native-manual-verification.md`
- Modify: this plan file

- [ ] Document manual CMake configure/build/run steps for Qt Creator and PowerShell.
- [ ] Document small-data smoke scenarios: open dataset, single chart, four charts, period switch, pan/zoom, indicators, drawing, replay, session restore.
- [ ] Document expected performance behavior: small pan no reload, edge pan prefetch/reload, large jump reload, LOD on wide ranges.
- [ ] Static verification: `rg -n "manual|smoke|pan|LOD|replay|session" docs/native-manual-verification.md`.
- [ ] Commit with message: `补充C++手动验收清单`.

---

## Progress Log

### 2026-04-26 Task Planning

- Created this living implementation plan.
- Current execution constraint: do not proactively compile or run the C++ native app.
- Next task: Task 1, data contract hardening.

### 2026-04-26 Task 1 Completed

- Hardened native data contracts with `DataError`, canonical indicator column names, metadata-only dataset state, candle window consistency helpers, and extended candle window request fields.
- Added native data contract tests in `native/tests/data/test_data_contracts.cpp`.
- User explicitly allowed a temporary MSVC build for this task. Verified with `C:\Build\TradeReview-native-task1-msvc`:
  - configure: `cmake -S native -B C:\Build\TradeReview-native-task1-msvc ...` exited 0;
  - build: `cmake --build C:\Build\TradeReview-native-task1-msvc --target tradereview_native_tests --config Debug` exited 0;
  - test: `ctest --test-dir C:\Build\TradeReview-native-task1-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- No native exe was launched.
- Next task: Task 2, DuckDB repository boundary.
