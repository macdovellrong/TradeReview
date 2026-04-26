# C++/Qt/OpenGL Full Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Python TradeReview application into the native C++/Qt/OpenGL application while preserving the Python version's product semantics and replacing full-data UI loading with viewport-windowed data flow.

**Architecture:** Treat Python as the reference behavior, not as code to translate line-by-line. Build the native app around DuckDB window queries, per-chart scene models, OpenGL batched rendering, independent indicator panels, timestamp-based sync, canonical drawing storage, and chunked replay.

**Tech Stack:** C++20, Qt 6 Widgets, QOpenGLWidget/OpenGL, CMake, DuckDB C API boundary, CTest-native test harness, existing Python tests as behavior references.

---

## Execution Rules

- Work from `V:\gold\TradeReview\.worktrees\cpp-opengl-native-m0m1` unless explicitly redirected.
- Keep commits small and use Chinese commit messages.
- For this full-port execution, the user explicitly permits Qt + MSVC configure/build/CTest validation in temporary directories under `C:\Build`, such as `C:\Build\TradeReview-native-taskN-msvc`.
- Do not launch the native exe unless the user explicitly asks for it.
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

- [x] Add schema inspection types independent of the concrete DuckDB C API handle.
- [x] Implement table name mapping for existing period semantics including `1min`/`1m` and `1M`/month separation.
- [x] Implement pure schema validation helpers for `ticks`, candle OHLCV columns, optional indicator columns, and gap-row allowance.
- [x] Add a `DuckDbRepository` class with methods matching `IDataStore`, with C API calls isolated in this class.
- [x] Gate DuckDB linkage behind a CMake option if local headers/libs are not always present.
- [x] Static verification: `rg -n "DuckDbRepository|DuckDbSchema|validate_.*candle|candles_1mo|TRADEREVIEW_NATIVE_WITH_DUCKDB" native`.
- [x] Commit with message: `增加C++ DuckDB仓储边界`.

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

- [x] Replace the `Load Data` placeholder with a signal/callback path that opens a file dialog.
- [x] Add a controller that opens the repository, reads metadata, selects an initial center time, and requests the first candle window.
- [x] Pass the returned `CandleWindow` into the first `ChartViewWidget`.
- [x] Keep the UI responsive by keeping the synchronous path small; full async arrives in Task 6.
- [x] Static verification: `rg -n "DataLoadController|open_readonly|query_candles|Load Data is not wired" native`.
- [x] Commit with message: `接入C++单图数据加载路径`.

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

- [x] Add shader/program and buffer lifetime wrappers owned by the GL thread.
- [x] Build candle body and wick geometry from the visible `CandleWindow`.
- [x] Render up/down candles with distinct colors and stable background/grid color.
- [x] Keep GPU buffers scoped to the current window/generation.
- [x] Static verification: `rg -n "CandleLayer|glBufferData|glDraw|generation|wick|body" native/chart`.
- [x] Commit with message: `实现C++ OpenGL K线图层`.

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

- [x] Track visible dense-x range separately from loaded window data.
- [x] Add mouse drag panning and wheel zoom around cursor.
- [x] Add visual right padding after the last candle.
- [x] Emit/request reload when the visible time range leaves or nears the loaded range.
- [x] Preserve Python behavior from `ui/chart_windowing.py` and right-padding tests.
- [x] Static verification: `rg -n "ChartInteractionController|right.*padding|wheelEvent|mouseMoveEvent|reload" native/chart`.
- [x] Commit with message: `增加C++图表视口交互`.

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

- [x] Add request coalescing by chart id, generation, period, and visible range.
- [x] Add a small LRU window cache keyed by dataset, period, range, and indicator version.
- [x] Return results to the UI thread through Qt queued callbacks.
- [x] Drop stale results whose generation no longer matches the chart.
- [x] Static verification: `rg -n "DataScheduler|WindowCache|generation|queued|stale|LRU" native`.
- [x] Commit with message: `增加C++异步窗口加载`.

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

- [x] Add pane rectangles for price, MACD, and RSI.
- [x] Draw EMA and BB using price pane scale.
- [x] Draw MACD lines/histogram and RSI line in separate panes with independent Y scaling.
- [x] Connect toolbar indicator toggles to scene state rather than placeholders.
- [x] Static verification: `rg -n "PaneLayout|IndicatorLayer|HistogramLayer|MACD|RSI|EMA20|BB" native/chart`.
- [x] Commit with message: `实现C++指标图层和窗格`.

### Task 8: Multi-Chart Workspace and Layouts

**Goal:** Restore the Python workspace behavior: chart count, tabs, vertical, dual vertical, grid 2x2, period switching, and layout detach placeholders progressing toward real detach.

**Files:**
- Create: `native/chart/include/tradereview/chart/ChartPanelWidget.h`
- Create: `native/chart/src/ChartPanelWidget.cpp`
- Modify: `native/chart/include/tradereview/chart/ChartWorkspaceWidget.h`
- Modify: `native/chart/src/ChartWorkspaceWidget.cpp`
- Modify: `native/app/src/MainControlsBar.cpp`
- Modify: `native/app/src/MainWindow.cpp`

- [x] Replace one chart with managed chart panels.
- [x] Support 1-4 enabled charts.
- [x] Support Tabs, Vertical, Dual Vertical, and Grid 2x2 layouts.
- [x] Preserve each chart's selected period.
- [x] Use shared dataset metadata and independent chart generations.
- [x] Static verification: `rg -n "ChartPanelWidget|Grid 2x2|Dual Vertical|setChartCount|setLayoutMode|period" native`.
- [x] Commit with message: `实现C++多图工作区布局`.

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

- [x] Emit crosshair updates as canonical timestamp/price.
- [x] Convert incoming timestamp to each chart's local dense x.
- [x] Skip disabled source charts and avoid feedback loops.
- [x] Add center-time sync and optional Y-center sync.
- [x] Static verification: `rg -n "CrosshairSyncController|timestamp_ns|price|sync_crosshair|center" native`.
- [x] Commit with message: `实现C++多图十字线同步`.

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

### 2026-04-26 Task 2 Completed

- Added `DuckDbSchema` pure schema helpers and tests for period-to-table mapping, timestamp/datetime/time aliases, ticks columns, candle OHLCV columns, requested indicator columns, and missing-column reporting.
- Added `DuckDbRepository` behind `TRADEREVIEW_NATIVE_WITH_DUCKDB`, defaulting OFF so builds do not require DuckDB headers/libs yet.
- Main-thread review caught and fixed a schema mismatch: external DuckDB tables validate `timestamp`/`datetime`/`time` aliases, not internal `timestamp_ns`.
- Verified with `C:\Build\TradeReview-native-task2-msvc`:
  - configure exited 0;
  - `cmake --build C:\Build\TradeReview-native-task2-msvc --target tradereview_native_tests --config Debug` exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task2-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- No native exe was launched.
- Next task: Task 3, single chart data loading path.

### 2026-04-26 DuckDB Local Dependency Downloaded

- Downloaded official DuckDB `v1.5.2` Windows x64 C API package from `https://github.com/duckdb/duckdb/releases/download/v1.5.2/libduckdb-windows-amd64.zip`.
- Local zip: `C:\Build\deps\libduckdb-windows-amd64-v1.5.2.zip`.
- Extracted directory: `C:\Build\deps\duckdb-v1.5.2`.
- Extracted files: `duckdb.h`, `duckdb.hpp`, `duckdb.lib`, `duckdb.dll`.
- SHA256:
  - zip: `C60BD7DEB0EF6C2D5C12A9765B93ED930C34B984D4DB79104A8C2955BC57017D`
  - `duckdb.lib`: `66274DB0EFECE69AC05A0C54F626C04A17B12A99E5302CDF34E0DC6F696773AE`
  - `duckdb.dll`: `8BDBF7CFE619482E64F30E99C628EAEE28936C5B10FE8459BAEA9DF8B78095B6`
- Suggested CMake variables for DuckDB-enabled temporary builds:
  - `-DTRADEREVIEW_NATIVE_WITH_DUCKDB=ON`
  - `-DDUCKDB_INCLUDE_DIR=C:\Build\deps\duckdb-v1.5.2`
  - `-DDUCKDB_LIBRARY=C:\Build\deps\duckdb-v1.5.2\duckdb.lib`

### 2026-04-26 Task 3 Completed

- Wired the native `Load Data` action to a file dialog and a new `DataLoadController`.
- Implemented the first real single-chart DuckDB loading path: open read-only metadata, select an initial 6-hour window around the dataset midpoint, query `1min` candles, and apply the returned `CandleWindow` to the first chart.
- Completed the first real DuckDB C API repository path for metadata and candle windows, including timestamp/datetime/time aliases, canonical indicator column discovery, empty-window handling, and post-build `duckdb.dll` copy for the app and tests.
- Main-thread review fixed an integration bug where the loaded window generation did not match the chart scene generation.
- Verified with `C:\Build\TradeReview-native-task3-msvc` using DuckDB ON:
  - configure exited 0;
  - full build exited 0 and produced both `tradereview_native_tests.exe` and `tradereview_native.exe`;
  - `ctest --test-dir C:\Build\TradeReview-native-task3-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- Verified with `C:\Build\TradeReview-native-task3-off-msvc` using DuckDB OFF:
  - configure exited 0;
  - full build exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task3-off-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- No native exe was launched.
- Next task: Task 4, real OpenGL candle layer.

### 2026-04-26 Task 4 Completed

- Added `GLResources` wrappers for OpenGL buffers, vertex arrays, and shader programs, plus `CandleLayer` batched geometry/upload/rendering for candle bodies, wicks, and a stable grid.
- Integrated `GLChartRenderer` with `CandleLayer` and Qt 6 `QOpenGLVersionFunctionsFactory`, and made `ChartViewWidget` release GL resources on context destruction without launching the native app.
- Added scene revision tracking so same-generation data replacement invalidates the uploaded GPU buffers, and kept doji candles visible with a minimum body height.
- Added native tests for candle geometry body/wick/grid output, inconsistent windows, visible doji bodies, and same-generation scene revisions.
- Code review found context recreation, same-generation upload caching, doji visibility, and direct include gaps; all were fixed before commit.
- Verified with `C:\Build\TradeReview-native-task4-msvc` using DuckDB ON:
  - configure exited 0;
  - full build exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task4-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- Verified with `C:\Build\TradeReview-native-task4-off-msvc` using DuckDB OFF:
  - configure exited 0;
  - full build exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task4-off-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- Static verification: `rg -n "CandleLayer|glBufferData|glDraw|generation|wick|body" native/chart`.
- `git diff --check HEAD -- native/chart native/tests docs/superpowers/plans/2026-04-26-cpp-opengl-full-port.md` exited 0 with only LF-to-CRLF notices.
- No native exe was launched.
- Next task: Task 5, chart view interaction.

### 2026-04-26 Task 5 Completed

- Added `ChartInteractionController` for pure, tested viewport state: visible dense-x range, mouse-drag pan math, cursor-centered wheel zoom, right-edge visual padding, visible time range mapping, and reload decisions.
- Wired `ChartViewWidget` mouse drag and wheel events to the controller, and added a reload request callback surfaced through the native main window status path as a placeholder until Task 6 adds async reload execution.
- Updated `ChartSceneModel` to keep visible dense range separate from loaded `CandleWindow` data, with scene revisions invalidating OpenGL uploads when the viewport changes.
- Updated `CandleLayer` to build geometry from the current visible dense range so the last candle can sit away from the right boundary and y scaling uses visible rows rather than off-screen loaded rows.
- Added tests for pan, zoom, wheel delta magnitude/pixel delta, buffered-window visible range preservation, right padding, reload decisions, visible range revisions, right-padded geometry, and visible-row y scaling.
- Code review found that buffered window application could snap to the full loaded range, y scaling could include off-screen rows, and high-resolution wheel events could be ignored; all were fixed before commit.
- Verified with `C:\Build\TradeReview-native-task5-msvc` using DuckDB ON:
  - configure exited 0;
  - full build exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task5-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- Verified with `C:\Build\TradeReview-native-task5-off-msvc` using DuckDB OFF:
  - configure exited 0;
  - full build exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task5-off-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- Static verification: `rg -n "ChartInteractionController|right.*padding|wheelEvent|mouseMoveEvent|reload" native/chart`.
- `git diff --check HEAD -- native/app native/chart native/tests docs/superpowers/plans/2026-04-26-cpp-opengl-full-port.md` exited 0 with only LF-to-CRLF notices.
- No native exe was launched.
- Next task: Task 6, async window scheduler and cache.

### 2026-04-26 Task 6 Completed

- Added `DataScheduler` and `WindowCache` for off-UI-thread candle window queries, request coalescing, a small LRU cache, Qt queued callbacks, stale generation drops, and a joined single-worker queue.
- Wired `DataLoadController` and `MainWindow` so initial loads and viewport reloads reuse one scheduler-backed controller instead of querying DuckDB directly from the UI path; dataset open now shares the scheduler's store lock with window queries.
- Added native tests for LRU eviction, in-flight coalescing, cache hits across generations, queued receiver callbacks, destroyed receiver drops, serialized dataset open, queued stale-result drops, queued stale query skipping, and stale generation suppression.
- Verified with `C:\Build\TradeReview-native-task6-off-msvc` using DuckDB OFF:
  - configure exited 0;
  - `cmake --build C:\Build\TradeReview-native-task6-off-msvc --target tradereview_native_tests --config Debug` exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task6-off-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`;
  - `cmake --build C:\Build\TradeReview-native-task6-off-msvc --target tradereview_native --config Debug` exited 0.
- Verified with `C:\Build\TradeReview-native-task6-msvc` using DuckDB ON:
  - configure exited 0;
  - full build exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task6-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- Static verification: `rg -n "DataScheduler|WindowCache|generation|queued|stale|LRU" native`.
- `git diff --check` exited 0 with only LF-to-CRLF notices.
- No native exe was launched.
- Next task: Task 7, indicator columns and panels.

### 2026-04-26 Task 7 Completed

- Added pane layout support for a price pane plus MACD and RSI indicator panes.
- Added price overlay and panel indicator geometry for EMA, Bollinger Bands, MACD, MACD signal, RSI, and MACD histogram bars.
- Wired `GLChartRenderer` to render candles in the price pane, draw EMA/BB against price scaling, and draw MACD/RSI with independent panel scaling.
- Connected toolbar EMA/BB/MACD-RSI toggles to `ChartSceneModel` state, forwarded the selected indicator columns into `CandleWindowRequest`, and request a current-window reload when enabling indicator columns that may not be loaded yet.
- Added native tests for pane layout, indicator line geometry, histogram geometry, and scene-model indicator request state.
- Verified with `C:\Build\TradeReview-native-task7-off-msvc` using DuckDB OFF:
  - configure exited 0;
  - `cmake --build C:\Build\TradeReview-native-task7-off-msvc --target tradereview_native_tests --config Debug` exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task7-off-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`;
  - `cmake --build C:\Build\TradeReview-native-task7-off-msvc --target tradereview_native --config Debug` exited 0.
- Verified with `C:\Build\TradeReview-native-task7-msvc` using DuckDB ON:
  - configure exited 0;
  - full build exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task7-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- Static verification: `rg -n "PaneLayout|IndicatorLayer|HistogramLayer|MACD|RSI|EMA20|BB" native/chart`.
- No native exe was launched.
- Next task: Task 8, multi-chart workspace and layouts.

### 2026-04-26 Task 8 Completed

- Added `ChartPanelWidget` and `ChartWorkspaceState` so the native workspace owns four persistent chart slots, each with its own toolbar, view, selected period, indicator state, and generation counter.
- Rebuilt `ChartWorkspaceWidget` around managed panels with Tabs, Vertical, Dual Vertical, and Grid 2x2 layouts, plus 1-4 enabled chart support.
- Wired the main controls bar to change chart count and layout mode, and forwarded period changes from each panel into independent window reload requests.
- Updated `DataLoadController` so one opened dataset metadata snapshot is shared while enabled charts submit independent candle-window requests with per-chart period, pixel width, indicators, and generation.
- Added native state tests for chart count clamping, layout mode storage, per-chart period preservation, and active chart clamping.
- Verified the RED step first: DuckDB OFF test build failed on missing `ChartWorkspaceState.h` before implementation.
- Verified with `C:\Build\TradeReview-native-task8-off-msvc` using DuckDB OFF:
  - `cmake --build C:\Build\TradeReview-native-task8-off-msvc --target tradereview_native_tests --config Debug` exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task8-off-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`;
  - `cmake --build C:\Build\TradeReview-native-task8-off-msvc --target tradereview_native --config Debug` exited 0.
- Verified with `C:\Build\TradeReview-native-task8-msvc` using DuckDB ON:
  - configure exited 0;
  - full build exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task8-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- Static verification: `rg -n "ChartPanelWidget|Grid 2x2|Dual Vertical|setChartCount|setLayoutMode|period" native`.
- No native exe was launched.
- Next task: Task 9, crosshair and time sync.

### 2026-04-26 Task 9 Completed

- Added `CrosshairSyncController` as the sync module's concrete controller for registered chart callbacks, enabled-chart filtering, canonical timestamp/price fan-out, center-time sync, Y-center sync, and feedback-loop suppression.
- Added dense timestamp interpolation in `ChartIndexMapper` and range-preserving center movement in `ChartInteractionController` so incoming timestamps can map into each chart's local dense x space.
- Wired `ChartViewWidget` to emit local crosshair timestamp/price from mouse movement, accept synced crosshair state, center on synced timestamps, and store optional synced Y-center price.
- Wired `ChartWorkspaceWidget` to register chart views with the sync controller and refresh enabled/disabled chart state from the workspace layout.
- Added native tests for canonical crosshair fan-out, disabled source/target skipping, reentrant feedback-loop suppression, center sync, Y-center sync, timestamp-to-dense interpolation, and center-on-dense range preservation.
- Verified RED first:
  - OFF test build failed before implementation on missing `tradereview/sync/CrosshairSyncController.h`;
  - OFF test build failed before dense/center helper implementation on missing `ChartIndexMapper::dense_x_from_timestamp` and `ChartInteractionController::center_on_dense_x`.
- Verified with `C:\Build\TradeReview-native-task9-off-msvc` using DuckDB OFF:
  - configure exited 0;
  - full build exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task9-off-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- Verified with `C:\Build\TradeReview-native-task9-msvc` using DuckDB ON:
  - configure exited 0;
  - full build exited 0;
  - `ctest --test-dir C:\Build\TradeReview-native-task9-msvc --output-on-failure -C Debug` reported `100% tests passed, 0 tests failed out of 1`.
- Static verification: `rg -n "CrosshairSyncController|timestamp_ns|price|sync_crosshair|center" native --glob '!build/**'`.
- `git diff --check` exited 0 with only LF-to-CRLF notices.
- No native exe was launched.
- Next task: Task 10, drawing store and Fib math.
