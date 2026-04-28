# Native Manual Verification

This checklist is for the C++ Qt/OpenGL native version. Per `AGENTS.md`, Codex should not proactively compile or run this native app unless you explicitly ask. Use this document when you manually verify the current branch.

## Source Tree

- Native CMake entry: `native/CMakeLists.txt`
- Recommended working branch/worktree: `cpp-opengl-native-m0m1`
- Recommended temporary build root: `C:\Build`
- Avoid reusing stale build directories whose CMake cache points at another source path.

## Qt Creator Manual Build

1. Open Qt Creator.
2. Choose `File > Open File or Project`.
3. Open the worktree's `native/CMakeLists.txt`, not the repository root CMake file.
4. Select a MSVC 64-bit Qt 6 kit, for example `Desktop Qt 6.11.0 MSVC2022 64bit`.
5. Set the build directory to a local path such as `C:\Build\TradeReview-native-qtcreator`.
6. Configure with these CMake options for a real DuckDB app run:
   - `TRADEREVIEW_NATIVE_BUILD_APP=ON`
   - `TRADEREVIEW_NATIVE_BUILD_TESTS=ON`
   - `TRADEREVIEW_NATIVE_WITH_DUCKDB=ON`
7. If DuckDB is not found automatically, add explicit values:
   - `DUCKDB_INCLUDE_DIR=<folder containing duckdb.h>`
   - `DUCKDB_LIBRARY=<path to duckdb.lib>`
   - `DUCKDB_DLL=<path to duckdb.dll>`
8. Build `tradereview_native`.
9. Run `tradereview_native` from Qt Creator.

## PowerShell Manual Build

Use a new build directory if CMake reports that the cache belongs to another source path.

```powershell
$src = "\\10.0.0.23\code\gold\TradeReview\.worktrees\cpp-opengl-native-m0m1\native"
$build = "C:\Build\TradeReview-native-manual-msvc"
$qt = "C:\Qt\6.11.0\msvc2022_64"
$env:TRADEREVIEW_NATIVE_SRC = $src
$env:TRADEREVIEW_NATIVE_BUILD = $build
$env:TRADEREVIEW_QT_PREFIX = $qt
```

Configure and build from a Visual Studio developer environment:

```powershell
cmd.exe /v:on /d /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"" -arch=amd64 && set ""PATH=C:\Qt\6.11.0\msvc2022_64\bin;C:\Qt\Tools\CMake_64\bin;!PATH!"" && cmake -S ""%TRADEREVIEW_NATIVE_SRC%"" -B ""%TRADEREVIEW_NATIVE_BUILD%"" -DCMAKE_PREFIX_PATH=""%TRADEREVIEW_QT_PREFIX%"" -DTRADEREVIEW_NATIVE_BUILD_APP=ON -DTRADEREVIEW_NATIVE_BUILD_TESTS=ON -DTRADEREVIEW_NATIVE_WITH_DUCKDB=ON && cmake --build ""%TRADEREVIEW_NATIVE_BUILD%"" --config Debug"
```

If DuckDB is not found automatically, add:

```powershell
-DDUCKDB_INCLUDE_DIR=<folder containing duckdb.h> -DDUCKDB_LIBRARY=<path to duckdb.lib> -DDUCKDB_DLL=<path to duckdb.dll>
```

Optional test command:

```powershell
ctest --test-dir C:\Build\TradeReview-native-manual-msvc --output-on-failure -C Debug
```

Run command after a successful build:

```powershell
C:\Build\TradeReview-native-manual-msvc\app\tradereview_native.exe
```

## Smoke Checklist

Use a small DuckDB dataset first so failures are easy to isolate.

- Load dataset: open a `.duckdb` file. The status bar should show the loaded row count and visible time range.
- Single chart: confirm the first chart renders candles, price scale, time scale, indicators, and crosshair movement.
- Four charts: set chart count to 4, switch `Tabs`, `Vertical`, `Dual Vertical`, and `Grid 2x2`; each visible chart should keep its own period and reload without blocking the others.
- Period switch: change a chart period, for example `1m` to `5m`; the chart should reload that period and preserve the current visible time intent.
- pan/zoom: drag and wheel inside the chart. Small pan operations should feel immediate; edge pan should request a background window reload.
- Indicators: toggle EMA, BB, and MACD/RSI. Missing optional indicator columns should not crash the app.
- Drawing: test `H`, `V`, `Line`, `Fib`, `Fib Ext`, `Sel`, `Clear`, and Delete/Backspace. Drawings should remain tied to timestamp/price after period changes.
- Replay: enable `Replay Mode`, play, pause, step forward/back, and adjust speed. Replay should stop at dataset end and keep chart windows updating.
- Date jump and session: jump to a specific time, use `Save View`, close the app, reopen, and confirm dataset path, center time, chart count, layout, and periods restore.
- Error handling: try a missing file or incompatible schema. The app should show a clear status/dialog message and remain usable.

## Performance Expectations

- Small pan: moving within the loaded range should update the viewport without a new query and without visible stutter.
- Edge pan: moving close to the loaded edge should prefetch/reload a wider time window in the background.
- Large time jump: jumping far outside the current loaded range should issue a fresh data request and show a lightweight loading overlay for the affected chart.
- Wide visible range: chart rendering should use LOD behavior so dense candle ranges do not require one visible body per source row.
- Multi-chart partial failure: if one chart query fails, other chart windows should still apply and stay interactive.
- replay: replay currently streams capped tick chunks and rebuilds bars forward from the replay cursor; historical indicator warmup during replay remains a future improvement.

## Pass Criteria

- All smoke scenarios above complete without a crash.
- Status messages identify file/schema/query failures clearly.
- No chart stays permanently in loading state after success or error.
- No stale data visibly replaces newer data after rapid pan, period switch, or date jump.
- The final manual run uses a build directory whose CMake cache points at the intended worktree.

## C++ 终端化界面手动验收

- 无数据启动时，主窗口应显示顶部菜单、主控制栏、左侧绘图工具栏、中央图表空状态、右侧信息面板和底部状态栏。
- 点击未加载数据前的回放/绘图按钮时，不应崩溃；未接线动作应显示明确状态消息。
- DuckDB OFF 构建加载数据时，仍应弹出错误框，同时底部状态栏和右侧面板保留清晰状态。
- DuckDB ON 构建加载数据后，右侧面板应显示数据集名称、数据范围、可见范围、布局和回放状态。
- 切换图表数量和布局模式后，中央图表区、右侧面板和状态栏不应出现文字重叠。
- 切换周期、EMA、BB、MACD/RSI 后，原有图表刷新和指标开关行为应保持。
- 左侧 H/V/Line/Fib/Fib Ext/Clear 工具应走现有绘图 action，不新增未定义行为。
