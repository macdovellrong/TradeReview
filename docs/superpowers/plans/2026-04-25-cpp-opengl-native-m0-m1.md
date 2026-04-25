# C++/Qt/OpenGL Native M0-M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first native C++ foundation for TradeReview: a modular `native/` CMake project with pure data/view models, a Qt Widgets shell, and an OpenGL chart widget capable of receiving a candle window.

**Architecture:** Keep the Python application untouched while adding a new native application under `native/`. Split C++ code by product modules: `core`, `data`, `chart`, `drawing`, `replay`, `sync`, `app`, and `tests`. M0 establishes the build and directories; M1 establishes pure model behavior and a minimal Qt/OpenGL chart surface without yet implementing full DuckDB I/O or product features.

**Tech Stack:** C++20, CMake, Qt 6 Widgets/OpenGLWidgets/OpenGL, OpenGL 3.3 core profile, CTest, PowerShell, Git.

---

## Scope

This plan covers only M0 and the first half of M1 from the design document:

- Native CMake workspace.
- Module folder layout.
- Pure C++ core types and tests.
- LOD/window range behavior matching the Python rules.
- Chart index mapping with dense x semantics.
- Minimal Qt Widgets application shell.
- Minimal `QOpenGLWidget` chart component.
- Scene model that accepts `CandleWindow`.

This plan intentionally does not implement:

- DuckDB C API integration.
- Async worker pool.
- Actual OpenGL candle geometry.
- Drawing tools.
- Replay.
- Multi-chart workspace.
- Packaging.

Those belong to later plans: M1 data I/O, M2 async/LOD cache, M3 workspace sync, M4 drawing, M5 replay.

## Target File Structure

Create:

```text
native/
  CMakeLists.txt
  cmake/
    TradeReviewNativeOptions.cmake
  app/
    CMakeLists.txt
    include/tradereview/app/MainWindow.h
    include/tradereview/app/NativeApp.h
    src/MainWindow.cpp
    src/NativeApp.cpp
    src/main.cpp
  core/
    CMakeLists.txt
    include/tradereview/core/Assertions.h
    include/tradereview/core/Period.h
    include/tradereview/core/TimeRange.h
    src/Period.cpp
    src/TimeRange.cpp
  data/
    CMakeLists.txt
    include/tradereview/data/CandleWindow.h
    include/tradereview/data/DataSetInfo.h
    include/tradereview/data/IDataStore.h
    src/CandleWindow.cpp
  chart/
    CMakeLists.txt
    include/tradereview/chart/ChartIndexMapper.h
    include/tradereview/chart/ChartSceneModel.h
    include/tradereview/chart/ChartViewWidget.h
    include/tradereview/chart/LodResolver.h
    include/tradereview/chart/Windowing.h
    src/ChartIndexMapper.cpp
    src/ChartSceneModel.cpp
    src/ChartViewWidget.cpp
    src/LodResolver.cpp
    src/Windowing.cpp
    rendering/include/tradereview/chart/rendering/GLChartRenderer.h
    rendering/src/GLChartRenderer.cpp
  drawing/
    CMakeLists.txt
    include/tradereview/drawing/DrawingTypes.h
    src/DrawingTypes.cpp
  replay/
    CMakeLists.txt
    include/tradereview/replay/ReplayTypes.h
    src/ReplayTypes.cpp
  sync/
    CMakeLists.txt
    include/tradereview/sync/SyncTypes.h
    src/SyncTypes.cpp
  tests/
    CMakeLists.txt
    test_main.cpp
    core/test_period.cpp
    core/test_time_range.cpp
    chart/test_lod_resolver.cpp
    chart/test_windowing.cpp
    chart/test_chart_index_mapper.cpp
    data/test_candle_window.cpp
```

Modify:

- None outside `native/` and this plan.

## Build Commands

Use these commands from the repository root:

```powershell
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Debug
cmake --build native-build --config Debug
ctest --test-dir native-build --output-on-failure -C Debug
```

If Qt is not installed or CMake cannot find it, configure will fail. That is an environment issue, not a design issue. The pure core tasks should still be written so they can later run as soon as the native toolchain is installed.

## Task 1: Native CMake Skeleton

**Files:**
- Create: `native/CMakeLists.txt`
- Create: `native/cmake/TradeReviewNativeOptions.cmake`
- Create: `native/tests/CMakeLists.txt`
- Create: `native/tests/test_main.cpp`
- Create: module `CMakeLists.txt` files under `native/app`, `native/core`, `native/data`, `native/chart`, `native/drawing`, `native/replay`, `native/sync`

- [ ] **Step 1: Write the failing top-level configure expectation**

Run:

```powershell
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Debug
```

Expected: FAIL because `native/CMakeLists.txt` does not exist.

- [ ] **Step 2: Create `native/cmake/TradeReviewNativeOptions.cmake`**

```cmake
option(TRADEREVIEW_NATIVE_BUILD_APP "Build the Qt native application" ON)
option(TRADEREVIEW_NATIVE_BUILD_TESTS "Build native C++ tests" ON)
option(TRADEREVIEW_NATIVE_WARNINGS_AS_ERRORS "Treat native warnings as errors" OFF)
```

- [ ] **Step 3: Create `native/CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.24)

project(TradeReviewNative LANGUAGES CXX)

include(cmake/TradeReviewNativeOptions.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
    if(TRADEREVIEW_NATIVE_WARNINGS_AS_ERRORS)
        add_compile_options(/WX)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(TRADEREVIEW_NATIVE_WARNINGS_AS_ERRORS)
        add_compile_options(-Werror)
    endif()
endif()

add_subdirectory(core)
add_subdirectory(data)
add_subdirectory(chart)
add_subdirectory(drawing)
add_subdirectory(replay)
add_subdirectory(sync)

if(TRADEREVIEW_NATIVE_BUILD_APP)
    add_subdirectory(app)
endif()

if(TRADEREVIEW_NATIVE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

- [ ] **Step 4: Create minimal module CMake files**

Create `native/core/CMakeLists.txt`:

```cmake
add_library(tradereview_core STATIC)

target_include_directories(tradereview_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
```

Create `native/data/CMakeLists.txt`:

```cmake
add_library(tradereview_data STATIC)

target_include_directories(tradereview_data
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tradereview_data
    PUBLIC
        tradereview_core
)
```

Create `native/chart/CMakeLists.txt`:

```cmake
add_library(tradereview_chart STATIC)

target_include_directories(tradereview_chart
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/rendering/include
)

target_link_libraries(tradereview_chart
    PUBLIC
        tradereview_core
        tradereview_data
)
```

Create `native/drawing/CMakeLists.txt`:

```cmake
add_library(tradereview_drawing STATIC)

target_include_directories(tradereview_drawing
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tradereview_drawing
    PUBLIC
        tradereview_core
)
```

Create `native/replay/CMakeLists.txt`:

```cmake
add_library(tradereview_replay STATIC)

target_include_directories(tradereview_replay
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tradereview_replay
    PUBLIC
        tradereview_core
        tradereview_data
)
```

Create `native/sync/CMakeLists.txt`:

```cmake
add_library(tradereview_sync STATIC)

target_include_directories(tradereview_sync
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tradereview_sync
    PUBLIC
        tradereview_core
)
```

Create `native/app/CMakeLists.txt`:

```cmake
find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets)

qt_standard_project_setup()

qt_add_executable(tradereview_native
    src/main.cpp
)

target_include_directories(tradereview_native
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tradereview_native
    PRIVATE
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        tradereview_chart
        tradereview_data
        tradereview_drawing
        tradereview_replay
        tradereview_sync
)
```

- [ ] **Step 5: Create the initial test runner**

Create `native/tests/test_main.cpp`:

```cpp
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace tradereview::tests {

struct TestCase {
    std::string name;
    std::function<void()> run;
};

std::vector<TestCase>& registry()
{
    static std::vector<TestCase> tests;
    return tests;
}

void register_test(std::string name, std::function<void()> run)
{
    registry().push_back(TestCase{std::move(name), std::move(run)});
}

} // namespace tradereview::tests

int main()
{
    int failures = 0;
    for (const auto& test : tradereview::tests::registry()) {
        try {
            test.run();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& ex) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << ex.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }
    if (failures != 0) {
        return 1;
    }
    std::cout << tradereview::tests::registry().size() << " native test(s) passed\n";
    return 0;
}
```

Create `native/tests/CMakeLists.txt`:

```cmake
add_executable(tradereview_native_tests
    test_main.cpp
)

target_link_libraries(tradereview_native_tests
    PRIVATE
        tradereview_core
        tradereview_data
        tradereview_chart
        tradereview_drawing
        tradereview_replay
        tradereview_sync
)

add_test(NAME tradereview_native_tests COMMAND tradereview_native_tests)
```

- [ ] **Step 6: Create temporary app entry point**

Create `native/app/src/main.cpp`:

```cpp
#include <QApplication>
#include <QLabel>

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    QLabel label("TradeReview Native");
    label.resize(480, 120);
    label.show();
    return QApplication::exec();
}
```

- [ ] **Step 7: Run configure/build/tests**

Run:

```powershell
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Debug
cmake --build native-build --config Debug
ctest --test-dir native-build --output-on-failure -C Debug
```

Expected: configure and build pass if Qt/C++ toolchain is installed; tests pass with `0 native test(s) passed`.

- [ ] **Step 8: Commit**

```powershell
git add native
git commit -m "搭建C++原生工程骨架"
```

## Task 2: Test Harness Assertions

**Files:**
- Create: `native/core/include/tradereview/core/Assertions.h`
- Modify: `native/tests/CMakeLists.txt`
- Create: `native/tests/core/test_assertions.cpp`

- [ ] **Step 1: Write failing assertion test**

Create `native/tests/core/test_assertions.cpp`:

```cpp
#include <stdexcept>

#include "tradereview/core/Assertions.h"

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {
struct RegisterAssertionsTests {
    RegisterAssertionsTests()
    {
        tradereview::tests::register_test("assert_equal accepts equal integers", [] {
            tradereview::core::assert_equal(3, 3, "integer equality");
        });
        tradereview::tests::register_test("assert_true throws on false", [] {
            bool threw = false;
            try {
                tradereview::core::assert_true(false, "false condition");
            } catch (const std::runtime_error&) {
                threw = true;
            }
            tradereview::core::assert_true(threw, "assert_true should throw");
        });
    }
} register_assertions_tests;
} // namespace
```

Modify `native/tests/CMakeLists.txt`:

```cmake
add_executable(tradereview_native_tests
    test_main.cpp
    core/test_assertions.cpp
)
```

- [ ] **Step 2: Run test build and verify failure**

Run:

```powershell
cmake --build native-build --config Debug
```

Expected: FAIL because `tradereview/core/Assertions.h` does not exist.

- [ ] **Step 3: Implement assertions**

Create `native/core/include/tradereview/core/Assertions.h`:

```cpp
#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tradereview::core {

inline void assert_true(bool value, const std::string& message)
{
    if (!value) {
        throw std::runtime_error(message);
    }
}

template <typename T, typename U>
void assert_equal(const T& actual, const U& expected, const std::string& message)
{
    if (!(actual == expected)) {
        std::ostringstream out;
        out << message << ": expected " << expected << ", got " << actual;
        throw std::runtime_error(out.str());
    }
}

inline void assert_near(double actual, double expected, double tolerance, const std::string& message)
{
    if (std::abs(actual - expected) > tolerance) {
        std::ostringstream out;
        out << message << ": expected " << expected << ", got " << actual;
        throw std::runtime_error(out.str());
    }
}

} // namespace tradereview::core
```

- [ ] **Step 4: Run tests**

Run:

```powershell
cmake --build native-build --config Debug
ctest --test-dir native-build --output-on-failure -C Debug
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add native/core/include/tradereview/core/Assertions.h native/tests/CMakeLists.txt native/tests/core/test_assertions.cpp
git commit -m "增加C++测试断言工具"
```

## Task 3: Core TimeRange and Period Types

**Files:**
- Create: `native/core/include/tradereview/core/TimeRange.h`
- Create: `native/core/src/TimeRange.cpp`
- Create: `native/core/include/tradereview/core/Period.h`
- Create: `native/core/src/Period.cpp`
- Modify: `native/core/CMakeLists.txt`
- Modify: `native/tests/CMakeLists.txt`
- Create: `native/tests/core/test_time_range.cpp`
- Create: `native/tests/core/test_period.cpp`

- [ ] **Step 1: Write failing TimeRange tests**

Create `native/tests/core/test_time_range.cpp`:

```cpp
#include <functional>
#include <string>

#include "tradereview/core/Assertions.h"
#include "tradereview/core/TimeRange.h"

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {
struct RegisterTimeRangeTests {
    RegisterTimeRangeTests()
    {
        tradereview::tests::register_test("time range span is end minus start", [] {
            const tradereview::core::TimeRange range{100, 250};
            tradereview::core::assert_equal(range.span_ns(), int64_t{150}, "span");
        });
        tradereview::tests::register_test("time range normalizes reversed endpoints", [] {
            const auto range = tradereview::core::TimeRange::normalized(250, 100);
            tradereview::core::assert_equal(range.start_ns, int64_t{100}, "start");
            tradereview::core::assert_equal(range.end_ns, int64_t{250}, "end");
        });
    }
} register_time_range_tests;
} // namespace
```

- [ ] **Step 2: Write failing Period tests**

Create `native/tests/core/test_period.cpp`:

```cpp
#include <functional>
#include <string>

#include "tradereview/core/Assertions.h"
#include "tradereview/core/Period.h"

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {
struct RegisterPeriodTests {
    RegisterPeriodTests()
    {
        tradereview::tests::register_test("period parses minute and hour strings", [] {
            tradereview::core::assert_equal(tradereview::core::period_seconds("1min"), int64_t{60}, "1min");
            tradereview::core::assert_equal(tradereview::core::period_seconds("4h"), int64_t{14400}, "4h");
        });
        tradereview::tests::register_test("period keeps month distinct from minute", [] {
            tradereview::core::assert_equal(tradereview::core::duckdb_candle_table("1min"), std::string{"candles_1m"}, "minute table");
            tradereview::core::assert_equal(tradereview::core::duckdb_candle_table("1M"), std::string{"candles_1mo"}, "month table");
        });
    }
} register_period_tests;
} // namespace
```

Modify `native/tests/CMakeLists.txt` to include both new test files.

- [ ] **Step 3: Run build and verify failure**

Run:

```powershell
cmake --build native-build --config Debug
```

Expected: FAIL because `TimeRange.h` and `Period.h` do not exist.

- [ ] **Step 4: Implement TimeRange**

Create `native/core/include/tradereview/core/TimeRange.h`:

```cpp
#pragma once

#include <cstdint>

namespace tradereview::core {

struct TimeRange {
    int64_t start_ns = 0;
    int64_t end_ns = 0;

    [[nodiscard]] int64_t span_ns() const;
    [[nodiscard]] bool contains(int64_t timestamp_ns) const;
    [[nodiscard]] static TimeRange normalized(int64_t first_ns, int64_t second_ns);
};

} // namespace tradereview::core
```

Create `native/core/src/TimeRange.cpp`:

```cpp
#include "tradereview/core/TimeRange.h"

#include <algorithm>

namespace tradereview::core {

int64_t TimeRange::span_ns() const
{
    return end_ns - start_ns;
}

bool TimeRange::contains(int64_t timestamp_ns) const
{
    return start_ns <= timestamp_ns && timestamp_ns <= end_ns;
}

TimeRange TimeRange::normalized(int64_t first_ns, int64_t second_ns)
{
    return TimeRange{std::min(first_ns, second_ns), std::max(first_ns, second_ns)};
}

} // namespace tradereview::core
```

- [ ] **Step 5: Implement Period**

Create `native/core/include/tradereview/core/Period.h`:

```cpp
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace tradereview::core {

[[nodiscard]] std::optional<int64_t> try_period_seconds(const std::string& period);
[[nodiscard]] int64_t period_seconds(const std::string& period);
[[nodiscard]] std::string duckdb_candle_table(const std::string& period);

} // namespace tradereview::core
```

Create `native/core/src/Period.cpp`:

```cpp
#include "tradereview/core/Period.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace tradereview::core {
namespace {

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool ends_with(const std::string& value, const std::string& suffix)
{
    return value.size() >= suffix.size()
        && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int64_t parse_prefix(const std::string& value, size_t suffix_len)
{
    return std::stoll(value.substr(0, value.size() - suffix_len));
}

} // namespace

std::optional<int64_t> try_period_seconds(const std::string& period)
{
    const std::string original = period;
    const std::string value = lower_copy(period);
    try {
        if (ends_with(value, "min")) {
            return parse_prefix(value, 3) * 60;
        }
        if (ends_with(value, "s")) {
            return parse_prefix(value, 1);
        }
        if (ends_with(value, "h")) {
            return parse_prefix(value, 1) * 60 * 60;
        }
        if (ends_with(value, "d")) {
            return parse_prefix(value, 1) * 24 * 60 * 60;
        }
        if (ends_with(value, "w")) {
            return parse_prefix(value, 1) * 7 * 24 * 60 * 60;
        }
        if (ends_with(original, "M")) {
            return parse_prefix(original, 1) * 30 * 24 * 60 * 60;
        }
    } catch (...) {
        return std::nullopt;
    }
    return std::nullopt;
}

int64_t period_seconds(const std::string& period)
{
    const auto seconds = try_period_seconds(period);
    if (!seconds.has_value()) {
        throw std::invalid_argument("Unsupported period: " + period);
    }
    return *seconds;
}

std::string duckdb_candle_table(const std::string& period)
{
    const std::string lower = lower_copy(period);
    if (ends_with(lower, "min")) {
        return "candles_" + lower.substr(0, lower.size() - 3) + "m";
    }
    if (ends_with(period, "M")) {
        return "candles_" + period.substr(0, period.size() - 1) + "mo";
    }
    return "candles_" + lower;
}

} // namespace tradereview::core
```

- [ ] **Step 6: Add core sources to CMake**

Modify `native/core/CMakeLists.txt`:

```cmake
add_library(tradereview_core STATIC
    src/Period.cpp
    src/TimeRange.cpp
)
```

- [ ] **Step 7: Run tests**

Run:

```powershell
cmake --build native-build --config Debug
ctest --test-dir native-build --output-on-failure -C Debug
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add native/core native/tests
git commit -m "增加C++核心时间和周期类型"
```

## Task 4: CandleWindow Data Model

**Files:**
- Create: `native/data/include/tradereview/data/CandleWindow.h`
- Create: `native/data/include/tradereview/data/DataSetInfo.h`
- Create: `native/data/include/tradereview/data/IDataStore.h`
- Create: `native/data/src/CandleWindow.cpp`
- Modify: `native/data/CMakeLists.txt`
- Modify: `native/tests/CMakeLists.txt`
- Create: `native/tests/data/test_candle_window.cpp`

- [ ] **Step 1: Write failing candle window tests**

Create `native/tests/data/test_candle_window.cpp`:

```cpp
#include <functional>
#include <string>

#include "tradereview/core/Assertions.h"
#include "tradereview/data/CandleWindow.h"

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {
struct RegisterCandleWindowTests {
    RegisterCandleWindowTests()
    {
        tradereview::tests::register_test("candle window reports row count", [] {
            tradereview::data::CandleWindow window;
            window.timestamp_ns = {100, 200, 300};
            window.open = {1.0, 2.0, 3.0};
            window.high = {2.0, 3.0, 4.0};
            window.low = {0.5, 1.5, 2.5};
            window.close = {1.5, 2.5, 3.5};
            window.volume = {10.0, 20.0, 30.0};

            tradereview::core::assert_equal(window.row_count(), size_t{3}, "row count");
            tradereview::core::assert_true(window.has_consistent_columns(), "columns");
        });
    }
} register_candle_window_tests;
} // namespace
```

Modify `native/tests/CMakeLists.txt` to include `data/test_candle_window.cpp`.

- [ ] **Step 2: Run build and verify failure**

Run:

```powershell
cmake --build native-build --config Debug
```

Expected: FAIL because `CandleWindow.h` does not exist.

- [ ] **Step 3: Implement data model headers**

Create `native/data/include/tradereview/data/CandleWindow.h`:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tradereview/core/TimeRange.h"

namespace tradereview::data {

struct CandleWindow {
    uint64_t chart_id = 0;
    uint64_t generation = 0;
    std::string requested_period;
    std::string actual_period;
    core::TimeRange loaded_range;
    core::TimeRange visible_range;
    std::vector<int64_t> timestamp_ns;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;
    std::unordered_map<std::string, std::vector<double>> indicators;
    bool from_cache = false;

    [[nodiscard]] size_t row_count() const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] bool has_consistent_columns() const;
};

} // namespace tradereview::data
```

Create `native/data/include/tradereview/data/DataSetInfo.h`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tradereview/core/TimeRange.h"

namespace tradereview::data {

struct DataSetInfo {
    std::string path;
    int64_t tick_count = 0;
    core::TimeRange tick_range;
    std::vector<std::string> periods;
    std::vector<std::string> indicators;
    std::string schema_version;
    std::string indicator_version;
};

} // namespace tradereview::data
```

Create `native/data/include/tradereview/data/IDataStore.h`:

```cpp
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"
#include "tradereview/data/DataSetInfo.h"

namespace tradereview::data {

struct CandleWindowRequest {
    uint64_t chart_id = 0;
    uint64_t generation = 0;
    std::string requested_period;
    core::TimeRange visible_range;
    int pixel_width = 0;
    double buffer_multiplier = 2.0;
    bool include_indicators = true;
    int warmup_bars = 0;
};

struct TickSlice {
    std::vector<int64_t> timestamp_ns;
    std::vector<double> price;
    std::vector<double> volume;
};

struct ReplayChunk {
    TickSlice ticks;
    bool reached_end = false;
};

class IDataStore {
public:
    virtual ~IDataStore() = default;
    virtual DataSetInfo open_readonly(const std::string& path) = 0;
    virtual CandleWindow query_candles(const CandleWindowRequest& request) = 0;
    virtual TickSlice query_ticks(core::TimeRange range, size_t max_rows) = 0;
    virtual ReplayChunk query_replay_ticks(int64_t from_ns, int64_t to_ns, size_t max_ticks) = 0;
};

} // namespace tradereview::data
```

- [ ] **Step 4: Implement CandleWindow**

Create `native/data/src/CandleWindow.cpp`:

```cpp
#include "tradereview/data/CandleWindow.h"

namespace tradereview::data {

size_t CandleWindow::row_count() const
{
    return timestamp_ns.size();
}

bool CandleWindow::empty() const
{
    return row_count() == 0;
}

bool CandleWindow::has_consistent_columns() const
{
    const size_t rows = row_count();
    if (open.size() != rows || high.size() != rows || low.size() != rows || close.size() != rows || volume.size() != rows) {
        return false;
    }
    for (const auto& [name, values] : indicators) {
        (void)name;
        if (values.size() != rows) {
            return false;
        }
    }
    return true;
}

} // namespace tradereview::data
```

Modify `native/data/CMakeLists.txt`:

```cmake
add_library(tradereview_data STATIC
    src/CandleWindow.cpp
)
```

- [ ] **Step 5: Run tests**

Run:

```powershell
cmake --build native-build --config Debug
ctest --test-dir native-build --output-on-failure -C Debug
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add native/data native/tests
git commit -m "增加C++窗口K线数据模型"
```

## Task 5: LOD Resolver and Windowing Helpers

**Files:**
- Create: `native/chart/include/tradereview/chart/LodResolver.h`
- Create: `native/chart/include/tradereview/chart/Windowing.h`
- Create: `native/chart/src/LodResolver.cpp`
- Create: `native/chart/src/Windowing.cpp`
- Modify: `native/chart/CMakeLists.txt`
- Modify: `native/tests/CMakeLists.txt`
- Create: `native/tests/chart/test_lod_resolver.cpp`
- Create: `native/tests/chart/test_windowing.cpp`

- [ ] **Step 1: Write failing LOD tests**

Create `native/tests/chart/test_lod_resolver.cpp`:

```cpp
#include <functional>
#include <string>
#include <vector>

#include "tradereview/chart/LodResolver.h"
#include "tradereview/core/Assertions.h"
#include "tradereview/core/TimeRange.h"

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {
struct RegisterLodResolverTests {
    RegisterLodResolverTests()
    {
        tradereview::tests::register_test("lod keeps requested period when density fits", [] {
            const std::vector<std::string> periods{"1min", "5min", "1h", "1D"};
            const tradereview::core::TimeRange six_hours{0, 6LL * 60 * 60 * 1000000000};
            const auto actual = tradereview::chart::choose_lod_period("1min", six_hours, 1600, periods);
            tradereview::core::assert_equal(actual, std::string{"1min"}, "period");
        });
        tradereview::tests::register_test("lod chooses coarser period for multi-year view", [] {
            const std::vector<std::string> periods{"30s", "1min", "5min", "1h", "4h", "1D"};
            const tradereview::core::TimeRange five_years{0, 5LL * 365 * 24 * 60 * 60 * 1000000000};
            const auto actual = tradereview::chart::choose_lod_period("30s", five_years, 1600, periods);
            tradereview::core::assert_equal(actual, std::string{"1D"}, "period");
        });
        tradereview::tests::register_test("lod never chooses finer period than requested", [] {
            const std::vector<std::string> periods{"5min", "1h", "1D"};
            const tradereview::core::TimeRange ten_days{0, 10LL * 24 * 60 * 60 * 1000000000};
            const auto actual = tradereview::chart::choose_lod_period("1h", ten_days, 1600, periods);
            tradereview::core::assert_equal(actual, std::string{"1h"}, "period");
        });
    }
} register_lod_resolver_tests;
} // namespace
```

- [ ] **Step 2: Write failing windowing tests**

Create `native/tests/chart/test_windowing.cpp`:

```cpp
#include <functional>
#include <string>

#include "tradereview/chart/Windowing.h"
#include "tradereview/core/Assertions.h"
#include "tradereview/core/TimeRange.h"

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {
struct RegisterWindowingTests {
    RegisterWindowingTests()
    {
        tradereview::tests::register_test("query window adds buffer on both sides", [] {
            const tradereview::core::TimeRange visible{1000, 2000};
            const auto query = tradereview::chart::build_query_window(visible, 2.0);
            tradereview::core::assert_equal(query.start_ns, int64_t{-1000}, "start");
            tradereview::core::assert_equal(query.end_ns, int64_t{4000}, "end");
        });
        tradereview::tests::register_test("view inside loaded window returns true", [] {
            const tradereview::core::TimeRange visible{1000, 2000};
            const tradereview::core::TimeRange loaded{0, 3000};
            tradereview::core::assert_true(tradereview::chart::is_view_inside_loaded_window(visible, loaded), "inside");
        });
    }
} register_windowing_tests;
} // namespace
```

Modify `native/tests/CMakeLists.txt` to include both files.

- [ ] **Step 3: Run build and verify failure**

Run:

```powershell
cmake --build native-build --config Debug
```

Expected: FAIL because `LodResolver.h` and `Windowing.h` do not exist.

- [ ] **Step 4: Implement LOD resolver**

Create `native/chart/include/tradereview/chart/LodResolver.h`:

```cpp
#pragma once

#include <string>
#include <vector>

#include "tradereview/core/TimeRange.h"

namespace tradereview::chart {

[[nodiscard]] std::string choose_lod_period(
    const std::string& requested_period,
    core::TimeRange visible_range,
    int pixel_width,
    const std::vector<std::string>& available_periods,
    double max_bars_per_pixel = 2.0);

} // namespace tradereview::chart
```

Create `native/chart/src/LodResolver.cpp`:

```cpp
#include "tradereview/chart/LodResolver.h"

#include <algorithm>

#include "tradereview/core/Period.h"

namespace tradereview::chart {

std::string choose_lod_period(
    const std::string& requested_period,
    core::TimeRange visible_range,
    int pixel_width,
    const std::vector<std::string>& available_periods,
    double max_bars_per_pixel)
{
    const int64_t requested_seconds = core::period_seconds(requested_period);
    const double span_seconds = static_cast<double>(visible_range.span_ns()) / 1000000000.0;
    const double max_bars = std::max(1, pixel_width) * max_bars_per_pixel;
    if ((span_seconds / requested_seconds) <= max_bars) {
        return requested_period;
    }

    std::string best = requested_period;
    int64_t best_seconds = requested_seconds;
    bool found = false;
    for (const auto& period : available_periods) {
        const auto seconds = core::try_period_seconds(period);
        if (!seconds.has_value()) {
            continue;
        }
        if (*seconds < requested_seconds) {
            continue;
        }
        if ((span_seconds / *seconds) <= max_bars && (!found || *seconds < best_seconds)) {
            best = period;
            best_seconds = *seconds;
            found = true;
        }
    }
    if (found) {
        return best;
    }
    for (const auto& period : available_periods) {
        const auto seconds = core::try_period_seconds(period);
        if (seconds.has_value() && *seconds >= best_seconds) {
            best = period;
            best_seconds = *seconds;
        }
    }
    return best;
}

} // namespace tradereview::chart
```

- [ ] **Step 5: Implement windowing helpers**

Create `native/chart/include/tradereview/chart/Windowing.h`:

```cpp
#pragma once

#include "tradereview/core/TimeRange.h"

namespace tradereview::chart {

[[nodiscard]] core::TimeRange build_query_window(core::TimeRange visible_range, double buffer_multiplier);
[[nodiscard]] bool is_view_inside_loaded_window(core::TimeRange visible_range, core::TimeRange loaded_range);
[[nodiscard]] bool should_prefetch_window(core::TimeRange visible_range, core::TimeRange loaded_range, double edge_fraction);

} // namespace tradereview::chart
```

Create `native/chart/src/Windowing.cpp`:

```cpp
#include "tradereview/chart/Windowing.h"

#include <algorithm>

namespace tradereview::chart {
namespace {

int64_t positive_span(core::TimeRange range)
{
    return std::max<int64_t>(range.span_ns(), 60LL * 1000000000);
}

} // namespace

core::TimeRange build_query_window(core::TimeRange visible_range, double buffer_multiplier)
{
    const int64_t buffer = static_cast<int64_t>(positive_span(visible_range) * buffer_multiplier);
    return core::TimeRange{visible_range.start_ns - buffer, visible_range.end_ns + buffer};
}

bool is_view_inside_loaded_window(core::TimeRange visible_range, core::TimeRange loaded_range)
{
    return loaded_range.start_ns <= visible_range.start_ns && visible_range.end_ns <= loaded_range.end_ns;
}

bool should_prefetch_window(core::TimeRange visible_range, core::TimeRange loaded_range, double edge_fraction)
{
    const int64_t margin = static_cast<int64_t>(positive_span(visible_range) * edge_fraction);
    return visible_range.start_ns <= loaded_range.start_ns + margin
        || visible_range.end_ns >= loaded_range.end_ns - margin;
}

} // namespace tradereview::chart
```

Modify `native/chart/CMakeLists.txt`:

```cmake
add_library(tradereview_chart STATIC
    src/LodResolver.cpp
    src/Windowing.cpp
)
```

- [ ] **Step 6: Run tests**

Run:

```powershell
cmake --build native-build --config Debug
ctest --test-dir native-build --output-on-failure -C Debug
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add native/chart native/tests
git commit -m "增加C++图表LOD和窗口计算"
```

## Task 6: ChartIndexMapper

**Files:**
- Create: `native/chart/include/tradereview/chart/ChartIndexMapper.h`
- Create: `native/chart/src/ChartIndexMapper.cpp`
- Modify: `native/chart/CMakeLists.txt`
- Modify: `native/tests/CMakeLists.txt`
- Create: `native/tests/chart/test_chart_index_mapper.cpp`

- [ ] **Step 1: Write failing mapper tests**

Create `native/tests/chart/test_chart_index_mapper.cpp`:

```cpp
#include <functional>
#include <string>
#include <vector>

#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/core/Assertions.h"

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {
struct RegisterChartIndexMapperTests {
    RegisterChartIndexMapperTests()
    {
        tradereview::tests::register_test("mapper finds nearest dense x for timestamp", [] {
            tradereview::chart::ChartIndexMapper mapper;
            mapper.set_timestamps({100, 200, 300});
            tradereview::core::assert_equal(mapper.nearest_dense_x(240), 1, "nearest");
            tradereview::core::assert_equal(mapper.timestamp_at_dense_x(2), int64_t{300}, "timestamp");
        });
        tradereview::tests::register_test("mapper extends beyond right edge using median step", [] {
            tradereview::chart::ChartIndexMapper mapper;
            mapper.set_timestamps({100, 200, 300});
            tradereview::core::assert_equal(mapper.timestamp_from_x(4.0), int64_t{500}, "extended timestamp");
        });
    }
} register_chart_index_mapper_tests;
} // namespace
```

Modify `native/tests/CMakeLists.txt` to include the new test.

- [ ] **Step 2: Run build and verify failure**

Run:

```powershell
cmake --build native-build --config Debug
```

Expected: FAIL because `ChartIndexMapper.h` does not exist.

- [ ] **Step 3: Implement mapper**

Create `native/chart/include/tradereview/chart/ChartIndexMapper.h`:

```cpp
#pragma once

#include <cstdint>
#include <vector>

namespace tradereview::chart {

class ChartIndexMapper {
public:
    void set_timestamps(std::vector<int64_t> timestamps_ns);

    [[nodiscard]] bool empty() const;
    [[nodiscard]] int row_count() const;
    [[nodiscard]] int nearest_dense_x(int64_t timestamp_ns) const;
    [[nodiscard]] int64_t timestamp_at_dense_x(int dense_x) const;
    [[nodiscard]] int64_t timestamp_from_x(double x) const;

private:
    [[nodiscard]] int64_t step_ns() const;

    std::vector<int64_t> timestamps_ns_;
};

} // namespace tradereview::chart
```

Create `native/chart/src/ChartIndexMapper.cpp`:

```cpp
#include "tradereview/chart/ChartIndexMapper.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tradereview::chart {

void ChartIndexMapper::set_timestamps(std::vector<int64_t> timestamps_ns)
{
    timestamps_ns_ = std::move(timestamps_ns);
}

bool ChartIndexMapper::empty() const
{
    return timestamps_ns_.empty();
}

int ChartIndexMapper::row_count() const
{
    return static_cast<int>(timestamps_ns_.size());
}

int ChartIndexMapper::nearest_dense_x(int64_t timestamp_ns) const
{
    if (timestamps_ns_.empty()) {
        throw std::runtime_error("ChartIndexMapper has no timestamps");
    }
    const auto it = std::lower_bound(timestamps_ns_.begin(), timestamps_ns_.end(), timestamp_ns);
    if (it == timestamps_ns_.begin()) {
        return 0;
    }
    if (it == timestamps_ns_.end()) {
        return row_count() - 1;
    }
    const int right = static_cast<int>(it - timestamps_ns_.begin());
    const int left = right - 1;
    const auto left_distance = timestamp_ns - timestamps_ns_[left];
    const auto right_distance = timestamps_ns_[right] - timestamp_ns;
    return left_distance <= right_distance ? left : right;
}

int64_t ChartIndexMapper::timestamp_at_dense_x(int dense_x) const
{
    if (dense_x < 0 || dense_x >= row_count()) {
        throw std::out_of_range("dense_x out of range");
    }
    return timestamps_ns_[dense_x];
}

int64_t ChartIndexMapper::timestamp_from_x(double x) const
{
    if (timestamps_ns_.empty()) {
        throw std::runtime_error("ChartIndexMapper has no timestamps");
    }
    const int rounded = static_cast<int>(std::llround(x));
    if (0 <= rounded && rounded < row_count()) {
        return timestamps_ns_[rounded];
    }
    if (rounded < 0) {
        return timestamps_ns_.front() + rounded * step_ns();
    }
    return timestamps_ns_.back() + (rounded - (row_count() - 1)) * step_ns();
}

int64_t ChartIndexMapper::step_ns() const
{
    if (timestamps_ns_.size() < 2) {
        return 60LL * 1000000000;
    }
    std::vector<int64_t> deltas;
    deltas.reserve(timestamps_ns_.size() - 1);
    for (size_t index = 1; index < timestamps_ns_.size(); ++index) {
        deltas.push_back(timestamps_ns_[index] - timestamps_ns_[index - 1]);
    }
    std::sort(deltas.begin(), deltas.end());
    return deltas[deltas.size() / 2];
}

} // namespace tradereview::chart
```

Add `src/ChartIndexMapper.cpp` to `native/chart/CMakeLists.txt`.

- [ ] **Step 4: Run tests**

Run:

```powershell
cmake --build native-build --config Debug
ctest --test-dir native-build --output-on-failure -C Debug
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add native/chart native/tests
git commit -m "增加C++图表时间索引映射"
```

## Task 7: ChartSceneModel

**Files:**
- Create: `native/chart/include/tradereview/chart/ChartSceneModel.h`
- Create: `native/chart/src/ChartSceneModel.cpp`
- Modify: `native/chart/CMakeLists.txt`
- Modify: `native/tests/CMakeLists.txt`
- Create: `native/tests/chart/test_chart_scene_model.cpp`

- [ ] **Step 1: Write failing scene model test**

Create `native/tests/chart/test_chart_scene_model.cpp`:

```cpp
#include <functional>
#include <string>

#include "tradereview/chart/ChartSceneModel.h"
#include "tradereview/core/Assertions.h"
#include "tradereview/data/CandleWindow.h"

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {
tradereview::data::CandleWindow sample_window(uint64_t generation)
{
    tradereview::data::CandleWindow window;
    window.generation = generation;
    window.timestamp_ns = {100, 200};
    window.open = {1.0, 2.0};
    window.high = {2.0, 3.0};
    window.low = {0.5, 1.5};
    window.close = {1.5, 2.5};
    window.volume = {10.0, 20.0};
    return window;
}

struct RegisterChartSceneModelTests {
    RegisterChartSceneModelTests()
    {
        tradereview::tests::register_test("scene model accepts matching generation", [] {
            tradereview::chart::ChartSceneModel model;
            model.bump_generation();
            const bool accepted = model.apply_window(sample_window(model.generation()));
            tradereview::core::assert_true(accepted, "accepted");
            tradereview::core::assert_equal(model.row_count(), size_t{2}, "rows");
        });
        tradereview::tests::register_test("scene model rejects stale generation", [] {
            tradereview::chart::ChartSceneModel model;
            model.bump_generation();
            model.bump_generation();
            const bool accepted = model.apply_window(sample_window(1));
            tradereview::core::assert_true(!accepted, "stale result rejected");
        });
    }
} register_chart_scene_model_tests;
} // namespace
```

Modify `native/tests/CMakeLists.txt` to include this file.

- [ ] **Step 2: Run build and verify failure**

Run:

```powershell
cmake --build native-build --config Debug
```

Expected: FAIL because `ChartSceneModel.h` does not exist.

- [ ] **Step 3: Implement scene model**

Create `native/chart/include/tradereview/chart/ChartSceneModel.h`:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/data/CandleWindow.h"

namespace tradereview::chart {

class ChartSceneModel {
public:
    [[nodiscard]] uint64_t generation() const;
    uint64_t bump_generation();
    [[nodiscard]] bool apply_window(data::CandleWindow window);
    [[nodiscard]] size_t row_count() const;
    [[nodiscard]] const data::CandleWindow& window() const;
    [[nodiscard]] const ChartIndexMapper& index_mapper() const;

private:
    uint64_t generation_ = 0;
    data::CandleWindow window_;
    ChartIndexMapper index_mapper_;
};

} // namespace tradereview::chart
```

Create `native/chart/src/ChartSceneModel.cpp`:

```cpp
#include "tradereview/chart/ChartSceneModel.h"

namespace tradereview::chart {

uint64_t ChartSceneModel::generation() const
{
    return generation_;
}

uint64_t ChartSceneModel::bump_generation()
{
    ++generation_;
    return generation_;
}

bool ChartSceneModel::apply_window(data::CandleWindow window)
{
    if (window.generation != generation_) {
        return false;
    }
    if (!window.has_consistent_columns()) {
        return false;
    }
    index_mapper_.set_timestamps(window.timestamp_ns);
    window_ = std::move(window);
    return true;
}

size_t ChartSceneModel::row_count() const
{
    return window_.row_count();
}

const data::CandleWindow& ChartSceneModel::window() const
{
    return window_;
}

const ChartIndexMapper& ChartSceneModel::index_mapper() const
{
    return index_mapper_;
}

} // namespace tradereview::chart
```

Add `src/ChartSceneModel.cpp` to `native/chart/CMakeLists.txt`.

- [ ] **Step 4: Run tests**

Run:

```powershell
cmake --build native-build --config Debug
ctest --test-dir native-build --output-on-failure -C Debug
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add native/chart native/tests
git commit -m "增加C++图表场景模型"
```

## Task 8: Qt Main Window and Native App Classes

**Files:**
- Create: `native/app/include/tradereview/app/MainWindow.h`
- Create: `native/app/include/tradereview/app/NativeApp.h`
- Create: `native/app/src/MainWindow.cpp`
- Create: `native/app/src/NativeApp.cpp`
- Modify: `native/app/src/main.cpp`
- Modify: `native/app/CMakeLists.txt`

- [ ] **Step 1: Run current app build**

Run:

```powershell
cmake --build native-build --config Debug
```

Expected: PASS from prior tasks. This is the baseline before replacing the temporary `QLabel` app.

- [ ] **Step 2: Add `MainWindow` header**

Create `native/app/include/tradereview/app/MainWindow.h`:

```cpp
#pragma once

#include <QMainWindow>

namespace tradereview::app {

class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
};

} // namespace tradereview::app
```

- [ ] **Step 3: Add `MainWindow` implementation**

Create `native/app/src/MainWindow.cpp`:

```cpp
#include "tradereview/app/MainWindow.h"

#include <QMenuBar>
#include <QStatusBar>

namespace tradereview::app {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("TradeReview Native");
    resize(1400, 950);
    menuBar()->addMenu("&File");
    statusBar()->showMessage("Native workspace ready");
}

} // namespace tradereview::app
```

- [ ] **Step 4: Add `NativeApp` wrapper**

Create `native/app/include/tradereview/app/NativeApp.h`:

```cpp
#pragma once

namespace tradereview::app {

int run_native_app(int argc, char** argv);

} // namespace tradereview::app
```

Create `native/app/src/NativeApp.cpp`:

```cpp
#include "tradereview/app/NativeApp.h"

#include <QApplication>

#include "tradereview/app/MainWindow.h"

namespace tradereview::app {

int run_native_app(int argc, char** argv)
{
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    return QApplication::exec();
}

} // namespace tradereview::app
```

- [ ] **Step 5: Replace app entry point**

Modify `native/app/src/main.cpp`:

```cpp
#include "tradereview/app/NativeApp.h"

int main(int argc, char** argv)
{
    return tradereview::app::run_native_app(argc, argv);
}
```

Modify `native/app/CMakeLists.txt`:

```cmake
qt_add_executable(tradereview_native
    include/tradereview/app/MainWindow.h
    include/tradereview/app/NativeApp.h
    src/MainWindow.cpp
    src/NativeApp.cpp
    src/main.cpp
)
```

- [ ] **Step 6: Run build**

Run:

```powershell
cmake --build native-build --config Debug
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add native/app
git commit -m "增加C++原生主窗口"
```

## Task 9: Minimal QOpenGLWidget Chart Surface

**Files:**
- Create: `native/chart/include/tradereview/chart/ChartViewWidget.h`
- Create: `native/chart/src/ChartViewWidget.cpp`
- Create: `native/chart/rendering/include/tradereview/chart/rendering/GLChartRenderer.h`
- Create: `native/chart/rendering/src/GLChartRenderer.cpp`
- Modify: `native/chart/CMakeLists.txt`
- Modify: `native/app/src/MainWindow.cpp`
- Modify: `native/app/CMakeLists.txt`

- [ ] **Step 1: Write the chart widget header**

Create `native/chart/include/tradereview/chart/ChartViewWidget.h`:

```cpp
#pragma once

#include <QOpenGLWidget>

#include "tradereview/chart/ChartSceneModel.h"
#include "tradereview/chart/rendering/GLChartRenderer.h"

namespace tradereview::chart {

class ChartViewWidget final : public QOpenGLWidget {
    Q_OBJECT

public:
    explicit ChartViewWidget(QWidget* parent = nullptr);

    ChartSceneModel& scene_model();
    const ChartSceneModel& scene_model() const;

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;

private:
    ChartSceneModel scene_model_;
    rendering::GLChartRenderer renderer_;
};

} // namespace tradereview::chart
```

- [ ] **Step 2: Write the renderer interface**

Create `native/chart/rendering/include/tradereview/chart/rendering/GLChartRenderer.h`:

```cpp
#pragma once

namespace tradereview::chart {
class ChartSceneModel;
}

namespace tradereview::chart::rendering {

class GLChartRenderer {
public:
    void initialize();
    void resize(int width, int height);
    void render(const ChartSceneModel& scene_model);

private:
    int width_ = 0;
    int height_ = 0;
};

} // namespace tradereview::chart::rendering
```

- [ ] **Step 3: Implement renderer clear pass**

Create `native/chart/rendering/src/GLChartRenderer.cpp`:

```cpp
#include "tradereview/chart/rendering/GLChartRenderer.h"

#include <QOpenGLFunctions>

#include "tradereview/chart/ChartSceneModel.h"

namespace tradereview::chart::rendering {

void GLChartRenderer::initialize()
{
}

void GLChartRenderer::resize(int width, int height)
{
    width_ = width;
    height_ = height;
}

void GLChartRenderer::render(const ChartSceneModel& scene_model)
{
    (void)scene_model;
    QOpenGLFunctions* gl = QOpenGLContext::currentContext()->functions();
    gl->glViewport(0, 0, width_, height_);
    gl->glClearColor(0.02F, 0.025F, 0.03F, 1.0F);
    gl->glClear(GL_COLOR_BUFFER_BIT);
}

} // namespace tradereview::chart::rendering
```

- [ ] **Step 4: Implement chart widget**

Create `native/chart/src/ChartViewWidget.cpp`:

```cpp
#include "tradereview/chart/ChartViewWidget.h"

namespace tradereview::chart {

ChartViewWidget::ChartViewWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    setMinimumSize(640, 360);
}

ChartSceneModel& ChartViewWidget::scene_model()
{
    return scene_model_;
}

const ChartSceneModel& ChartViewWidget::scene_model() const
{
    return scene_model_;
}

void ChartViewWidget::initializeGL()
{
    renderer_.initialize();
}

void ChartViewWidget::resizeGL(int width, int height)
{
    renderer_.resize(width, height);
}

void ChartViewWidget::paintGL()
{
    renderer_.render(scene_model_);
}

} // namespace tradereview::chart
```

- [ ] **Step 5: Link Qt OpenGL to chart module**

Modify `native/chart/CMakeLists.txt` to find and link Qt:

```cmake
find_package(Qt6 REQUIRED COMPONENTS OpenGL OpenGLWidgets)

add_library(tradereview_chart STATIC
    src/ChartIndexMapper.cpp
    src/ChartSceneModel.cpp
    src/ChartViewWidget.cpp
    src/LodResolver.cpp
    src/Windowing.cpp
    rendering/src/GLChartRenderer.cpp
)

target_link_libraries(tradereview_chart
    PUBLIC
        tradereview_core
        tradereview_data
        Qt6::OpenGL
        Qt6::OpenGLWidgets
)
```

- [ ] **Step 6: Put ChartViewWidget in MainWindow**

Modify `native/app/src/MainWindow.cpp`:

```cpp
#include "tradereview/app/MainWindow.h"

#include <QMenuBar>
#include <QStatusBar>

#include "tradereview/chart/ChartViewWidget.h"

namespace tradereview::app {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("TradeReview Native");
    resize(1400, 950);
    menuBar()->addMenu("&File");
    setCentralWidget(new chart::ChartViewWidget(this));
    statusBar()->showMessage("Native OpenGL chart ready");
}

} // namespace tradereview::app
```

- [ ] **Step 7: Run build**

Run:

```powershell
cmake --build native-build --config Debug
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add native/chart native/app
git commit -m "增加C++ OpenGL图表画布"
```

## Task 10: Final M0-M1 Verification

**Files:**
- No code changes unless verification reveals issues.

- [ ] **Step 1: Run native verification**

Run:

```powershell
cmake -S native -B native-build -DCMAKE_BUILD_TYPE=Debug
cmake --build native-build --config Debug
ctest --test-dir native-build --output-on-failure -C Debug
```

Expected: configure/build/tests pass.

- [ ] **Step 2: Run Python regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: Python tests still pass. Native work must not regress the existing Python app.

- [ ] **Step 3: Check git state**

Run:

```powershell
git status --short --branch
```

Expected: clean except intentionally ignored build artifacts and any pre-existing untracked local files such as `project_code_structure.svg`.

- [ ] **Step 4: Commit verification notes if needed**

If no code changes are needed, do not create an empty commit. If verification required fixes, commit them:

```powershell
git add native
git commit -m "修复C++原生骨架验证问题"
```

## Self-Review Notes

Spec coverage in this plan:

- Native monorepo subdirectory: covered by Task 1.
- Module folders: covered by Task 1 and target file structure.
- Pure C++ time, period, LOD, window, mapper types: covered by Tasks 3, 5, and 6.
- Candle window model and generation behavior: covered by Tasks 4 and 7.
- Qt Widgets and QOpenGLWidget baseline: covered by Tasks 8 and 9.
- Tests and frequent commits: covered in every task.

Gaps intentionally deferred:

- DuckDB repository and C API binding.
- Async scheduler and worker pool.
- Real OpenGL candle/volume/indicator geometry.
- Drawing, replay, sync, workspace layouts, session restore.

These gaps are not omissions; they require separate plans after M0-M1 is stable.
