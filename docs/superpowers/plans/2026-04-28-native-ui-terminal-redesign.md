# Native UI Terminal Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the C++/Qt/OpenGL UI into a polished professional dark trading terminal without changing the data, replay, drawing, or OpenGL business logic.

**Architecture:** Keep `MainWindow` as the composition root and add focused widgets for theme, side information, bottom status, and drawing rail. Move duplicated local stylesheets into `AppTheme`, then wire existing status and callbacks through the new shell without expanding feature scope.

**Tech Stack:** C++20, Qt 6 Widgets, QOpenGLWidget, CMake, existing `tradereview_app_logic` / `tradereview_chart` libraries.

---

## Constraints

- Git commit messages must be Chinese.
- Do not proactively compile or run the C++ native version. Only run MSVC/CMake/native executable commands after the user explicitly asks.
- Existing untracked build and QtCreator directories are local artifacts and must not be committed.
- Preserve existing callbacks and data flow. This is a UI appearance and layout pass.

## File Structure

- Create `native/app/include/tradereview/app/AppTheme.h`
  - Centralizes theme colors, dimensions, object names, and stylesheet application.
- Create `native/app/src/AppTheme.cpp`
  - Provides one application stylesheet for Qt Widgets and helper functions for semantic colors.
- Modify `native/app/CMakeLists.txt`
  - Adds `AppTheme`, `SideInfoPanelWidget`, and `StatusStripWidget` to the executable target.
- Modify `native/app/src/NativeApp.cpp`
  - Applies the theme once after constructing `QApplication`.
- Modify `native/app/include/tradereview/app/MainWindow.h`
  - Stores pointers to the main controls, chart workspace, side panel, and status strip.
- Modify `native/app/src/MainWindow.cpp`
  - Replaces the simple vertical central layout with the terminal shell.
  - Updates status helpers so the native Qt status bar and the custom status strip stay consistent.
- Modify `native/app/include/tradereview/app/MainControlsBar.h`
  - Adds minimal public setters only if the implementation needs explicit loading/status state.
- Modify `native/app/src/MainControlsBar.cpp`
  - Removes local stylesheet and rebuilds the same controls in grouped sections.
- Create `native/app/include/tradereview/app/SideInfoPanelWidget.h`
  - Displays dataset, visible range, layout, chart count, replay state, speed, and last OHLC placeholders.
- Create `native/app/src/SideInfoPanelWidget.cpp`
  - Implements setter methods and a compact right-side panel.
- Create `native/app/include/tradereview/app/StatusStripWidget.h`
  - Displays global status, DuckDB availability, data range, renderer, and last message.
- Create `native/app/src/StatusStripWidget.cpp`
  - Implements the bottom status strip.
- Create `native/chart/include/tradereview/chart/DrawingToolRailWidget.h`
  - Provides a vertical drawing tool rail with callbacks for existing drawing actions.
- Create `native/chart/src/DrawingToolRailWidget.cpp`
  - Implements fixed-size drawing buttons and emits existing action strings.
- Modify `native/chart/CMakeLists.txt`
  - Adds `DrawingToolRailWidget`.
- Modify `native/chart/include/tradereview/chart/ChartPanelWidget.h`
  - Exposes a `trigger_drawing_action(const QString&)` method so the rail can call existing behavior.
- Modify `native/chart/src/ChartPanelWidget.cpp`
  - Reuses `handle_drawing_action` for external rail actions.
- Modify `native/chart/include/tradereview/chart/ChartWorkspaceWidget.h`
  - Adds a `triggerDrawingAction(const QString&)` helper for the active chart.
- Modify `native/chart/src/ChartWorkspaceWidget.cpp`
  - Routes left rail actions to the active chart panel.
- Modify `native/chart/src/ChartToolbarWidget.cpp`
  - Removes local stylesheet and improves grouping/object names while preserving callbacks.
- Modify `native/chart/src/ChartViewWidget.cpp`
  - Adds themed empty/loading overlay and a small right padding constant for the visual edge.

## Task 1: Central Theme

**Files:**
- Create: `native/app/include/tradereview/app/AppTheme.h`
- Create: `native/app/src/AppTheme.cpp`
- Modify: `native/app/CMakeLists.txt`
- Modify: `native/app/src/NativeApp.cpp`

- [ ] **Step 1: Create `AppTheme.h`**

Add this public interface:

```cpp
#pragma once

class QApplication;
class QColor;
class QString;

namespace tradereview::app {

namespace theme {

struct Size {
    static constexpr int MainToolbarHeight = 38;
    static constexpr int ChartToolbarHeight = 36;
    static constexpr int ControlHeight = 28;
    static constexpr int ToolRailWidth = 42;
    static constexpr int SidePanelWidth = 260;
    static constexpr int StatusStripHeight = 30;
};

void apply(QApplication& app);
[[nodiscard]] QColor chartBackground();
[[nodiscard]] QColor chartEmptyText();
[[nodiscard]] QColor loadingOverlay();
[[nodiscard]] QString styleSheet();

} // namespace theme

} // namespace tradereview::app
```

- [ ] **Step 2: Create `AppTheme.cpp`**

Use one stylesheet source for the app. Include object names used by the new widgets and existing controls:

```cpp
#include "tradereview/app/AppTheme.h"

#include <QApplication>
#include <QColor>
#include <QString>

namespace tradereview::app::theme {

QColor chartBackground()
{
    return QColor(11, 16, 23);
}

QColor chartEmptyText()
{
    return QColor(143, 160, 184);
}

QColor loadingOverlay()
{
    return QColor(15, 18, 24, 132);
}

QString styleSheet()
{
    return R"(
        QMainWindow {
            background-color: #0d1118;
            color: #d7dde6;
        }
        QMenuBar {
            background-color: #151b24;
            color: #d7dde6;
            border-bottom: 1px solid #2a3545;
            padding: 3px 8px;
        }
        QMenuBar::item:selected {
            background-color: #222d3b;
        }
        QStatusBar {
            background-color: #101720;
            color: #8fa0b8;
            border-top: 1px solid #2a3545;
        }
        #MainControlsBar,
        #ChartToolbarWidget,
        #SideInfoPanelWidget,
        #StatusStripWidget,
        #DrawingToolRailWidget {
            background-color: #131a23;
            color: #d7dde6;
        }
        #MainControlsBar {
            border-bottom: 1px solid #2a3545;
        }
        #ChartToolbarWidget {
            background-color: #151d27;
            border-bottom: 1px solid #2a3545;
        }
        #SideInfoPanelWidget {
            border-left: 1px solid #2a3545;
        }
        #StatusStripWidget {
            border-top: 1px solid #2a3545;
        }
        #DrawingToolRailWidget {
            border-right: 1px solid #2a3545;
        }
        QLabel,
        QCheckBox {
            color: #cbd5e1;
        }
        QPushButton,
        QComboBox,
        QDateTimeEdit {
            min-height: 24px;
            background-color: #222d3b;
            border: 1px solid #354256;
            border-radius: 6px;
            color: #d7dde6;
            padding: 3px 8px;
        }
        QPushButton:hover,
        QComboBox:hover,
        QDateTimeEdit:hover {
            background-color: #293548;
            border-color: #42516a;
            color: #f8fafc;
        }
        QPushButton:checked,
        QPushButton[primary="true"] {
            background-color: #0f766e;
            border-color: #13a39b;
            color: #f8fafc;
        }
        QPushButton:disabled,
        QComboBox:disabled,
        QDateTimeEdit:disabled {
            background-color: #17202b;
            border-color: #273142;
            color: #58677d;
        }
        QTabWidget::pane {
            border: 1px solid #2a3545;
        }
        QTabBar::tab {
            background-color: #18202b;
            color: #8fa0b8;
            padding: 6px 12px;
            border: 1px solid #2a3545;
        }
        QTabBar::tab:selected {
            background-color: #222d3b;
            color: #d7dde6;
        }
    )";
}

void apply(QApplication& app)
{
    app.setStyleSheet(styleSheet());
}

} // namespace tradereview::app::theme
```

- [ ] **Step 3: Register files in CMake**

Add `include/tradereview/app/AppTheme.h` and `src/AppTheme.cpp` to the `tradereview_native` executable source list in `native/app/CMakeLists.txt`.

- [ ] **Step 4: Apply theme in `NativeApp.cpp`**

Insert the include and call:

```cpp
#include "tradereview/app/AppTheme.h"

// after QApplication app(argc, argv);
theme::apply(app);
```

- [ ] **Step 5: Static verification**

Run:

```powershell
git diff -- native/app/include/tradereview/app/AppTheme.h native/app/src/AppTheme.cpp native/app/CMakeLists.txt native/app/src/NativeApp.cpp
```

Expected: only theme files, CMake registration, and the `NativeApp.cpp` apply call changed.

- [ ] **Step 6: Commit**

```powershell
git add native/app/include/tradereview/app/AppTheme.h native/app/src/AppTheme.cpp native/app/CMakeLists.txt native/app/src/NativeApp.cpp
git commit -m "集中C++界面主题样式"
```

## Task 2: Main Window Shell, Side Panel, Status Strip

**Files:**
- Create: `native/app/include/tradereview/app/SideInfoPanelWidget.h`
- Create: `native/app/src/SideInfoPanelWidget.cpp`
- Create: `native/app/include/tradereview/app/StatusStripWidget.h`
- Create: `native/app/src/StatusStripWidget.cpp`
- Modify: `native/app/include/tradereview/app/MainWindow.h`
- Modify: `native/app/src/MainWindow.cpp`
- Modify: `native/app/CMakeLists.txt`

- [ ] **Step 1: Create `SideInfoPanelWidget.h`**

```cpp
#pragma once

#include <QWidget>

class QLabel;
class QString;

namespace tradereview::app {

class SideInfoPanelWidget final : public QWidget {
public:
    explicit SideInfoPanelWidget(QWidget* parent = nullptr);

    void setDatasetName(const QString& name);
    void setDataRange(const QString& range);
    void setVisibleRange(const QString& range);
    void setLayoutSummary(const QString& summary);
    void setReplaySummary(const QString& summary);
    void setLastMessage(const QString& message);
    void resetDataset();

private:
    QLabel* dataset_value_ = nullptr;
    QLabel* data_range_value_ = nullptr;
    QLabel* visible_range_value_ = nullptr;
    QLabel* layout_value_ = nullptr;
    QLabel* replay_value_ = nullptr;
    QLabel* message_value_ = nullptr;
};

} // namespace tradereview::app
```

- [ ] **Step 2: Create `SideInfoPanelWidget.cpp`**

Use fixed width from `theme::Size::SidePanelWidth`, a section header, and six compact rows:

```cpp
#include "tradereview/app/SideInfoPanelWidget.h"

#include "tradereview/app/AppTheme.h"

#include <QFrame>
#include <QLabel>
#include <QVBoxLayout>

namespace tradereview::app {
namespace {

QLabel* addRow(QVBoxLayout& layout, QWidget& parent, const QString& label)
{
    auto* row = new QWidget(&parent);
    auto* row_layout = new QHBoxLayout(row);
    row_layout->setContentsMargins(0, 3, 0, 3);
    row_layout->setSpacing(8);

    auto* name = new QLabel(label, row);
    name->setStyleSheet("color: #8fa0b8;");
    auto* value = new QLabel("--", row);
    value->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    value->setWordWrap(true);

    row_layout->addWidget(name);
    row_layout->addWidget(value, 1);
    layout.addWidget(row);
    return value;
}

} // namespace

SideInfoPanelWidget::SideInfoPanelWidget(QWidget* parent)
    : QWidget(parent)
{
    setObjectName("SideInfoPanelWidget");
    setFixedWidth(theme::Size::SidePanelWidth);

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(12, 10, 12, 10);
    root->setSpacing(8);

    auto* title = new QLabel("Session", this);
    title->setStyleSheet("font-weight: 600; color: #d7dde6;");
    root->addWidget(title);

    dataset_value_ = addRow(*root, *this, "Dataset");
    data_range_value_ = addRow(*root, *this, "Data");
    visible_range_value_ = addRow(*root, *this, "Visible");
    layout_value_ = addRow(*root, *this, "Layout");
    replay_value_ = addRow(*root, *this, "Replay");
    message_value_ = addRow(*root, *this, "Message");
    root->addStretch();

    resetDataset();
}

void SideInfoPanelWidget::setDatasetName(const QString& name) { dataset_value_->setText(name.isEmpty() ? "--" : name); }
void SideInfoPanelWidget::setDataRange(const QString& range) { data_range_value_->setText(range.isEmpty() ? "--" : range); }
void SideInfoPanelWidget::setVisibleRange(const QString& range) { visible_range_value_->setText(range.isEmpty() ? "--" : range); }
void SideInfoPanelWidget::setLayoutSummary(const QString& summary) { layout_value_->setText(summary.isEmpty() ? "--" : summary); }
void SideInfoPanelWidget::setReplaySummary(const QString& summary) { replay_value_->setText(summary.isEmpty() ? "--" : summary); }
void SideInfoPanelWidget::setLastMessage(const QString& message) { message_value_->setText(message.isEmpty() ? "--" : message); }

void SideInfoPanelWidget::resetDataset()
{
    setDatasetName("No dataset loaded");
    setDataRange("--");
    setVisibleRange("--");
    setLayoutSummary("4 charts / Tabs");
    setReplaySummary("Disabled");
    setLastMessage("Ready");
}

} // namespace tradereview::app
```

- [ ] **Step 3: Create `StatusStripWidget.h` and `.cpp`**

Header:

```cpp
#pragma once

#include <QWidget>

class QLabel;
class QString;

namespace tradereview::app {

class StatusStripWidget final : public QWidget {
public:
    explicit StatusStripWidget(QWidget* parent = nullptr);

    void setStateText(const QString& text);
    void setDuckDbText(const QString& text);
    void setDataRangeText(const QString& text);
    void setRendererText(const QString& text);
    void setMessageText(const QString& text);

private:
    QLabel* state_label_ = nullptr;
    QLabel* duckdb_label_ = nullptr;
    QLabel* data_range_label_ = nullptr;
    QLabel* renderer_label_ = nullptr;
    QLabel* message_label_ = nullptr;
};

} // namespace tradereview::app
```

Implementation:

```cpp
#include "tradereview/app/StatusStripWidget.h"

#include "tradereview/app/AppTheme.h"

#include <QHBoxLayout>
#include <QLabel>

namespace tradereview::app {

StatusStripWidget::StatusStripWidget(QWidget* parent)
    : QWidget(parent)
{
    setObjectName("StatusStripWidget");
    setFixedHeight(theme::Size::StatusStripHeight);

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(12, 0, 12, 0);
    layout->setSpacing(16);

    state_label_ = new QLabel("Ready", this);
    duckdb_label_ = new QLabel("DuckDB: build setting", this);
    data_range_label_ = new QLabel("Range: --", this);
    message_label_ = new QLabel("Native OpenGL chart ready", this);
    renderer_label_ = new QLabel("Renderer: OpenGL", this);

    layout->addWidget(state_label_);
    layout->addWidget(duckdb_label_);
    layout->addWidget(data_range_label_);
    layout->addWidget(message_label_, 1);
    layout->addWidget(renderer_label_);
}

void StatusStripWidget::setStateText(const QString& text) { state_label_->setText(text); }
void StatusStripWidget::setDuckDbText(const QString& text) { duckdb_label_->setText(text); }
void StatusStripWidget::setDataRangeText(const QString& text) { data_range_label_->setText(text); }
void StatusStripWidget::setRendererText(const QString& text) { renderer_label_->setText(text); }
void StatusStripWidget::setMessageText(const QString& text) { message_label_->setText(text); }

} // namespace tradereview::app
```

- [ ] **Step 4: Register new files in `native/app/CMakeLists.txt`**

Add the two headers and two source files to `tradereview_native`.

- [ ] **Step 5: Modify `MainWindow.h`**

Forward declare widgets and store pointers:

```cpp
namespace tradereview::chart {
class ChartWorkspaceWidget;
}

namespace tradereview::app {

class MainControlsBar;
class SideInfoPanelWidget;
class StatusStripWidget;

class MainWindow final : public QMainWindow {
public:
    explicit MainWindow(QWidget* parent = nullptr);

private:
    void showStatusMessage(const QString& message);
    void setLoadingStatus(const QString& message);
    void setReadyStatus(const QString& message);

    QSettings settings_;
    std::unique_ptr<DataLoadController> data_load_controller_;
    MainControlsBar* main_controls_ = nullptr;
    chart::ChartWorkspaceWidget* chart_workspace_ = nullptr;
    SideInfoPanelWidget* side_info_panel_ = nullptr;
    StatusStripWidget* status_strip_ = nullptr;
};

} // namespace tradereview::app
```

- [ ] **Step 6: Modify `MainWindow.cpp` layout**

Replace local `mainControls` and `chartWorkspace` variables with member pointers. Build this shell:

```cpp
auto* central = new QWidget(this);
auto* root = new QVBoxLayout(central);
root->setContentsMargins(0, 0, 0, 0);
root->setSpacing(0);

main_controls_ = new MainControlsBar(central);
chart_workspace_ = new chart::ChartWorkspaceWidget(central);
side_info_panel_ = new SideInfoPanelWidget(central);
status_strip_ = new StatusStripWidget(central);

auto* workspace = new QWidget(central);
auto* workspace_layout = new QHBoxLayout(workspace);
workspace_layout->setContentsMargins(0, 0, 0, 0);
workspace_layout->setSpacing(0);
workspace_layout->addWidget(chart_workspace_, 1);
workspace_layout->addWidget(side_info_panel_);

root->addWidget(main_controls_);
root->addWidget(workspace, 1);
root->addWidget(status_strip_);
setCentralWidget(central);
```

Keep the existing callback bodies, replacing `mainControls` with `main_controls_` and `chartWorkspace` with `chart_workspace_`.

- [ ] **Step 7: Add status helper methods**

Append these implementations:

```cpp
void MainWindow::showStatusMessage(const QString& message)
{
    statusBar()->showMessage(message);
    if (status_strip_ != nullptr) {
        status_strip_->setMessageText(message);
    }
    if (side_info_panel_ != nullptr) {
        side_info_panel_->setLastMessage(message);
    }
}

void MainWindow::setLoadingStatus(const QString& message)
{
    if (status_strip_ != nullptr) {
        status_strip_->setStateText("Loading");
    }
    showStatusMessage(message);
}

void MainWindow::setReadyStatus(const QString& message)
{
    if (status_strip_ != nullptr) {
        status_strip_->setStateText("Ready");
    }
    showStatusMessage(message);
}
```

Replace direct `statusBar()->showMessage(...)` calls with `showStatusMessage(...)`, `setLoadingStatus(...)`, or `setReadyStatus(...)`.

- [ ] **Step 8: Update panel state from existing events**

In load success callbacks, set:

```cpp
side_info_panel_->setDatasetName(QFileInfo(path).fileName());
side_info_panel_->setDataRange(formatTimestamp(result.dataset_info.tick_range.start_ns) + " to " + formatTimestamp(result.dataset_info.tick_range.end_ns));
side_info_panel_->setVisibleRange(formatTimestamp(result.window.visible_range.start_ns) + " to " + formatTimestamp(result.window.visible_range.end_ns));
status_strip_->setDataRangeText("Range: " + formatTimestamp(result.dataset_info.tick_range.start_ns) + " to " + formatTimestamp(result.dataset_info.tick_range.end_ns));
```

In layout/chart-count callbacks, set:

```cpp
side_info_panel_->setLayoutSummary(QString("%1 charts / %2").arg(static_cast<int>(chart_workspace_->chart_count())).arg(layoutModeText(chart_workspace_->layout_mode())));
```

In replay callbacks, set:

```cpp
side_info_panel_->setReplaySummary(enabled ? "Enabled" : "Disabled");
```

For speed:

```cpp
side_info_panel_->setReplaySummary(QString("Enabled / %1x").arg(speed));
```

- [ ] **Step 9: Static verification**

Run:

```powershell
rg -n "statusBar\\(\\)->showMessage" native\\app\\src\\MainWindow.cpp
git diff -- native/app
```

Expected: no direct `statusBar()->showMessage` calls remain outside `showStatusMessage`, and only app UI files changed.

- [ ] **Step 10: Commit**

```powershell
git add native/app
git commit -m "完善C++主窗口终端布局"
```

## Task 3: Drawing Tool Rail

**Files:**
- Create: `native/chart/include/tradereview/chart/DrawingToolRailWidget.h`
- Create: `native/chart/src/DrawingToolRailWidget.cpp`
- Modify: `native/chart/CMakeLists.txt`
- Modify: `native/chart/include/tradereview/chart/ChartPanelWidget.h`
- Modify: `native/chart/src/ChartPanelWidget.cpp`
- Modify: `native/chart/include/tradereview/chart/ChartWorkspaceWidget.h`
- Modify: `native/chart/src/ChartWorkspaceWidget.cpp`
- Modify: `native/app/src/MainWindow.cpp`

- [ ] **Step 1: Create `DrawingToolRailWidget.h`**

```cpp
#pragma once

#include <QWidget>

#include <functional>

class QString;

namespace tradereview::chart {

class DrawingToolRailWidget final : public QWidget {
public:
    using DrawingActionCallback = std::function<void(const QString&)>;

    explicit DrawingToolRailWidget(QWidget* parent = nullptr);

    void setDrawingActionCallback(DrawingActionCallback callback);

private:
    void emitAction(const QString& action) const;

    DrawingActionCallback drawing_action_callback_;
};

} // namespace tradereview::chart
```

- [ ] **Step 2: Create `DrawingToolRailWidget.cpp`**

Use only existing wired action names:

```cpp
#include "tradereview/chart/DrawingToolRailWidget.h"

#include "tradereview/app/AppTheme.h"

#include <QPushButton>
#include <QStringList>
#include <QVBoxLayout>

#include <utility>

namespace tradereview::chart {
namespace {

QPushButton* addRailButton(QVBoxLayout& layout, QWidget& parent, const QString& label, const QString& action)
{
    auto* button = new QPushButton(label, &parent);
    button->setFixedSize(30, 30);
    button->setToolTip(action);
    layout.addWidget(button);
    QObject::connect(button, &QPushButton::clicked, &parent, [&parent, action](bool) {
        auto* rail = qobject_cast<DrawingToolRailWidget*>(&parent);
        if (rail != nullptr) {
            rail->emitAction(action);
        }
    });
    return button;
}

} // namespace

DrawingToolRailWidget::DrawingToolRailWidget(QWidget* parent)
    : QWidget(parent)
{
    setObjectName("DrawingToolRailWidget");
    setFixedWidth(app::theme::Size::ToolRailWidth);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 8, 6, 8);
    layout->setSpacing(6);

    addRailButton(*layout, *this, "S", "Sel");
    addRailButton(*layout, *this, "H", "H");
    addRailButton(*layout, *this, "V", "V");
    addRailButton(*layout, *this, "/", "Line");
    addRailButton(*layout, *this, "F", "Fib");
    addRailButton(*layout, *this, "E", "Fib Ext");
    addRailButton(*layout, *this, "C", "Clear");
    layout->addStretch();
}

void DrawingToolRailWidget::setDrawingActionCallback(DrawingActionCallback callback)
{
    drawing_action_callback_ = std::move(callback);
}

void DrawingToolRailWidget::emitAction(const QString& action) const
{
    if (drawing_action_callback_) {
        drawing_action_callback_(action);
    }
}

} // namespace tradereview::chart
```

- [ ] **Step 3: Register rail in `native/chart/CMakeLists.txt`**

Add the header and source to `tradereview_chart`.

- [ ] **Step 4: Expose active chart drawing action**

In `ChartPanelWidget.h`, add public:

```cpp
void trigger_drawing_action(const QString& action);
```

In `ChartPanelWidget.cpp`, add:

```cpp
void ChartPanelWidget::trigger_drawing_action(const QString& action)
{
    handle_drawing_action(action);
}
```

In `ChartWorkspaceWidget.h`, add:

```cpp
bool triggerDrawingAction(const QString& action);
```

In `ChartWorkspaceWidget.cpp`, add:

```cpp
bool ChartWorkspaceWidget::triggerDrawingAction(const QString& action)
{
    auto* active_panel = panel(active_chart_id());
    if (active_panel == nullptr) {
        return false;
    }
    active_panel->trigger_drawing_action(action);
    return true;
}
```

- [ ] **Step 5: Add rail to `MainWindow.cpp` shell**

Include:

```cpp
#include "tradereview/chart/DrawingToolRailWidget.h"
```

Create the rail beside the workspace:

```cpp
auto* drawing_rail = new chart::DrawingToolRailWidget(workspace);
drawing_rail->setDrawingActionCallback([this](const QString& action) {
    if (chart_workspace_ != nullptr && chart_workspace_->triggerDrawingAction(action)) {
        showStatusMessage(QString("Drawing tool ") + action);
    }
});
workspace_layout->addWidget(drawing_rail);
workspace_layout->addWidget(chart_workspace_, 1);
workspace_layout->addWidget(side_info_panel_);
```

- [ ] **Step 6: Static verification**

Run:

```powershell
rg -n "triggerDrawingAction|DrawingToolRailWidget|trigger_drawing_action" native
git diff -- native/chart native/app/src/MainWindow.cpp
```

Expected: the rail routes actions through the existing chart drawing action path.

- [ ] **Step 7: Commit**

```powershell
git add native/chart native/app/src/MainWindow.cpp
git commit -m "添加C++绘图工具侧栏"
```

## Task 4: Main Controls and Chart Toolbar Polish

**Files:**
- Modify: `native/app/src/MainControlsBar.cpp`
- Modify: `native/chart/src/ChartToolbarWidget.cpp`

- [ ] **Step 1: Remove duplicated local stylesheets**

Delete `setStyleSheet(R"(...)")` blocks from both files. Keep object names:

```cpp
setObjectName("MainControlsBar");
setObjectName("ChartToolbarWidget");
```

- [ ] **Step 2: Add group helper to `MainControlsBar.cpp`**

Add a helper near `addButton`:

```cpp
QHBoxLayout* addGroup(QHBoxLayout* root, QWidget* parent)
{
    auto* group_widget = new QWidget(parent);
    group_widget->setObjectName("ToolbarGroup");
    auto* group_layout = new QHBoxLayout(group_widget);
    group_layout->setContentsMargins(0, 0, 10, 0);
    group_layout->setSpacing(5);
    root->addWidget(group_widget);
    return group_layout;
}
```

Then use groups for data actions, layout, replay, speed, and time. Preserve every existing `connect(...)` body.

- [ ] **Step 3: Mark primary buttons**

After creating Load Data and Play buttons:

```cpp
button->setProperty("primary", true);
button->style()->unpolish(button);
button->style()->polish(button);
```

For Play, refresh the property in `setReplayPlaying`:

```cpp
replay_play_button_->setProperty("primary", true);
replay_play_button_->setText(playing ? "Pause" : "Play");
replay_play_button_->style()->unpolish(replay_play_button_);
replay_play_button_->style()->polish(replay_play_button_);
```

- [ ] **Step 4: Improve chart toolbar grouping**

In `ChartToolbarWidget.cpp`, add small group widgets for periods, indicators, and drawing actions. Keep the period `QScrollArea` for dense periods and keep all existing `connect(...)` behavior.

Use this pattern:

```cpp
auto* indicator_widget = new QWidget(this);
auto* indicator_layout = new QHBoxLayout(indicator_widget);
indicator_layout->setContentsMargins(8, 0, 8, 0);
indicator_layout->setSpacing(4);
toolbarLayout->addWidget(indicator_widget);
```

Add indicator buttons to `indicator_layout` instead of directly to `toolbarLayout`.

- [ ] **Step 5: Static verification**

Run:

```powershell
rg -n "setStyleSheet" native\\app\\src\\MainControlsBar.cpp native\\chart\\src\\ChartToolbarWidget.cpp
rg -n "setLoadDataCallback|setReplayPlayCallback|setIndicatorToggleCallback|setPeriodSelectedCallback|setDrawingActionCallback" native\\app\\src\\MainControlsBar.cpp native\\chart\\src\\ChartToolbarWidget.cpp
```

Expected: no local `setStyleSheet` remains in these two widgets, and all existing callback setters still appear.

- [ ] **Step 6: Commit**

```powershell
git add native/app/src/MainControlsBar.cpp native/chart/src/ChartToolbarWidget.cpp
git commit -m "美化C++主控制栏和图表工具栏"
```

## Task 5: Chart Empty and Loading State

**Files:**
- Modify: `native/chart/src/ChartViewWidget.cpp`

- [ ] **Step 1: Include theme**

Add:

```cpp
#include "tradereview/app/AppTheme.h"
```

- [ ] **Step 2: Add overlay helper**

Inside the anonymous namespace:

```cpp
void drawCenteredOverlay(QOpenGLWidget& widget, const QString& text, QColor fill, QColor pen)
{
    QPainter painter(&widget);
    painter.fillRect(widget.rect(), fill);
    painter.setPen(pen);
    painter.drawText(widget.rect(), Qt::AlignCenter, text);
}
```

- [ ] **Step 3: Update `paintGL`**

Replace the current loading-only overlay with:

```cpp
void ChartViewWidget::paintGL()
{
    renderer_.render(scene_model_, drawing_state_.drawings(), drawing_state_.preview(), drawing_state_.revision());

    if (scene_model_.row_count() == 0 && !scene_model_.loading()) {
        drawCenteredOverlay(*this, "No dataset loaded", app::theme::chartBackground(), app::theme::chartEmptyText());
        return;
    }

    if (!scene_model_.loading()) {
        return;
    }

    drawCenteredOverlay(*this, "Loading...", app::theme::loadingOverlay(), QColor(235, 238, 245));
}
```

- [ ] **Step 4: Static verification**

Run:

```powershell
rg -n "No dataset loaded|Loading\\.\\.\\.|drawCenteredOverlay|chartBackground" native\\chart\\src\\ChartViewWidget.cpp
git diff -- native/chart/src/ChartViewWidget.cpp
```

Expected: empty and loading overlays are both visible in code.

- [ ] **Step 5: Commit**

```powershell
git add native/chart/src/ChartViewWidget.cpp
git commit -m "完善C++图表空状态和加载状态"
```

## Task 6: Manual Review Notes and Push

**Files:**
- Modify: `docs/native-manual-verification.md`

- [ ] **Step 1: Add UI terminal shell checklist**

Append this section:

```markdown
## C++ 终端化界面手动验收

- 无数据启动时，主窗口应显示顶部菜单、主控制栏、左侧绘图工具栏、中央图表空状态、右侧信息面板和底部状态栏。
- 点击未加载数据前的回放/绘图按钮时，不应崩溃；未接线动作应显示明确状态消息。
- DuckDB OFF 构建加载数据时，仍应弹出错误框，同时底部状态栏和右侧面板保留清晰状态。
- DuckDB ON 构建加载数据后，右侧面板应显示数据集名称、数据范围、可见范围、布局和回放状态。
- 切换图表数量和布局模式后，中央图表区、右侧面板和状态栏不应出现文字重叠。
- 切换周期、EMA、BB、MACD/RSI 后，原有图表刷新和指标开关行为应保持。
- 左侧 H/V/Line/Fib/Fib Ext/Clear 工具应走现有绘图 action，不新增未定义行为。
```

- [ ] **Step 2: Verify git scope**

Run:

```powershell
git status --short --branch
git log --oneline -8
```

Expected: only known untracked build folders remain, and the UI commits are at the top of the branch.

- [ ] **Step 3: Commit docs**

```powershell
git add docs/native-manual-verification.md
git commit -m "补充C++界面手动验收清单"
```

- [ ] **Step 4: Push branch**

Run:

```powershell
git push
```

Expected: `cpp-opengl-native-m0m1` updates on GitHub.

## Self-Review

- Spec coverage: theme, main window shell, left drawing rail, right info panel, bottom status strip, toolbar polish, chart empty/loading states, error/loading expectations, and manual review are all mapped to tasks.
- Placeholder scan: no unfinished placeholder markers are used in task steps.
- Type consistency: `SideInfoPanelWidget`, `StatusStripWidget`, `DrawingToolRailWidget`, `trigger_drawing_action`, and `triggerDrawingAction` use consistent names across headers, source files, and call sites.
- Scope: data loading, replay, indicators, drawing internals, DuckDB, and OpenGL rendering are preserved. The plan limits itself to UI composition, styling, and status presentation.
