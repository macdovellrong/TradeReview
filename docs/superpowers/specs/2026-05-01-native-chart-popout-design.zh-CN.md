# C++ 原生图表分离窗口设计

## 目标

支持把单个图表从主工作区弹出为独立窗口，用于多屏查看。第一版只实现“弹出”和“关闭后自动回到主工作区”，不做停靠拖回、窗口位置记忆或启动恢复。

## 交互

- 每个图表顶部工具栏增加 `Pop` 按钮。
- 点击 `Pop` 后，对应图表从主工作区布局中移除，并显示为独立顶层窗口。
- 浮动窗口标题显示图表编号和当前周期。
- 关闭浮动窗口时，同一个图表控件回到主工作区。
- 已弹出的图表不在主工作区布局中显示，但仍使用原来的 `chart_id`。

## 架构

- 新增 `FloatingChartWindow`，作为独立 `QMainWindow` 承载一个 `ChartPanelWidget`。
- `ChartWorkspaceWidget` 负责创建、记录和回收浮动窗口。
- `ChartWorkspaceState` 记录 detached chart id，并提供主工作区可见图表列表。
- `ChartToolbarWidget` 增加 pop-out 回调，`ChartPanelWidget` 把回调转发给 workspace。

## 数据流

弹出不会复制图表对象。`ChartPanelWidget` 原对象被迁移到浮动窗口，因此周期、指标、画线、十字线和当前视图状态都会保留。数据请求仍按原 `chart_id` 进入 `DataLoadController`，加载结果仍由 `ChartWorkspaceWidget::apply_window()` 分发给同一个 panel。

## 边界

- `enabled_chart_ids()` 仍返回启用图表，包括已弹出的图表，保证它们继续参与加载和同步。
- 主工作区布局只使用 `visible_chart_ids()`，跳过已弹出的图表。
- 如果降低图表数量导致某个弹出图表不再启用，应先回收该浮动窗口。
- 关闭应用时，浮动窗口由 workspace 统一清理，避免 panel 被窗口误删或泄漏。

## 验证

- 为 `ChartWorkspaceState` 增加 detached/visible id 测试。
- 静态检查 `git diff --check` 和 `git diff --cached --check`。
- 按项目约定，不主动编译或运行 C++ 原生版本。
