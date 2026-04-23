# TradeReview UI 架构整理设计

日期：2026-04-24

## 背景

当前 `ui/main_window.py` 约 2145 行，包含：

- `MockYScale`
- `TimeAxisItem`
- `CandlestickItem`
- `ChartWidget`
- `FloatingChartWindow`
- `MainWindow`

其中 `ChartWidget` 本身已经承担单图表工具栏、PyQtGraph 初始化、K 线与指标渲染、鼠标交互、绘图、十字光标、可见切片刷新等职责；`MainWindow` 同时承担应用装配、数据加载、回放状态机、时间跳转、会话保存、多图布局、浮窗管理、跨图同步、绘图广播和 Fib 配置下发。

这种结构已经影响后续开发：新增复盘对象、工作区持久化、replay scrubber、绘图持久化时，都会继续挤进 `MainWindow`，使风险和测试成本持续上升。

## 子代理共识

本次并行审阅从三个角度完成：

- UI 组件拆分角度：建议先把图表底层、单图表组件、浮窗宿主迁出 `main_window.py`。
- 状态/控制器角度：建议 `MainWindow` 只作为 composition root，回放、数据加载、工作区布局逐步抽成 controller/service。
- 测试迁移角度：建议先补主流程行为测试，再迁移文件；不要同时改布局、回放和数据加载。

三方结论一致：第一阶段应以“纯搬家 + 兼容导出”为主，第二阶段再抽 controller/service。不要一上来重写算法、重写 UI 或引入大型框架。

## 目标

1. 降低 `ui/main_window.py` 的职责密度。
2. 让单图表、浮窗、主控条、工作区布局、回放控制各自有清晰模块边界。
3. 保持现有外部行为和测试通过。
4. 为后续 `WorkspaceState`、复盘对象、replay scrubber、标注持久化预留位置。
5. 让新增功能不再默认落进 `MainWindow`。

## 非目标

- 不重写 `DataEngine` 或 `ReplayEngine` 算法。
- 不重新设计视觉风格。
- 不做插件系统。
- 不为每个按钮创建独立类。
- 不引入通用 `BaseController`、MVP、MVVM 框架层。
- 不在本轮实现复盘对象、工作区持久化或 replay scrubber。

## 目标模块结构

```text
ui/
  main_window.py
  chart_primitives.py
  chart_widget.py
  chart_window.py
  main_controls.py
  chart_performance.py
  crosshair_sync.py
  session_state.py
  time_navigation.py
  controllers/
    __init__.py
    replay_controller.py
    workspace_layout_manager.py
  services/
    __init__.py
    data_loading.py
  drawings/
    __init__.py
    dialogs.py
    fib_config.py
    fib_math.py
    renderers.py
    specs.py
    tools.py
```

## 模块责任

### `ui/main_window.py`

保留为应用装配层，只负责：

- 创建 `DataEngine`、`ReplayEngine`、`QSettings`
- 创建主控条和图表工作区
- 连接顶层 signal/slot
- 展示文件选择、错误提示、warning dialog
- 调用 controller/service 完成业务编排

不再直接包含：

- K 线绘制 item
- 单图表 widget 的全部实现
- 浮窗宿主类
- replay 细节状态机
- layout/detach 细节算法

### `ui/chart_primitives.py`

负责底层图表 primitive：

- `MockYScale`
- `TimeAxisItem`
- `CandlestickItem`

这些对象只依赖 PyQtGraph、Qt painter 和 numpy，不依赖 `MainWindow`。

### `ui/chart_widget.py`

负责单图表组件：

- 周期按钮
- 指标开关
- BB / EMA / MACD / RSI 显示
- PyQtGraph plot 初始化
- 单图内十字光标
- 单图内绘图交互
- `update_chart()`
- 可见切片刷新
- `sig_period_changed`
- `sig_detach_requested`
- `sig_mouse_moved_with_price`
- 绘图相关信号

`ChartWidget` 可以继续使用 `ui.drawings.*`，但不负责保存全局绘图状态。

### `ui/chart_window.py`

负责浮窗宿主：

- `FloatingChartWindow`
- 接收一个 `ChartWidget`
- 显示独立窗口标题
- close 时发出 `sig_window_closed`

### `ui/main_controls.py`

负责主控制条：

- 加载数据
- 重置视图
- 保存视图
- 布局选择
- 图表数量
- replay mode
- play/pause
- step back / forward
- step size
- speed
- date edit

它只暴露语义信号，例如：

- `load_requested`
- `reset_requested`
- `save_view_requested`
- `layout_changed`
- `chart_count_changed`
- `replay_mode_changed`
- `play_toggled`
- `step_back_requested`
- `step_forward_requested`
- `speed_changed`
- `date_edit_finished`

`MainWindow` 仍决定这些信号触发什么业务动作。

### `ui/controllers/replay_controller.py`

负责 UI 层回放编排，不替代 `engine.replay_engine.ReplayEngine`：

- replay 是否启用
- 是否播放中
- replay speed
- 当前时间
- 初始化 replay engine
- reset
- advance
- 读取当前 period 的 replay view

它的目标是让 `MainWindow` 不再直接读写 `ReplayEngine.states`、`tick_pos`、`max_count_map` 等细节。

### `ui/controllers/workspace_layout_manager.py`

负责多图工作区：

- chart count
- enabled charts
- attached charts
- layout charts
- tabs / grid / splitter 布局应用
- attach / detach
- floating windows 列表
- crosshair 目标刷新

这个模块依赖 Qt widget 树，迁移风险较高，应放在纯搬家完成之后。

### `ui/services/data_loading.py`

负责数据加载结果结构化：

- 调用 `DataEngine.load_data()`
- 返回 `DataLoadResult`
- 封装 `error`
- 封装 `warnings`
- 封装 `initial_time`

目标是消除 `MainWindow.load_data_file()` 里对 `df_ticks`、`last_load_error`、`last_load_warnings` 的隐式组合判断。

### 现有模块保留

- `ui/drawings/*` 当前边界合理，不在本轮继续拆。
- `ui/session_state.py` 保留为持久化模型和读写函数，后续可扩展为 `WorkspaceState`。
- `ui/time_navigation.py` 保留并逐步加强为时间导航纯逻辑模块。
- `ui/crosshair_sync.py` 保留为十字同步唯一注册和广播入口。
- `ui/chart_performance.py` 应只保留可见切片相关函数；其中与生产路径无关的 crosshair helper 应删除或迁到真实调用路径。

## 迁移策略

### Phase 1: 纯搬家和兼容导出

目标：不改行为，只降低文件体积。

步骤：

1. 新增测试锁定模块边界。
2. 将 `MockYScale`、`TimeAxisItem`、`CandlestickItem` 移到 `ui/chart_primitives.py`。
3. 将 `FloatingChartWindow` 移到 `ui/chart_window.py`。
4. 将 `ChartWidget` 移到 `ui/chart_widget.py`。
5. `ui/main_window.py` 继续兼容导出 `ChartWidget` 和 `FloatingChartWindow`，避免旧测试和旧导入立即失效。
6. 现有测试全部通过后，再更新测试 import 到新模块。

### Phase 2: 主控制条组件化

目标：把 `create_control_panel()` 从 `MainWindow` 拆出。

步骤：

1. 新建 `MainControls`。
2. 保持控件文案、默认值、速度选项、step 选项不变。
3. 通过信号把动作交还给 `MainWindow`。
4. `MainWindow` 只保留业务响应方法。

### Phase 3: 回放和时间导航 controller 化

目标：把回放状态机从 `MainWindow` 中抽离。

步骤：

1. 新建 `ReplayController`。
2. 先包住现有 `ReplayEngine`，不改 replay 算法。
3. 保留 `MainWindow.jump_to_time()`、`on_timer_tick()` 等旧方法名作为转发层。
4. 补 `tests/test_main_window_time_navigation.py`。

### Phase 4: 工作区布局 manager 化

目标：把 layout/detach/floating window 从 `MainWindow` 中抽离。

步骤：

1. 新建 `WorkspaceLayoutManager`。
2. 先迁移 chart count、enabled chart、attached chart 计算。
3. 再迁移 `switch_layout()`。
4. 最后迁移 attach/detach/floating window。
5. 补 `tests/test_main_window_layout_and_crosshair.py`。

### Phase 5: 数据加载 facade 化

目标：让数据加载返回结构化结果，减少隐式状态读取。

步骤：

1. 新建 `DataLoadResult`。
2. 新建 `DataLoadingFacade`。
3. `MainWindow.load_data_file()` 保留 dialog 和 UI 刷新，只消费结果对象。
4. 不改初始时间选择规则。

## 测试策略

已有测试保留：

- `tests/test_session_state.py`
- `tests/test_time_navigation.py`
- `tests/test_crosshair_sync.py`
- `tests/test_chart_performance.py`
- `tests/test_drawing_chart_widget.py`
- `tests/test_main_window_crosshair_sync.py`

建议新增：

- `tests/test_ui_module_boundaries.py`
- `tests/test_chart_widget.py`
- `tests/test_main_window_time_navigation.py`
- `tests/test_main_window_persistence.py`
- `tests/test_main_window_layout_and_crosshair.py`
- `tests/test_replay_controller.py`
- `tests/test_data_loading_facade.py`

验收命令：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\.venv\Scripts\python.exe -m pytest -q
```

## 风险控制

- 不同时改 `jump_to_time()` 和 `load_data_file()`。
- 不同时改 crosshair 注册和 detach/layout。
- 不同时拆 `ChartWidget` 和改 replay 逻辑。
- 不先大规模改测试文件名。
- 每个阶段结束都跑全量测试。
- 每个阶段单独提交，提交信息使用中文。

## 设计决策

推荐方案：先文件级拆分，再 controller/service 化。

理由：

- 纯搬家风险最低，收益直接。
- 兼容导出能降低一次性迁移成本。
- `MainWindow` 先变小，再变薄，比直接抽象 controller 更稳。
- 当前测试对小模块保护较好，但对主流程保护偏薄，必须渐进。

明确不采用：

- 一次性重写 `MainWindow`
- 引入 UI 框架模式
- 插件化指标/绘图/replay
- 把每个按钮拆成独立类
- 在重构期改变数据语义、时区规则或 replay 行为

