# TradeReview C++ 原生界面终端化设计

日期：2026-04-28

## 背景

C++/Qt/OpenGL 原生版本已经具备主窗口、主控制栏、图表工作区、图表工具栏、OpenGL K 线渲染、回放控制和数据加载错误提示。当前界面仍偏工程占位：按钮文本化、控件层级弱、深色主题分散在各 widget 的局部 stylesheet 中，主窗口中间区域打开后容易显得“黑”和未完成。

本轮目标是把 C++ 版本外观提升到“专业交易终端深色主题”，并一次性完善窗口布局、工具栏分组、图表区域、指标区、右侧信息区和底部状态栏。按钮功能范围暂不扩张，优先让已有能力看起来像一个完整复盘工作站。

## 已确认方向

用户确认选择“专业交易终端深色主题”，并确认采用完整布局方案：

- 顶部菜单栏和主工具栏。
- 左侧绘图工具栏。
- 中央主 K 线图表。
- MACD / RSI 等指标面板。
- 右侧会话与行情信息面板。
- 底部状态栏。

本轮设计稿通过本地 visual companion 展示并获得确认。设计稿只用于沟通，不作为仓库源码提交。

## 目标

1. 统一 C++ 原生版本的视觉主题，避免每个 widget 各自维护一份零散 stylesheet。
2. 主窗口打开后应呈现完整交易终端结构，而不是仅有按钮和黑色中央区域。
3. 主控制栏和图表工具栏保持高密度，但用分组、间距、选中态、禁用态提升可读性。
4. 图表区域具备清晰层次：图表背景、网格、坐标轴、时间轴、K 线、回放标记、加载态、指标面板。
5. 右侧信息面板先展示会话、数据集、可见范围、回放速度、当前 OHLC 等占位或已有状态，不引入新的业务计算。
6. 左侧绘图工具栏先承接现有绘图动作入口，不新增复杂绘图功能。
7. 保持现有数据加载、回放、同步、绘图、OpenGL 渲染逻辑不变。

## 非目标

- 不重写 `ChartViewWidget` 的 OpenGL 渲染管线。
- 不改变 DuckDB、Parquet、窗口加载、LOD、回放引擎的数据流。
- 不实现新的技术指标算法。
- 不增加真实交易、订单、账户或策略功能。
- 不把 Qt Widgets 改成 Qt Quick/QML。
- 不主动编译运行 C++ 版本；是否编译由用户手动执行或显式要求。

## 方案比较

### 方案 A：只做皮肤

集中 stylesheet，调整颜色、按钮状态、间距和菜单栏。风险最低，但主窗口结构仍然偏占位，不能解决“中间黑、整体不像成品”的问题。

### 方案 B：皮肤加轻度重排

在保持现有控件层级的基础上强化分组。效果比方案 A 明显，但缺少右侧信息区、左侧工具区和完整状态栏，仍不像完整交易工作站。

### 方案 C：完整终端化布局

在不改核心业务逻辑的前提下，引入完整视觉框架：左侧工具栏、中央图表、右侧信息面板、底部状态栏和统一主题。用户已选择该方案。该方案改动面更大，但边界清楚，适合作为 C++ 版本后续功能移植的外观基础。

## 推荐架构

继续使用 Qt Widgets 作为主界面框架，保留现有模块边界：

```text
native/
  app/
    include/tradereview/app/
      AppTheme.h                 # 新增：集中主题、尺寸、样式入口
      MainWindow.h
      MainControlsBar.h
      SideInfoPanelWidget.h       # 新增：右侧会话信息面板
      StatusStripWidget.h         # 新增：底部状态栏
    src/
      AppTheme.cpp
      MainWindow.cpp
      MainControlsBar.cpp
      SideInfoPanelWidget.cpp
      StatusStripWidget.cpp
  chart/
    include/tradereview/chart/
      ChartPanelWidget.h
      ChartToolbarWidget.h
      ChartWorkspaceWidget.h
      DrawingToolRailWidget.h     # 新增：左侧绘图工具栏
    src/
      ChartPanelWidget.cpp
      ChartToolbarWidget.cpp
      ChartWorkspaceWidget.cpp
      DrawingToolRailWidget.cpp
```

`AppTheme` 负责统一应用 stylesheet、颜色 token、控件尺寸和 objectName 约定。`MainWindow` 继续作为 composition root，负责装配主控制栏、图表工作区、右侧面板和状态栏。图表行为仍由 `ChartWorkspaceWidget`、`ChartPanelWidget`、`ChartToolbarWidget` 和 `ChartViewWidget` 承担。

## 组件设计

### AppTheme

提供统一主题入口：

- `apply(QApplication&)` 或 `apply(QWidget&)` 设置全局 stylesheet。
- 提供主色、背景色、边框色、文本色、涨跌色、选中态色等常量。
- 定义常用尺寸：主工具栏高度、图表工具栏高度、左侧工具栏宽度、右侧面板宽度、状态栏高度、按钮高度。
- 给关键 widget 使用稳定 `objectName`，避免 stylesheet 选择器过宽。

### MainWindow

主窗口布局调整为：

```text
QMainWindow
  menuBar
  centralWidget
    vertical root
      MainControlsBar
      horizontal workspace
        DrawingToolRailWidget
        ChartWorkspaceWidget
        SideInfoPanelWidget
      StatusStripWidget
```

`MainWindow` 负责把已有的加载状态、回放状态、图表数量、布局模式和错误提示传给对应 UI 面板。它不直接绘制 K 线，也不新增数据计算。

### MainControlsBar

保留现有按钮和回调：

- Load Data
- Reset View
- Save View
- Layout
- Pop Layout
- Charts
- Replay Mode
- Back / Play / Forward
- Step
- Speed
- Date Time Jump

改进点：

- 使用视觉分组：数据、布局、回放、时间跳转。
- Play / Load 等关键动作使用主色。
- Reset / Save / Back / Forward 等动作可使用轻量图标或短文本。
- 禁用态和加载态更明显。
- 控件高度、边距、圆角统一。

### ChartToolbarWidget

保留周期、指标和绘图动作入口：

- 周期按钮维持当前功能。
- EMA / BB / MACD / RSI 等指标按钮保持 toggle 表现。
- 绘图动作按钮集中成一组，未来可逐步迁移到左侧工具栏。

改进点：

- 当前周期使用主色选中态。
- 指标按钮使用 checked 态，不再只依赖文字状态。
- 工具栏背景与主控制栏区分一层，避免视觉粘连。

### DrawingToolRailWidget

新增左侧竖向工具栏，作为绘图入口的视觉容器：

- 首版只连接现有绘图 action，不新增绘图能力。
- 按钮采用固定宽度和固定高度，避免布局跳动。
- 未实现或暂未连接的按钮应禁用或显示占位，不产生误导操作。

### SideInfoPanelWidget

新增右侧信息面板，首版展示已有或可安全推导的状态：

- 数据集名称或文件名。
- 当前图表周期。
- 可见时间范围。
- 图表数量和布局模式。
- 回放模式、播放状态、速度。
- 当前十字光标或当前 K 线 OHLC；如果数据不可用，则显示占位。

不在该面板中新增复杂统计。后续可以扩展交易统计、标注列表、会话备注。

### StatusStripWidget

新增底部状态栏，展示低频全局状态：

- Ready / Loading / Error。
- DuckDB 是否启用。
- 当前数据范围。
- OpenGL 渲染状态。
- 最近一次错误或提示的短消息。

状态栏只展示摘要，详细错误继续使用 `ErrorPresenter` 的对话框。

### ChartViewWidget 和渲染层

保留现有 OpenGL 管线。视觉完善集中在已有渲染参数和 overlay 表现：

- 图表背景色与主窗口主题一致。
- 网格线、坐标轴、时间轴、十字线颜色进入统一主题。
- 加载态使用半透明 overlay，而不是黑屏或静默。
- 数据为空时显示专业的空状态文字。
- 最右侧 K 线预留一点视觉边距，避免贴边。

## 数据和状态流

本轮不新增数据源。状态流按现有路径传递：

```text
DataLoadController / ReplaySession / ChartWorkspaceWidget
  -> MainWindow
  -> MainControlsBar / SideInfoPanelWidget / StatusStripWidget
```

`SideInfoPanelWidget` 和 `StatusStripWidget` 只消费状态，不反向修改业务状态。用户操作仍从现有按钮、图表工具栏和图表交互进入原有 callback。

为了避免 UI 面板读取内部对象过深，推荐在 `MainWindow` 中构造轻量 view model：

```text
UiSessionSnapshot
  dataset_name
  data_range
  visible_range
  active_period
  chart_count
  layout_mode
  replay_enabled
  replay_playing
  replay_speed
  status_text
```

首版可先用简单 setter 更新面板，后续再抽结构体。

## 错误、加载和空状态

- DuckDB 未启用或打开失败时，仍使用 `ErrorPresenter` 弹窗。
- 主界面不应只停留黑屏；图表区域显示空状态或错误摘要。
- 加载中时主控制栏禁用高风险动作，图表区域显示 loading overlay，状态栏显示 Loading。
- 加载失败后保留上一份有效图表数据；若没有旧数据，显示空状态。
- 右侧信息面板在无数据时显示 `No dataset loaded`、`--` 等明确占位。

## 样式约定

首版采用克制的深色交易终端配色：

```text
Window background      #0d1118
Panel background       #131a23
Toolbar background     #18202b
Button background      #222d3b
Border                 #2a3545 / #354256
Primary accent         #0f766e / #13a39b
Text primary           #d7dde6
Text secondary         #8fa0b8
Up candle              #22c55e
Down candle            #ef4444
Warning                #f59e0b
```

控件圆角控制在 6px 左右，保持专业工具感，不做过度圆润或营销风格。字号不随窗口宽度缩放，避免不同分辨率下文本不稳定。

## 测试和验收

本轮是 UI 外观与布局重构，自动化验证重点是确保已有功能不退化：

- 现有 C++ 单元测试应继续通过。
- 主窗口能创建并显示新的布局容器。
- 主控制栏现有回调仍被触发。
- 图表工具栏周期、指标、绘图 action 仍能到达现有逻辑。
- 数据加载失败时仍显示 `ErrorPresenter` 弹窗，同时主界面显示合理状态。
- 多图布局、回放状态、时间跳转和图表重载入口不因布局调整失效。

手动验收建议：

- 无数据启动时界面完整，不出现大块无说明黑屏。
- DuckDB OFF 构建加载数据时，弹窗和状态栏提示清楚。
- 加载数据后主图、指标区、右侧信息区和状态栏同步更新。
- 切换周期、图表数量、布局模式后按钮状态和图表区域稳定。
- 窗口缩放时文字不重叠，按钮不挤出父容器。

## 实施边界

实现阶段应按小步提交：

1. 新增 `AppTheme` 并移除主控制栏、图表工具栏中的重复 stylesheet。
2. 重整 `MainWindow` 布局，加入右侧信息面板和底部状态栏。
3. 新增左侧绘图工具栏，并连接到现有绘图 action。
4. 优化 `MainControlsBar` 和 `ChartToolbarWidget` 的分组、状态和尺寸。
5. 优化图表空状态、加载态、坐标轴/网格/边距等视觉细节。
6. 补充或调整测试，确认现有行为未退化。

每一步都应保持可回滚、可单独验证。实现中如发现现有类职责阻碍布局调整，应优先做局部提取，不做跨模块大重构。
