# TradeReview 标题与跨窗口十字同步设计

日期：2026-04-19  
项目：TradeReview  
范围：修改主程序窗口标题，并把十字光标同步从布局相关逻辑中抽离为独立同步子系统，确保主界面图表与 `Pop` 浮动图表之间可以双向同步十字虚线。

## 1. 背景

当前主窗口启动后标题仍显示为 `Gemini Trade Review`，与产品名称不一致。

当前十字同步链路由 `MainWindow` 直接把 `ChartWidget.sig_mouse_moved_with_price` 连接到 `sync_all_charts_crosshair(...)`，而同步目标又通过 `get_crosshair_sync_targets(...)` 过滤。该过滤逻辑会排除 `is_detached=True` 的图表，因此一旦图表通过 `Pop` 脱离主布局，就不会再收到主图表的十字同步。

这会导致以下问题：

- 主界面图表移动鼠标时，浮动图表十字虚线不更新
- 浮动图表移动鼠标时，主界面图表也无法形成统一同步语义
- 十字同步逻辑与布局附着状态耦合，后续继续扩展布局时容易再次出错

## 2. 目标

- 主程序窗口标题改为 `TradeReview`
- 引入独立的十字同步控制器，统一管理所有活动图表的十字同步
- 主界面图表与 `Pop` 浮动图表之间实现双向十字同步
- 十字同步目标选择不再依赖 `is_detached`
- 尽量减少对现有布局、drawing、replay 逻辑的影响

## 3. 非目标

- 本轮不改 `ChartWidget` 的十字显示样式
- 不改现有图表布局系统（Tabs / Vertical / Grid 2x2 / Dual Vertical）
- 不重构 drawing、replay、指标刷新逻辑
- 不处理多窗口之间的更多全局同步功能，只处理十字光标

## 4. 方案概述

新增一个轻量的 `CrosshairSyncController`，作为主窗口持有的独立同步子系统。

职责：

- 注册需要参与十字同步的 `ChartWidget`
- 接收某个 chart 发出的 `(timestamp, price)` 十字事件
- 统一广播给其它 chart 的 `sync_crosshair(...)`
- 只排除事件源 chart 自己，不根据 `is_detached` 做过滤

这样十字同步将从“布局派生行为”变成“图表注册表驱动行为”，无论某个图表当前位于主布局还是浮动窗口，只要它仍是活动 chart，就参与同步。

## 5. 架构与职责

### 5.1 `CrosshairSyncController`

建议新建文件：`ui/crosshair_sync.py`

最小接口：

- `register_chart(chart)`
- `unregister_chart(chart)`（可选，如果当前关闭路径不方便接入，也允许先只做惰性跳过）
- `sync_from(source_chart, timestamp, price)`

同步规则：

- 遍历当前已注册图表
- 跳过 `source_chart`
- 跳过已销毁或明显无效的对象
- 对其余图表调用 `chart.sync_crosshair(timestamp, price)`

### 5.2 `MainWindow`

修改点：

- 启动标题改为 `TradeReview`
- 在 `__init__()` 中创建一个 `self.crosshair_sync_controller`
- 在 `init_charts()` 中为每个新建 chart 调用 `register_chart(chart)`
- 把原来 `chart.sig_mouse_moved_with_price.connect(partial(self.sync_all_charts_crosshair, chart))`
  替换为统一发往 controller 的同步入口

`detach_chart()`、`attach_chart()`、`FloatingChartWindow` 仍只负责 UI 挂载，不再参与十字同步目标选择。

### 5.3 `ChartWidget`

`ChartWidget` 不需要大改，只保留现有：

- 鼠标移动时发出 `sig_mouse_moved_with_price`
- 接收 `sync_crosshair(timestamp, price)` 并更新十字显示

也就是说，当前 chart 既可以是事件源，也可以是同步目标，但不需要知道对方是在主布局还是浮动窗口。

## 6. 数据流

### 6.1 主图表 -> 浮动图表

1. 主界面 chart 鼠标移动
2. `ChartWidget` 发出 `sig_mouse_moved_with_price(timestamp, price)`
3. `CrosshairSyncController.sync_from(source_chart, timestamp, price)` 收到事件
4. controller 遍历所有已注册 chart
5. 跳过源 chart，把事件广播给其它 chart
6. 浮动 chart 执行 `sync_crosshair(...)`

### 6.2 浮动图表 -> 主图表

流程相同，只是 `source_chart` 变为浮动 chart。

## 7. 文件变更

预计改动：

- 新建 `ui/crosshair_sync.py`
- 修改 `ui/main_window.py`
- 新增或修改相应测试文件

`ui/chart_performance.py` 中现有的 `get_crosshair_sync_targets(...)` 可以先保留，以减少回归风险；但新链路不再依赖它。

## 8. 测试策略

至少补以下测试：

- `MainWindow` 启动标题为 `TradeReview`
- 主界面 chart 发出十字事件时，浮动 chart 会收到 `sync_crosshair(...)`
- 浮动 chart 发出十字事件时，主界面 chart 会收到 `sync_crosshair(...)`
- 源 chart 不会同步给自己
- chart 经过 detach/attach 后，无需重新接线也能继续同步

## 9. 风险控制

- controller 只做广播，不保存复杂图形或视图状态
- 同步目标筛选只排除源 chart，避免把布局状态再次引入
- 若 chart 已关闭或状态无效，controller 只做安全跳过，不抛异常
- 原有布局与浮动窗口逻辑保持不动，降低回归面

## 10. 结果

完成后，系统应满足：

- 主窗口标题显示为 `TradeReview`
- 主界面图表与 `Pop` 图表之间可以双向同步十字虚线
- 十字同步逻辑不再依赖图表是否 detached
- 后续扩展布局时，不需要再修改十字同步规则
