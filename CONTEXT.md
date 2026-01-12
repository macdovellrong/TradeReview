# Development Context (Snapshot 2026-01-11)

## 架构说明 (Architecture)
1.  **数据流**: `DataEngine` 负责加载 Parquet 数据并利用 Pandas 的 `resample` 功能动态合成 K 线。指标计算（EMA, BB）也在 Engine 层预先计算完成。
2.  **UI 架构**:
    *   `MainWindow`: 管理全局控制面板、布局切换逻辑和定时器。
    *   `ChartWidget`: 封装了 PyqtGraph 的 `GraphicsLayoutWidget`。由于 Finplot 的 `create_plot_widget` 在手动嵌入时存在坐标轴显示不稳定的问题，目前的方案是使用原生的 `pg.PlotItem` 并通过 Monkeypatch 注入 Finplot 所需的属性 (`yscale`, `datasrc`, `win` 等)，从而利用 Finplot 的 `candlestick_ochl` 进行高性能绘制。
3.  **布局管理**: 通过动态重建布局容器（`switch_layout`）实现了 Vertical、Grid、Tabs 模式的无缝切换。

## 关键技术细节 (Technical Notes)
*   **缩放逻辑**: 通过覆盖 `ViewBox.wheelEvent` 实现了 `Wheel -> X Zoom` 和 `Ctrl + Wheel -> Y Zoom`。
*   **绘图性能**: 指标线使用 `pg.PlotCurveItem` 原生绘制以避免与 Finplot 的 `datasrc` 管理冲突。K 线使用 `fplt.candlestick_ochl` 绘制并设置 `ZValue(10)` 确保置顶。
*   **时区**: 数据在加载时从 UTC 转换为 `America/New_York`，在绘图前去除 tz-info 以兼容 Finplot 的坐标计算。

## 下一步 (Next Steps)
1.  **十字光标同步**: 需要实现 `SignalProxy` 监听鼠标移动，并通过全局时间戳同步各窗口的 `InfiniteLine`。
2.  **指标管理**: 目前指标是固化在代码里的，未来可考虑增加 UI 菜单动态增删指标。
