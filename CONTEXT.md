# Development Context (Snapshot 2026-01-11)

## 架构说明 (Architecture)
1.  **数据流**: `DataEngine` 负责加载 Parquet 数据并利用 Pandas 的 `resample` 功能动态合成 K 线。指标计算（EMA, BB）也在 Engine 层预先计算完成。
2.  **UI 架构**:
    *   `MainWindow`: 管理全局控制面板、布局切换逻辑、定时器和进度条同步。
    *   `ChartWidget`: 封装了 PyqtGraph 的 `GraphicsLayoutWidget`。
        *   **绘图核心**: 使用 `pg.GraphicsLayoutWidget` + `addPlot` 创建原生 PlotItem。
        *   **Monkeypatch**: 为了使用 Finplot 的高性能 K 线绘制函数 (`candlestick_ochl`)，对 `ViewBox` 进行了 Monkeypatch，模拟了 Finplot 所需的 `yscale`, `datasrc`, `win` 等属性。
        *   **指标绘制**: 指标线（EMA/BB）使用原生的 `pg.PlotCurveItem` 绘制，并强制将数据转换为 `float64` 的 numpy array，彻底避免了与 Finplot 内部数据源管理的冲突。
        *   **交互**: 
            *   自定义 `wheelEvent` 实现 X/Y 轴独立缩放（Ctrl 键修饰）。
            *   通过 `SignalProxy` 监听鼠标移动，实现了多窗口十字光标同步。

## 关键技术细节 (Technical Notes)
*   **缩放逻辑**: `wheelEvent` 被劫持，直接调用 `scaleBy`，绕过了 PyqtGraph 的默认行为。
*   **同步机制**: 利用 `pyqtSignal` 广播当前鼠标位置的时间戳，各子窗口接收信号后通过 `searchsorted` 快速定位并移动垂直线。
*   **布局管理**: 动态重建布局容器实现布局切换。每次切换布局或加载数据后，强制调用 `reset_charts_view` 触发 AutoRange，确保 K 线可见。

## 已知问题与待办 (Known Issues & Todo)
*   **性能**: 在极大数据量下（如全量 M1），重绘可能略有卡顿，未来可考虑分段加载或降采样。
*   **交互**: 已实现十字光标数值标签（价格与时间），支持随鼠标移动实时更新。
*   **DST/日历**: 需验证交易所日历与数据墙上时间在夏令时切换周完全对齐，避免时段错配。

## 近期更新记录 (Recent Updates)
*   **指标**: 主图新增 MACD + RSI 子图（RSI6/12/24 多线），MACD 柱体红/淡蓝区分正负。
*   **光标/测量**: 十字光标同步到 MACD/RSI 子图；Ctrl+左键拖动测量价差；Ctrl+滚轮仅缩放 Y 轴。
*   **回放/性能**: 回放改为 tick 级增量引擎，空档自动跳过，进度条替换为前进/后退步进。
*   **数据**: DuckDB 支持优先读取预计算 candles，转换脚本集成 CSV->Parquet->DuckDB。
