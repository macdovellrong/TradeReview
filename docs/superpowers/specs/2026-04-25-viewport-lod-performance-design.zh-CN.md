# TradeReview 大数据视口加载与 LOD 性能重构设计

日期：2026-04-25

## 背景

当前 TradeReview 已经支持从 DuckDB / Parquet 加载 XAUUSD tick 与预计算 K 线数据。实测当前 5 年数据规模为：

- `ticks`：278,200,009 行
- `candles_30s`：3,541,753 行
- `candles_1m`：1,773,581 行
- DuckDB 文件：约 3.6GB
- Parquet 文件：约 2.37GB

现有 Python / PyQtGraph 版本已经有可见窗口切片逻辑：`ChartWidget.refresh_visible_view()` 会按当前 X 轴范围从完整 DataFrame 中取一段数据绘制。但数据层仍然存在两个主要瓶颈：

1. DuckDB 加载路径会执行 `SELECT * FROM ticks ORDER BY timestamp`，把 2.78 亿 tick 全量读入 pandas。
2. 正常浏览模式下 `DataEngine.get_candles()` 会把某个周期的完整 K 线 DataFrame 读入缓存，再交给图表做视口切片。

这使得“加载后前面不卡，但拖到数据后段变卡”的问题难以靠调小绘制参数彻底解决。根因是 UI 与数据层仍然以全量数据为中心，而不是以当前视口为中心。

## 目标

1. 支持 5 年及更长历史数据在任意时间点跳转后保持交互流畅。
2. 正常浏览时只加载和绘制当前视口需要的数据窗口。
3. 拖动时在已加载 buffer 内只移动视图，不触发数据库查询。
4. 接近窗口边缘时异步预取下一段数据。
5. 大范围时间跳转时丢弃旧窗口，重新查询目标时间附近窗口。
6. 缩放跨度较大时自动使用更粗周期或聚合层级，避免一屏绘制过多 K 线。
7. 为后续 C++ / OpenGL 渲染层预留边界，但本设计不要求立即重写 UI。

## 非目标

- 不把 2.78 亿 tick 全量加载到 UI 内存。
- 不把全量 tick 上传到 GPU。
- 不在第一阶段重写 C++ / OpenGL。
- 不改变现有交易时区规则、NY Close 对齐规则和预计算 K 线语义。
- 不牺牲指标正确性来换取局部窗口速度。

## 推荐架构

采用“视口查询 + LOD + 异步预取”的架构：

```text
当前图表视口时间范围
        ↓
LOD Resolver 选择合适周期
        ↓
Window Provider 计算查询窗口和 buffer
        ↓
DuckDB 查询 candles 表
        ↓
ChartWidget 绘制当前窗口
        ↓
拖动接近边缘时后台预取
        ↓
大范围跳转时重新加载窗口
```

### 数据层

`DataEngine` 不再在打开 DuckDB 时读取全量 tick。加载文件时只读取元数据：

- tick 起止时间
- tick 总行数
- 可用 candles 表
- 每个 candles 表的起止时间和行数
- schema 校验结果与 warning

新增窗口查询接口：

```python
get_candles_window(period, start_time, end_time) -> pd.DataFrame
```

该接口从 DuckDB 的预计算 candles 表按时间范围查询：

```sql
SELECT *
FROM candles_1m
WHERE timestamp >= ?
  AND timestamp <= ?
ORDER BY timestamp
```

如果某周期没有预计算表，第一版不回退到全量 tick 重采样；应提示用户重建 DuckDB 或临时禁用该周期。后续可以加后台构建缓存。

### LOD 层

新增 `ChartLODResolver`，根据当前视口时间跨度、图表像素宽度和目标最大绘制点数选择周期。

示例规则：

```text
一屏 <= 2 天      使用 30s / 1m
2 天 - 30 天      使用 5m / 15m
30 天 - 1 年      使用 1h / 4h
> 1 年            使用 1D / 1W
```

第一版可以使用固定规则。后续再按屏幕宽度和 `max_points_per_screen` 动态计算。

### 图表层

`ChartWidget` 不再持有完整 `full_df` 作为浏览基础，而是持有当前加载窗口：

```text
active_period
visible_start_time
visible_end_time
loaded_start_time
loaded_end_time
window_df
window_generation
```

X 轴可以继续使用局部整数索引，但必须提供从局部 X 到时间、从时间到局部 X 的转换。绘图对象只绘制 `window_df`，不要依赖全量 DataFrame 长度。

### 预取策略

每次查询窗口包含：

```text
query_start = visible_start - buffer_span
query_end   = visible_end   + buffer_span
```

初始 buffer 可以取 `2x` 当前视口跨度。拖动时：

- 视口仍在 `[loaded_start, loaded_end]` 内：不查库。
- 视口距离窗口边缘小于 `0.5x visible_span`：后台预取。
- 视口已经超出窗口：同步加载目标窗口，并显示轻量 loading 状态。

后台查询必须带 generation id。旧查询返回时，如果 generation 已过期，直接丢弃，避免快速拖动时旧数据覆盖新视图。

### 指标正确性

EMA、MACD、RSI 等递推指标不能只用当前窗口临时从头计算，否则窗口开头会失真。推荐：

1. 优先使用 DuckDB candles 表中已经预计算好的指标列。
2. 如果必须运行时计算，查询窗口需要额外 warmup 段，例如至少 300 根到 1000 根。
3. 绘制时只显示可见窗口，但计算时保留 warmup 数据。

第一版应优先走预计算指标列，避免引入指标漂移。

### 后续 C++ / OpenGL 方向

C++ / OpenGL 适合作为渲染层升级，而不是数据层替代。即使未来使用 GPU，也仍应保留视口查询和 LOD：

- CPU / DuckDB 负责按时间范围取数和聚合层级。
- GPU 只持有当前窗口及邻近预取窗口。
- 拖动和缩放通过 view/projection matrix 实现。
- 大范围跳转仍然重新加载窗口。

不推荐把全量 tick 上传 GPU。278M tick 的时间、价格、volume 原始数组就需要数 GB 内存；加上顶点、索引、颜色和多周期缓存后显存压力过高，且屏幕不会真正显示这么多点。

## 当前 Python 版本的落地策略

先不重写 C++，在 Python 中完成以下重构：

1. DuckDB 加载改为元数据加载，不读取全量 ticks。
2. 新增 candles 窗口查询接口。
3. `ChartWidget` 改为窗口数据模型。
4. 拖动时使用 buffer 判断是否需要查询。
5. 增加简单 LOD 周期选择。
6. 大范围跳转时重新加载目标窗口。
7. 用测试锁定查询范围、buffer 行为、LOD 选择和旧查询丢弃行为。

这样可以先验证产品体验和数据边界。如果 PyQtGraph 绘制仍然成为瓶颈，再把图表渲染层替换为 C++ / OpenGL。

## 风险与约束

- DuckDB 查询必须使用时间索引友好的表结构，否则窗口查询会退化。
- 当前 `ChartWidget` 的十字线、绘图对象、同步中心逻辑依赖 X 轴整数索引，窗口化后需要统一时间到局部 X 的映射。
- 多图表、多周期、浮窗和同步十字线都必须在窗口数据模型下工作。
- 预取线程不能直接改 Qt UI，只能通过 signal 回主线程更新。
- 大范围缩放时自动切换周期可能改变用户看到的 K 线粒度，UI 需要明确当前实际显示周期。

## 验收标准

1. 打开 3.6GB DuckDB 时不再全量读取 `ticks`。
2. 首屏加载只查询当前时间附近窗口。
3. 跳转到任意日期后，图表能在可接受时间内重新显示。
4. 在已加载 buffer 内拖动不触发 DuckDB 查询。
5. 接近 buffer 边缘时最多触发一次有效预取。
6. 一屏绘制点数受上限控制，不随全库数据增长。
7. 现有测试通过，并新增性能边界测试。
