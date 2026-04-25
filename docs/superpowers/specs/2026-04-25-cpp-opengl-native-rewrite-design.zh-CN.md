# TradeReview C++/Qt/OpenGL 原生重构设计

日期：2026-04-25

## 1. 背景

当前 Python / PyQtGraph 版本已经完成一轮大数据视口加载优化：

- DuckDB 模式打开文件时只读取 metadata，不全量读取 `ticks`。
- `DataEngine.get_candles_window(period, start, end)` 支持按时间窗口查询预计算 candles。
- `ChartWidget` 可以持有当前窗口数据，并在越过窗口边界时请求重载。
- LOD 规则可以在大范围浏览时自动选择更粗周期。
- 多图、回放、十字线同步、绘图工具、Fib 配置和会话恢复已经有可用实现。

这些优化证明了一个关键架构方向：性能问题不能靠把全量 tick 交给 UI 或 GPU 解决，而应靠“视口窗口查询 + LOD + 异步预取 + 批量渲染”。C++/OpenGL 版本应继承这个数据模型，而不是把全量 tick 上传显存。

用户当前目标不是短期过渡，而是长期商用级复盘软件。因此本设计按“一步到位的 C++/Qt/OpenGL 原生应用”展开，同时保留 Python 版作为参考实现和行为对照。

## 2. 设计输入

本设计综合了四类输入：

1. 现有 Python 代码和测试：
   - `engine/data_engine.py`
   - `engine/replay_engine.py`
   - `ui/chart_widget.py`
   - `ui/main_window.py`
   - `ui/chart_lod.py`
   - `ui/chart_windowing.py`
   - `ui/drawings/*`
   - `tests/test_data_engine_window_queries.py`
   - `tests/test_main_window_time_navigation.py`
   - `tests/test_drawing_chart_widget.py`
2. 已有性能设计：
   - `docs/superpowers/specs/2026-04-25-viewport-lod-performance-design.zh-CN.md`
3. 并行子代理调研：
   - 产品能力边界调研
   - C++/Qt/OpenGL 渲染架构调研
   - DuckDB、线程、缓存和回放数据模型调研
4. Gemini CLI 架构评审：
   - Gemini 倾向 `Qt Quick + 自定义 QQuickItem/QSGRenderNode/QRhi`。
   - 本设计将其作为重要备选方案，但推荐首版使用 `Qt Widgets + QOpenGLWidget`。

官方资料依据：

- Qt 文档说明 Qt Widgets 可通过 `QOpenGLWidget` 集成 OpenGL，Qt Quick 默认硬件加速并可选择 OpenGL 后端。参考：https://doc.qt.io/qt-6/qtopengl-index.html
- Qt 文档说明 `QOpenGLWidget` 是 Widgets 中嵌入 OpenGL 的稳定跨平台方案，同时提示其 FBO、composition 和多窗口限制。参考：https://doc.qt.io/qt-6/qopenglwidget.html
- Qt 文档说明 `QSGRenderNode` 可在 Qt Quick scene graph 中插入 QRhi 或 native 3D API 渲染命令。参考：https://doc.qt.io/qt-6/qsgrendernode.html
- Blend2D 官方资料说明它是 C++ 高性能 2D 矢量图形引擎，核心能力包括分析式光栅化、JIT 生成 2D pipeline、SIMD 和多线程渲染。参考：https://blend2d.com/about.html
- MetaTrader 5 Build 5430 发布说明记录其图表图形核心从 GDI 替换为 Blend2D，用于改进图表、指标、HiDPI、透明度和跨系统一致性。参考：https://www.metatrader5.com/zh/releasenotes/terminal/2418
- DuckDB 文档说明 C++ API 是内部 API，不保证稳定，应用级集成建议优先考虑 C API。参考：https://duckdb.org/docs/current/clients/cpp.html
- DuckDB 文档说明 Parquet 是压缩列式格式，并支持高效读取、filter/projection pushdown。参考：https://duckdb.org/docs/current/data/parquet/overview.html
- DuckDB 文档说明有 zonemap，时间有序数据能改善压缩和查询跳过效果。参考：https://duckdb.org/docs/current/guides/performance/indexing.html

## 3. 总体决策

### 3.1 仓库策略

采用当前仓库内的新长期分支和 `native/` 子目录：

```text
TradeReview/
  engine/              # Python 参考实现
  ui/                  # Python 参考实现
  tools/               # 现有数据转换工具
  tests/               # Python 行为测试
  docs/                # 设计和计划
  native/              # C++/Qt/OpenGL 原生版
    CMakeLists.txt
    app/
    core/
    data/
    chart/
    replay/
    drawing/
    tests/
```

理由：

- Python 版仍是可运行参考实现，保留在同一仓库有利于行为对照。
- 数据格式、转换工具和业务文档可以共享。
- C++ 版成熟后，如果需要独立发布，再拆仓库也不迟。

分支策略：

```text
main                         # Python 稳定版
cpp-opengl-native-design      # 当前设计分支
cpp-opengl-native             # 后续实现主分支
```

### 3.2 推荐技术栈

推荐首版技术栈：

```text
C++20
Qt 6 Widgets
QOpenGLWidget
OpenGL 3.3+ core profile
CMake
DuckDB C API 或封装后的 libduckdb
Catch2 / GoogleTest
```

推荐 UI 路线：

```text
Qt Widgets 主应用
  -> QMainWindow / Dock / Toolbar / Menu / Dialog
  -> ChartViewWidget : QOpenGLWidget
  -> QPainter overlay 绘制文本和标签
```

不建议首版直接使用 Qt Quick/QML 作为主框架。原因是 TradeReview 更像专业桌面工作站，需要大量右键菜单、浮窗、多图布局、精确鼠标命中、工具栏和表单设置。Qt Quick 的优势是现代 UI、动画和触控，但会把难点转移到 SceneGraph、自定义节点、输入事件和 Widgets 混用。

保留分歧：Gemini 建议 `Qt Quick + QSGRenderNode/QRhi`，理由是 UI 现代感和未来多图形后端更好。该方案可作为第二阶段或产品 UI 重设计时的备选，而不是首版默认路线。

### 3.3 不做的事

首版明确不做：

- 不把全量 tick 上传 GPU。
- 不让 GPU 持有全量历史 candles。
- 不实时从 tick 重建所有周期 K 线。
- 不用可见窗口临时从头计算 EMA/MACD/RSI。
- 不在 UI 线程执行 DuckDB 查询或指标计算。
- 不为每根 K 线创建 C++/Qt 对象。
- 不用 GPU picking 做首版命中测试。

## 4. 目标与非目标

### 4.1 目标

1. 支持 5 年及更长 XAUUSD tick 数据的流畅浏览和回放。
2. 以 C++/Qt/OpenGL 原生应用实现长期商用级架构。
3. 数据层保留 DuckDB 窗口查询、预计算 candles 和 LOD。
4. 渲染层使用 OpenGL 批量绘制当前窗口和预取窗口。
5. 多图同步、十字线、绘图工具、Fib、回放和时间跳转从设计第一天纳入架构。
6. 保留 Python 版关键业务语义，避免 C++ 版显示和回放结果漂移。

### 4.2 非目标

1. 不追求第一版功能覆盖当前 Python 版全部产品细节。
2. 不把 C++ 版本做成 Python 扩展或 PyQt 嵌入控件。
3. 不以 GPU 作为数据仓库。
4. 不在第一版引入 QRhi/Vulkan/Metal/D3D 抽象。
5. 不把自定义 mmap 二进制格式作为第一版唯一数据源。

## 5. 总体架构

```text
TradeReview Native

UI / Workspace
  MainWindow
  WorkspaceController
  ChartDockArea
  FloatingChartWindow
  SettingsDialog

Chart
  ChartViewWidget : QOpenGLWidget
  ChartInteractionController
  ChartSceneModel
  ChartIndexMapper
  GLChartRenderer
    GridLayer
    CandleLayer
    VolumeLayer
    IndicatorLineLayer
    HistogramLayer
    DrawingLayer
    CrosshairLayer
    OverlayLayer

Data
  MarketDataService
  DuckDbRepository
  CandleWindowProvider
  LodResolver
  WindowCache
  IndicatorProvider

Replay
  ReplaySession
  ReplayScheduler
  TickChunkProvider
  BarBuilder
  IndicatorState

Drawing
  DrawingStore
  DrawingCompiler
  HitTestService
  FibSettings

Sync
  CrosshairSync
  ViewRangeSync
  ChartDataCoordinator
```

运行时主流程：

```text
用户打开 DuckDB
  -> DataStore.open_readonly()
  -> 读取 metadata 和可用周期
  -> WorkspaceController 初始化默认四图
  -> ChartDataCoordinator 请求每个 chart 的可见窗口
  -> Data scheduler 分发 DuckDB 查询
  -> CandleWindow 返回 UI 线程
  -> ChartSceneModel 接收 generation 匹配的数据
  -> GLChartRenderer 在 QOpenGLWidget 线程上传窗口 buffer
  -> OpenGL 批量绘制 K线、成交量、指标、绘图和十字线
```

## 6. 当前 Python 行为必须保留的业务规则

这些规则是 C++ 版的产品正确性边界：

1. QDM 导出的 naive tick 时间按 `America/New_York` 解释，不默认 UTC。
2. K 线按 NY Close 17:00 anchor 对齐。
3. OHLC 使用左闭左标语义。
4. 周期表名必须处理 `1m` 和 `1M` 的历史冲突，月线建议命名为 `1mo` 或统一映射。
5. DuckDB 打开时先读 metadata，不全量加载 `ticks`。
6. 常用指标优先使用 DuckDB candles 表里的预计算列。
7. 如果运行时计算指标，必须带 warmup，不能只对可见窗口从头计算。
8. 时间跳转需要 floor 到分钟，并 clamp 到数据范围内。
9. 绘图对象必须保存 canonical 坐标，不能保存局部 x。
10. Fib 配置要 snapshot 到绘图对象，后续全局配置变化不能影响旧图。
11. 十字线同步传播 timestamp/price，不传播某张图的局部 x。
12. 多图可以使用不同周期和不同 LOD，但共享时间语义。
13. 图表右侧应保留视觉空白，最后一根 K 线不贴边。

## 7. 数据层设计

### 7.1 数据源分层

推荐三层数据源：

```text
Parquet
  长期归档和重建来源

DuckDB
  运行时主数据源
  ticks + candles 多周期 + metadata + indicator columns

Optional mmap binary cache
  后续性能瓶颈确认后加入
  仅作为热缓存，不作为唯一真相源
```

首版只要求 DuckDB 运行时路径稳定。Parquet 继续作为导入和重建来源。自定义二进制缓存放入设计，但不作为第一阶段实现要求。

### 7.2 DuckDB schema

建议结构：

```sql
ticks(
  timestamp TIMESTAMP,
  price DOUBLE,
  volume DOUBLE
)

candles_30s(...)
candles_1m(...)
candles_5m(...)
candles_15m(...)
candles_1h(...)
candles_4h(...)
candles_1D(...)

dataset_meta(
  key VARCHAR,
  value VARCHAR
)

cache_manifest(
  period VARCHAR,
  row_count BIGINT,
  min_ts TIMESTAMP,
  max_ts TIMESTAMP,
  indicator_version VARCHAR,
  source_fingerprint VARCHAR
)
```

要求：

- `ticks` 和 candles 表按 `timestamp` 顺序写入，利用 DuckDB zonemap 改善范围查询。
- candles 表包含常用指标列：EMA20/30/40/50/60/100/240、BB、MACD、RSI。
- schema 版本、指标版本和源文件 fingerprint 必须写入 metadata。

### 7.3 C++ API 边界

DuckDB 官方说明 C++ API 是内部 API，不保证稳定，应用集成更建议 C API。因此 C++ 版应把 DuckDB 隔离在 `DuckDbRepository` 后面：

```cpp
struct TimeRange {
    int64_t start_ns;
    int64_t end_ns;
};

struct DataSetInfo {
    std::string path;
    int64_t tick_count;
    TimeRange tick_range;
    std::vector<std::string> periods;
    std::vector<std::string> indicators;
    std::string schema_version;
    std::string indicator_version;
};

struct CandleWindowRequest {
    uint64_t chart_id;
    uint64_t generation;
    std::string requested_period;
    TimeRange visible_range;
    int pixel_width;
    double buffer_multiplier;
    bool include_indicators;
    int warmup_bars;
};

struct CandleWindow {
    uint64_t chart_id;
    uint64_t generation;
    std::string requested_period;
    std::string actual_period;
    TimeRange loaded_range;
    TimeRange visible_range;
    std::vector<int64_t> timestamp_ns;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;
    std::unordered_map<std::string, std::vector<double>> indicators;
    bool from_cache;
};
```

同步 repository 接口：

```cpp
class IDataStore {
public:
    virtual ~IDataStore() = default;
    virtual DataSetInfo open_readonly(const std::string& path) = 0;
    virtual CandleWindow query_candles(const CandleWindowRequest& req) = 0;
    virtual TickSlice query_ticks(TimeRange range, size_t max_rows) = 0;
    virtual ReplayChunk query_replay_ticks(int64_t from_ns, int64_t to_ns, size_t max_ticks) = 0;
};
```

异步 service 接口：

```cpp
class IDataService {
public:
    virtual ~IDataService() = default;
    virtual uint64_t request_candle_window(CandleWindowRequest req) = 0;
    virtual void cancel_chart(uint64_t chart_id) = 0;
    virtual void bump_generation(uint64_t chart_id) = 0;
};
```

结果返回 UI 前必须检查：

```cpp
if (result.generation != chart.current_generation()) {
    discard_result();
}
```

## 8. LOD、窗口和缓存

### 8.1 LOD 规则

LOD 不应只按固定时间跨度判断。推荐按屏幕可承载点数动态选择：

```text
target_bars <= pixel_width * 2
actual_period >= requested_period
actual_period must exist in available candle tables
```

选择过程：

```text
visible_seconds / requested_period_seconds <= pixel_width * 2
  -> 保持 requested_period

否则从可用周期中选择最小满足者：
  5min, 15min, 1h, 4h, 1D, 1W
```

规则必须保证不会比用户请求周期更细。

### 8.2 查询窗口

每次图表请求包含可视范围和 buffer：

```text
query_start = visible_start - visible_span * buffer_multiplier
query_end   = visible_end   + visible_span * buffer_multiplier
```

默认：

```text
buffer_multiplier = 2.0
prefetch_edge_fraction = 0.5
```

行为：

- 可视范围仍在 loaded range 内：只更新 view matrix，不查库。
- 可视范围接近 loaded range 边缘：后台预取。
- 可视范围越界：提交高优先级窗口查询。
- 快速拖动时，只保留最新 generation 请求。

### 8.3 缓存分层

```text
L1 chart-local current window
  当前图表正在渲染的数据

L2 process LRU window cache
  key = dataset_id + period + rounded_range + indicator_version

L3 optional mmap binary cache
  后续确认 DuckDB 成为瓶颈后再加入
```

L3 二进制缓存如果引入，必须包含：

- schema version
- indicator version
- source fingerprint
- period
- row count
- min/max timestamp
- record layout

## 9. 线程模型

推荐三层线程模型：

```text
UI / GL thread
  Qt event loop
  QOpenGLWidget paintGL
  ChartSceneModel apply result
  OpenGL buffer create/update/destroy

Data scheduler thread
  request coalescing
  generation filtering
  priority queue
  LRU cache lookup
  cancellation of queued stale jobs

Worker pool
  DuckDB query
  indicator warmup calculation
  optional binary cache read
  array construction
```

约束：

- 每个 DuckDB worker 使用独立只读 connection。
- GL 对象只能在拥有 GL context 的线程创建、更新和销毁。
- Worker 不直接访问 Qt widget。
- Worker 返回 plain data，UI 线程接收后再提交给 renderer。
- 旧 generation 查询可以不强制中断，但结果必须丢弃。

请求优先级：

1. 当前可视窗口。
2. 当前方向的预取窗口。
3. 回放下一 tick chunk。
4. 后台缓存构建。

## 10. 渲染层设计

### 10.1 图表组件

核心图表组件：

```cpp
class ChartViewWidget : public QOpenGLWidget {
    ChartSceneModel scene_;
    ChartInteractionController interaction_;
    ChartIndexMapper index_mapper_;
    GLChartRenderer renderer_;
};
```

一个 `ChartViewWidget` 内部渲染主图、成交量、MACD、RSI 等多个 panel。不要为每个 panel 创建一个 `QOpenGLWidget`，避免 OpenGL context、FBO 和 composition 成本失控。

### 10.2 图层

```text
GLChartRenderer
  GridLayer
  CandleLayer
  VolumeLayer
  IndicatorLineLayer
  HistogramLayer
  DrawingLayer
  CrosshairLayer
  OverlayLayer
```

策略：

- `CandleLayer`：批量 VBO，K 线实体用实例化 quad，影线用 line batch。
- `VolumeLayer`：批量柱状实例。
- `IndicatorLineLayer`：每个指标一条 polyline VBO，支持 NaN 断线。
- `HistogramLayer`：MACD histogram 批量柱状。
- `GridLayer`：根据 viewport 生成少量 tick。
- `CrosshairLayer`：鼠标移动时更新少量动态 buffer。
- `DrawingLayer`：从 drawing spec 编译 render commands。
- `OverlayLayer`：首版用 Qt/QPainter 绘制价格标签、时间标签、K 信息、Fib 文本。

### 10.3 GPU buffer 生命周期

三类 buffer：

```text
Window buffers
  K 线、成交量、指标
  绑定 period + loaded_window + generation

Dynamic interaction buffers
  十字线、测量线、绘图预览
  鼠标移动时小量更新

Overlay CPU draw
  文本、标签、悬浮信息
  首版不进 GPU
```

同一 loaded window 内平移缩放只改 view/projection matrix，不重传 K 线 buffer。

## 11. 坐标系统

采用双坐标：

```text
Canonical coordinate
  timestamp_ns + price
  用于数据查询、绘图持久化、同步、回放定位

Render coordinate
  dense_x + price
  用于实际渲染
  保持无交易空洞压缩体验
```

`ChartIndexMapper` 是核心边界：

```text
timestamp_ns -> dense_x
dense_x -> timestamp_ns
mouse_px -> dense_x / price
dense_x / price -> screen_px
nearest timestamp lookup
visible time range lookup
```

同步规则：

- 十字线同步传 timestamp/price。
- 多图视图同步传 canonical time range。
- 各图自行映射到本地 dense_x。

## 12. 绘图系统

绘图对象存储 canonical 数据：

```text
drawing_id
type
points: [{ timestamp_ns, price }]
style
config_snapshot
created_at
updated_at
```

首版工具：

- 水平线
- 垂直线
- 趋势线
- 矩形
- Fib retracement
- 三点 Fib extension

渲染：

- DrawingStore 保存 specs。
- DrawingCompiler 根据当前 ChartIndexMapper 编译 screen-space primitives。
- DrawingLayer 批量绘制线、矩形、Fib levels。
- OverlayLayer 绘制 Fib 文本和标签。

命中测试：

- CPU screen-space tolerance，默认 4 到 8 px。
- 首版线性扫描绘图对象。
- 大量绘图对象后再加 spatial index。
- HitResult 显式表达命中对象：

```cpp
struct HitResult {
    HitKind kind;
    uint64_t object_id;
    std::string subpart;
};
```

## 13. 回放设计

当前 Python `ReplayEngine` 依赖全量 `df_ticks`。C++ 版必须改成 chunked replay。

```text
ReplaySession
  current_time
  next_tick_cursor
  per_period_bar_builder
  indicator_state
  tick_chunk_cache
```

流程：

1. 设置 replay start 时，查询起点附近 tick chunk。
2. 每次 advance 查询 `[last_tick_time, target_time]`。
3. 单帧限制 `max_ticks_per_frame`，超出则分帧推进。
4. 每个周期维护当前 bar builder。
5. 指标状态从 warmup candles 初始化，然后增量更新当前未完成 K 线。
6. 到达数据末尾时停止播放。

回放图表仍然使用 `CandleWindow` 模型，只是当前最后一根 K 线可由 tick stream 增量更新。

## 14. UI 工作台

首版工作台保留专业桌面工具形态：

- 主窗口
- 打开数据文件
- 默认四图：1h、15min、5min、1min
- 图表数量控制
- Tabs、Vertical、Dual、2x2 布局
- 浮窗 detach/attach
- 周期切换
- 指标开关
- 回放开关、播放、暂停、速度、步进
- 时间跳转
- 十字线同步
- 绘图工具栏
- Fib 设置
- 会话恢复：数据源、中心时间、布局、周期、启用图表

## 15. 错误处理

必须显式处理：

- DuckDB 文件打不开。
- schema 缺列。
- candles 表缺失。
- 指标版本不匹配。
- 时间范围为空。
- 查询超时。
- worker 返回旧 generation。
- GL context 丢失或重建。
- 数据源位于网络路径导致查询抖动。

用户可见错误策略：

- 数据文件错误：弹窗并阻止加载。
- candles 表缺失：提示重建 DuckDB 或切换周期。
- 窗口查询慢：图表轻量 loading 状态，不冻结 UI。
- 旧查询返回：静默丢弃。

## 16. 测试策略

### 16.1 单元测试

覆盖：

- Period parsing。
- LOD resolver。
- Window range builder。
- ChartIndexMapper。
- Drawing spec normalization。
- HitTestService。
- TimeRange clamp/floor。
- Table name mapping。

### 16.2 数据层测试

使用临时 DuckDB：

- metadata load 不读取全量 ticks。
- window query 只返回请求范围。
- 缺表返回明确错误。
- `1m` 与 `1M` 命名不冲突。
- 指标列存在性检查。
- generation 丢弃旧结果。

### 16.3 渲染测试

首版以工程可执行验证为主：

- 创建 OpenGL context。
- 上传小窗口数据。
- 渲染非空 framebuffer。
- resize 后 FBO 正常重建。
- 多 panel 坐标映射正确。

可选后续：

- 截图 golden image。
- GPU buffer 数量和显存上限检查。
- 60 FPS smoke test。

### 16.4 端到端测试

用固定小数据集验证：

- 打开数据。
- 显示四图。
- 时间跳转。
- 拖动到窗口边界触发加载。
- 大范围缩放触发 LOD。
- 十字线同步。
- 绘图对象跨周期显示。
- 回放推进和停止。

## 17. 里程碑

### M0：Native Skeleton

目标：跑通 C++/Qt 工程骨架。

范围：

- `native/` CMake 工程。
- Qt Widgets 主窗口。
- 一个空 `QOpenGLWidget`。
- 基础测试框架。

### M1：单图数据窗口 + OpenGL K线

目标：全链路显示一个 DuckDB candle window。

范围：

- DuckDB metadata load。
- `query_candles`。
- `CandleWindow`。
- `ChartIndexMapper`。
- CandleLayer。
- 基础平移缩放。

不包含：

- 绘图工具。
- 回放。
- 多图同步。

### M2：LOD、缓存和异步加载

目标：任意时间跳转和拖动不阻塞 UI。

范围：

- Data scheduler。
- Worker pool。
- generation。
- LRU window cache。
- LOD resolver。
- buffer/prefetch。

### M3：多图工作台和同步

目标：恢复 Python 版核心工作区。

范围：

- 四图布局。
- 周期切换。
- 十字线同步。
- 时间范围同步。
- 浮窗。

### M4：绘图和 Fib

目标：恢复当前绘图工具核心能力。

范围：

- DrawingStore。
- DrawingLayer。
- HitTestService。
- Fib settings snapshot。
- 删除/清空。

### M5：回放

目标：不全量加载 tick 的 chunked replay。

范围：

- ReplaySession。
- TickChunkProvider。
- BarBuilder。
- 回放速度和步进。
- 到末尾停止。

### M6：产品化

目标：稳定可用。

范围：

- 完整会话恢复。
- 错误提示。
- 设置系统。
- 性能 profiling。
- 打包发布。

## 18. 设计分歧与保留项

### 18.1 Qt Widgets/QOpenGLWidget vs Qt Quick/QSGRenderNode

推荐：`Qt Widgets + QOpenGLWidget`。

原因：

- 更贴近当前专业桌面工具形态。
- 右键菜单、浮窗、停靠布局、表单和精确鼠标交互更直接。
- OpenGL 调试路径更短。

保留：Gemini 建议 `Qt Quick + QSGRenderNode/QRhi`，适合现代 UI 和未来多图形后端。如果后续决定重做品牌级现代 UI，可以重新评估。

### 18.2 OpenGL vs Blend2D

推荐首版：继续使用 `OpenGL` 作为主图数据渲染后端。

原因：

- K 线、成交量、指标线在可见窗口内属于大量重复几何，OpenGL 批量 buffer 绘制更直接。
- 当前 M0/M1 计划已经围绕 `QOpenGLWidget`、`GLChartRenderer` 和视口窗口模型展开，能够先验证数据边界、坐标映射和窗口刷新语义。
- Blend2D 是高性能 CPU 2D 软件渲染引擎，不等同于 GPU 视口方案。它更适合作为 GDI/GDI+ 替代、复杂 2D 矢量图形、文字、半透明标注、画线工具和静态图层后端。

保留：后续在 `IRenderer` 或 chart rendering 边界稳定后，可以新增 `Blend2DChartRenderer` 实验后端，或采用混合路线：OpenGL 绘制主图高频数据，Blend2D/Qt 绘制文字、标注、复杂画线和离屏缓存层。首版不引入 Blend2D 依赖，避免同时承担 OpenGL 与 Blend2D 两套渲染集成成本。

### 18.3 DuckDB C API vs C++ API

推荐：首选封装 C API，或把 C++ API 使用严格限制在 `DuckDbRepository` 内。

原因：

- DuckDB 官方说明 C++ API 是内部 API，不保证稳定。
- C API 更适合长期应用边界。

### 18.4 内部时间基准

推荐首版：`int64 timestamp_ns`，语义与现有 DuckDB candles 的 NY wall-time 对齐。

保留分歧：

- 长期更严谨的做法是 UTC ns + session calendar。
- 但当前 Python 版和已生成 DuckDB 数据使用 NY wall-time 语义，首版应优先保证行为一致。

### 18.5 自定义二进制缓存是否首版实现

推荐：首版不实现，只设计接口和 metadata。

原因：

- DuckDB ordered timestamp + zonemap + LRU cache 足够验证多数场景。
- 过早引入 mmap cache 会增加 schema 演进和一致性风险。

### 18.6 GPU 是否做 LOD

推荐：LOD 主要在数据层/CPU 侧选择周期，GPU 负责批量渲染当前窗口。

保留：

- 后续可用 shader 做简单裁剪或像素级降采样。
- 不建议首版把复杂 LOD 放进 shader。

## 19. 验收标准

设计级验收：

1. C++ 版不依赖全量 tick 内存。
2. GPU 只持有当前窗口和预取窗口。
3. 数据查询接口以时间窗口为中心。
4. LOD 不会选择比用户请求更细的周期。
5. 多图同步基于 timestamp，而不是局部 x。
6. 绘图对象持久化基于 timestamp/price。
7. 回放不依赖全量 tick DataFrame。
8. OpenGL buffer 生命周期绑定 generation。
9. UI 线程不执行 DuckDB 查询。
10. Python 版关键业务语义有对应测试或迁移说明。

M1 技术验收：

1. 能打开临时 DuckDB。
2. 能读取 metadata。
3. 能查询单个 candle window。
4. 能在 `QOpenGLWidget` 中绘制 K 线。
5. 平移缩放不重查库、不重传窗口 buffer。
6. resize 后正常显示。

## 20. 下一步

本设计文档通过后，再编写实施计划：

```text
docs/superpowers/plans/2026-04-25-cpp-opengl-native-rewrite.md
```

实施计划应从 M0/M1 开始，采用小步提交，不直接实现完整产品。
