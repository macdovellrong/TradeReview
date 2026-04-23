# TradeReview Review Platform Roadmap

> **For agentic workers:** This roadmap is intentionally phase-based. Do not implement it as one large branch. Split execution into separate plans per phase.

**Goal:** 把 `TradeReview` 从“带回放和绘图的图表工具”推进到“可长期积累、可稳定复现、可持续改进的交易复盘平台”。

**Architecture:** 先补三层地基，再补表层能力。第一层是复盘对象模型，第二层是工作区与持久化，第三层是时间、K 线、回放、回测的一致性。只有这三层稳定后，交易记录、报表、训练闭环才不会碎片化。

**Tech Stack:** Python, PyQt6, pandas, DuckDB, pytest

---

## Scope Note

这不是一个适合“一次性做完”的需求包。它至少包含 4 个相对独立的子系统：

1. 复盘对象与持久化
2. 工作区与回放导航
3. 数据语义与正确性
4. 报表与复盘闭环

建议按阶段拆成单独实施计划，而不是直接并行改全仓库。

## Product Principles

- 先定义复盘对象，再定义界面动作。
- Replay 是一种工作模式，不是几个按钮。
- 所有标注、笔记、截图、标签都必须可持久化。
- 工作区恢复必须恢复上下文，不只是时间点。
- UI、回放、回测必须共享同一套时间和 K 线语义。
- 桌面产品优先坚持 `local-first`，先把单人闭环做深。

## Phase 0: 2 Weeks

**主题：先把地基补齐，但只做最小闭环。**

**目标**

- 让工作区状态不再只有 `db_path + center_time`
- 为后续复盘对象落地预留稳定存储入口
- 开始收敛时间与 candle 语义，避免继续扩散

**建议交付**

- 轻量版 `WorkspaceState`
  - 保存布局
  - 保存图表数量
  - 保存每个 chart 的周期
  - 保存 replay 开关、速度、当前位置
  - 保存最近文件列表
- 最近文件 / 最近数据源入口
- 回放时间精度分层
  - 保留分钟跳转
  - 增加 tick 级精确定位能力
- 单一 candle 语义入口的设计稿与落地骨架
  - 明确统一锚点
  - 明确 session/calendar 规则
  - 明确指标计算来源
- 关键语义测试补齐
  - DST 边界
  - workspace round-trip
  - precomputed candle 与 fallback 聚合一致性

**不做**

- 不在这一阶段做完整 trade journal
- 不做复杂 dashboard
- 不做多资产扩展

**完成标准**

- 重启应用后，用户能恢复到上次工作的基本现场
- 时间跳转不再被用户误解为“只有分钟粒度”
- UI 和数据层不再继续增加新的时间语义分叉

## Phase 1: 1 Month

**主题：把“看图”升级成“可记录的复盘”。**

**目标**

- 引入一等公民的复盘对象
- 让用户的标注、检查点、笔记可以累计
- 形成最小可用的单人复盘闭环

**建议交付**

- `ReviewSession`
  - 绑定数据源
  - 绑定时间范围
  - 绑定工作区快照
- `ReviewRecord`
  - long/short
  - entry/exit
  - stop/target
  - result/status
  - notes/tags/score
- `Bookmark / Checkpoint`
  - 命名
  - 快速跳转
  - 绑定某个 replay 时间点
- 标注持久化
  - drawings 与 review/checkpoint 绑定
  - 清除内存即丢失的行为结束
- replay scrubber
  - 连续时间轴
  - 当前进度可视化
  - 检查点定位入口

**不做**

- 不追求复杂协作
- 不做云同步
- 不把所有分析字段一次性铺满

**完成标准**

- 用户可以完成一次完整复盘并保存
- 下次可以重新打开同一份复盘并继续编辑
- 检查点、图形、笔记、交易记录之间有明确绑定关系

## Phase 2: 1-2 Months

**主题：把“可记录”升级成“可改进”。**

**目标**

- 报表能回答“下一步改什么”
- 回测、回放、复盘开始共享一套分析语言
- 产品从工具变成真正的工作流

**建议交付**

- review-oriented dashboard
  - tag 过滤
  - setup 胜率
  - 时段表现
  - 执行偏差
  - 规则破坏统计
- 回测 provenance
  - 数据版本
  - 参数
  - 代码版本
  - 生成时间
- intrabar 不确定性显式化
  - `unknown` / warning
  - 不再静默默认 stop
- replay / review / backtest 数据模型开始收敛
- 更清晰的同步策略
  - 主图
  - 跟随图
  - 独立图

**完成标准**

- 报表不只是展示结果，而是能定位问题
- 回测结果可追溯
- 用户开始能从“单次复盘”过渡到“周期性复盘改进”

## Deferred

这些方向有价值，但不应排在前面：

- 社区/分享优先
- AI 自动总结优先
- 更多技术指标优先
- 多市场大扩张优先
- 重 UI 视觉翻新优先

原因很简单：当前主要瓶颈不是“能不能再多看一点东西”，而是“用户做过的复盘能不能存住、找回、验证、复用”。

## Recommended Execution Order

1. 先单独写 `Phase 0` 实施计划
2. 完成后再写 `Phase 1` 实施计划
3. `Phase 2` 必须建立在前两阶段稳定之后

## Phase 0 Candidate Files

优先会触达这些区域：

- `ui/session_state.py`
- `ui/main_window.py`
- `ui/time_navigation.py`
- `engine/data_engine.py`
- `engine/replay_engine.py`
- `engine/data_validation.py`
- `tools/preprocess_qdm_tick_csv.py`
- `tools/convert_parquet_to_duckdb.py`
- `backtest/data.py`
- `tests/test_session_state.py`
- `tests/test_time_navigation.py`
- `tests/test_qdm_conversion.py`

## Phase 1 Candidate Files

- `ui/main_window.py`
- `ui/drawings/*`
- new: `ui/review_*`
- new: `engine/review_*`
- new: `tests/test_review_*`

## Phase 2 Candidate Files

- `backtest/*`
- `engine/*`
- new: `ui/report_*`
- new: `tests/test_backtest_*`
- new: `tests/test_replay_*`

