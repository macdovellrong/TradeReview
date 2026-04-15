# TradeReview Product Gap Report

Date: 2026-04-15
Status: Archived for later implementation

## Purpose

This file archives the current consensus on what TradeReview still lacks as a K-line replay/review product, and what existing capabilities need improvement before it can be treated as a mature review workstation.

The conclusions below are based on local code inspection plus parallel multi-agent review across:

- replay/review workflow
- trader annotation and journaling workflow
- data pipeline and replay/backtest correctness
- performance, rendering, and multi-window architecture

## Current Product Position

TradeReview already supports:

- loading QDM-processed DuckDB/parquet market data
- multi-period chart display
- replay mode with play/pause, speed, and step controls
- drawing tools
- chart pop-out windows
- basic session restore

But in its current shape, it is still closer to:

- a chart viewer with replay and drawing

than to:

- a complete discretionary trade review platform
- a scalable multi-market replay workstation

## Consensus Summary

The missing pieces fall into three layers:

1. review objects
2. data model and semantic consistency
3. workspace and rendering architecture

If those three layers are not strengthened, adding more surface features will create a fragmented product.

## P0: Must-Fix Product Gaps

### 1. Review record model is missing

The product does not yet have a durable object for:

- a review session
- a trade review record
- a bookmarked setup

Current `Save View` only stores:

- database path
- chart center time

It does not store:

- notes
- tags
- review status
- trade direction
- entry/exit
- screenshots
- scoring
- outcome
- linked drawings

Relevant code:

- `ui/main_window.py`
- `ui/session_state.py`

### 2. Annotation is not persistent

Current drawings are interactive, but they are still only in-memory chart objects.

That means:

- drawings are not saved as review data
- they are not bound to a trade or setup
- they cannot be exported as part of a review record
- they cannot be searched or filtered later

This blocks long-term review accumulation.

Relevant code:

- `ui/main_window.py`

### 3. Trade-level review workflow is missing

There is no first-class workflow for:

- mark entry
- mark exit
- classify long/short
- attach rationale
- record stop/target
- record result

The closest existing action is `Set Replay Start`, but that is a replay control, not a trade review object.

Relevant code:

- `ui/main_window.py`

### 4. Replay timeline scrubbing is missing

Replay currently has:

- play/pause
- speed buttons
- step back/forward
- datetime jump

But it still lacks a continuous draggable replay timeline/progress scrubber.

For long-form replay, this is a major efficiency gap.

Relevant files:

- `GEMINI.md`
- `ui/main_window.py`

### 5. Data model is still single-symbol / single-session oriented

Current assumptions are effectively:

- one instrument
- one session model
- one exchange/timezone behavior

There is no first-class handling for:

- `symbol`
- `session`
- `exchange`
- multi-contract or multi-asset datasets

This will block serious expansion beyond the current QDM/XAUUSD-style flow.

Relevant files:

- `tools/preprocess_qdm_tick_csv.py`
- `engine/data_validation.py`
- `engine/data_engine.py`
- `backtest/data.py`

### 6. Candle semantics are not unified across the product

There are still multiple candle-generation paths:

- preprocessing into DuckDB
- UI-side engine candle generation
- backtest-side aggregation
- replay-side state progression

These paths are similar, but not fully guaranteed to be identical in all cases.

That creates risk that:

- what the user sees
- what replay uses
- what backtest uses

may drift in edge cases.

Relevant files:

- `engine/data_engine.py`
- `tools/convert_parquet_to_duckdb.py`
- `backtest/data.py`

## P1: Should Be Improved Soon

### 7. Workspace restore is too weak

Current session restore does not recover:

- layout
- chart count
- pop-out window state
- timeframe mix
- replay mode state
- replay speed
- drawing state

So it restores a location, but not a working context.

Relevant files:

- `ui/session_state.py`
- `ui/main_window.py`

### 8. There is no multi-checkpoint / bookmark system

Only one saved state exists today.

Practical review work usually needs:

- multiple named checkpoints
- trade bookmarks
- setup bookmarks
- quick jump list

Relevant files:

- `ui/session_state.py`
- `ui/main_window.py`

### 9. Reporting is offline and strategy-oriented, not review-oriented

The current backtest/report layer is useful, but it is still:

- batch-oriented
- CSV-oriented
- strategy-summary oriented

It is not yet integrated into the review UI as:

- trade drilldown
- tag-based filtering
- screenshot-linked review records
- report export
- performance dashboard

Relevant files:

- `backtest/metrics.py`
- `backtest/run_ema_pullback.py`
- `backtest/results/*.csv`

### 10. Time jump precision is too coarse for tick review

Current time jump normalizes to minute granularity.

That is acceptable for some review flows, but weak for:

- tick-level replay
- event-level inspection
- exact stop/target reconstruction

Relevant files:

- `ui/main_window.py`
- `ui/time_navigation.py`

### 11. Default open position is not user-centric

For large datasets, the app currently drops the user near a fixed internal position instead of:

- last review point
- selected checkpoint
- dataset start
- a defined default session start

That adds friction when reopening data.

Relevant file:

- `ui/main_window.py`

### 12. Recent files / data source shortcuts are missing

The app still relies on repeated file selection through the load dialog.

That is functional, but inefficient for repeated review on the same datasets.

Relevant file:

- `ui/main_window.py`

## P1: Architecture-Limited Gaps

### 13. Data path is still full-load oriented

The current engine still leans on:

- loading full ticks into memory
- full-frame candle caches
- prefix slicing and resampling

This is the main reason large-data usage still has structural limits.

Relevant files:

- `engine/data_engine.py`
- `CONTEXT.md`

### 14. Rendering path still rebuilds visible state in Python

The current chart path still depends on:

- Python-managed visible slice refresh
- repeated `QPicture` candle regeneration
- repeated EMA/BB/MACD/RSI item updates

Recent optimizations improved it, but this is still not a true large-data workstation rendering model.

Relevant files:

- `ui/main_window.py`
- `ui/chart_performance.py`

### 15. Multi-window behavior is not yet a full workspace model

Current pop-out support is useful, but still based on reparenting the same widget rather than managing a richer workspace abstraction.

That limits:

- full workspace persistence
- richer cross-window coordination
- reproducible professional layouts

Relevant files:

- `ui/main_window.py`
- `ui/session_state.py`
- `ui/chart_performance.py`

## Existing Features That Need Completion

These features already exist, but are not yet complete enough:

- drawing tools
  - need persistence, editing, grouping, linking, export
- replay controls
  - need timeline scrubbing and bookmark integration
- Save View
  - needs to become Save Workspace / Save Review Record
- backtest output
  - needs UI integration and review-oriented drilldown
- data validation
  - needs stronger tick-order, symbol/session, and reproducibility constraints

## Recommended Development Order

### Phase 1: Product foundation

1. add review record model
2. persist drawings and bookmarks
3. add replay timeline scrubbing

### Phase 2: Review workflow closure

4. add trade-level marking workflow
5. upgrade Save View into workspace persistence
6. add multi-checkpoint / named bookmark system

### Phase 3: Data correctness foundation

7. promote `symbol/session/exchange` into the data model
8. unify candle semantics across preprocess/UI/replay/backtest
9. strengthen tick-level validation and reproducibility rules

### Phase 4: Scalability and professional UX

10. move toward windowed loading, stronger viewport caching, and better workspace architecture

## Short Version

If only the three highest-value next steps are chosen, the recommendation is:

1. review record model
2. persistent annotations and bookmarks
3. replay timeline scrubbing

Those three would shift TradeReview from “usable chart replay tool” toward “actual review platform”.

