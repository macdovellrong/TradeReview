# Chat Context Snapshot (2026-01-24 19:23:59)
## Summary
- Added MACD and RSI panels under each chart; MACD histogram uses red (above 0) and light blue (below 0); RSI guide lines set to 20/80.
- Added OHLC + RSI info display at cursor in top-left of main chart.
- Added vertical cursor lines for MACD/RSI panels synced with main chart.
- Added Ctrl+LMB measurement overlay and Ctrl+wheel Y-only zoom; Ctrl+drag prevents panning.
- Replaced replay slider with Back/Forward step controls.
- Implemented tick-driven replay engine with auto-skip for no-tick gaps.
- Added market calendar filtering to remove weekend/holiday gaps.
- Added DuckDB conversion script for precomputed multi-period candles + indicators (RSI6/12/24).
## New Script
- tools/convert_parquet_to_duckdb.py
  - Precomputes candles and indicators into a single DuckDB file.
  - Periods default: 1min,5min,15min,30min,1h,4h,1D.
  - Indicators include EMA, BB, MACD, RSI6/12/24.
## Notes
- QDM data already in America/New_York with DST applied.
- Replay now tick-driven; UI refresh still every timer tick.
## Recent Commits
- 43b6195 Add MACD/RSI panels and UI tweaks
- 16041d6 Adjust RSI guide lines to 20/80
- 8263639 Sync cursor line to MACD/RSI panels
- 8b85178 Show OHLC and RSI info at cursor
- 1bbc504 Adjust mouse interactions for measurement and Y zoom
- a84461d Replace replay slider with step controls
- ea9a8ff Implement tick-driven replay engine
