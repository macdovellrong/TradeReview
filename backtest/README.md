# EMA Pullback Backtest

This directory contains a standalone intraday backtest around first-touch EMA pullbacks.

Default strategy rules:

- Trend filter:
  - long: `EMA20 > EMA60`
  - short: `EMA20 < EMA60`
  - both EMAs must slope in the trend direction over the last 3 bars
  - `abs(EMA20 - EMA60) / ATR14 >= 0.25`
- Setup arming:
  - price must extend away from the target EMA by at least `0.75 ATR14`
  - after that extension, only the first touch is tradable
- Entry:
  - signal bar touches `EMA20` or `EMA60`
  - signal bar closes back on the trend side of the touched EMA
  - enter at the next bar open
- Exit:
  - stop uses the farther of:
    - `0.75 ATR14`
    - signal bar extreme plus a `0.05 ATR14` buffer
  - target is `1.5R`
  - time stop after 12 bars
  - if a bar hits both stop and target, the module queries the `ticks` table to decide which was touched first

Run:

```powershell
& .\.venv\Scripts\python.exe .\backtest\run_ema_pullback.py
```

Useful flags:

```powershell
& .\.venv\Scripts\python.exe .\backtest\run_ema_pullback.py --timeframes 5m,15m --target-rr 2.0 --stop-atr 1.0
```

Outputs are written to `backtest/results/`:

- `ema_pullback_trades.csv`
- `ema_pullback_summary.csv`
- `ema_pullback_hourly.csv`
