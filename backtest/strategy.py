from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from backtest.data import DuckDBMarketData, normalize_timeframe


@dataclass(slots=True)
class StrategyConfig:
    timeframe: str
    ema_span: int
    direction: str
    trend_fast_span: int = 20
    trend_slow_span: int = 60
    atr_period: int = 14
    slope_lookback: int = 3
    min_separation_atr: float = 0.25
    min_extension_atr: float = 0.75
    touch_tolerance_atr: float = 0.05
    stop_atr: float = 0.75
    signal_stop_buffer_atr: float = 0.05
    target_rr: float = 1.5
    max_hold_bars: int = 12
    stop_mode: str = "atr_or_signal"

    def normalized(self) -> "StrategyConfig":
        params = asdict(self)
        params["timeframe"] = normalize_timeframe(self.timeframe)
        params["direction"] = self.direction.lower()
        if params["direction"] not in {"long", "short"}:
            raise ValueError(f"Unsupported direction: {self.direction}")
        if self.ema_span not in {20, 60}:
            raise ValueError("Only EMA20 and EMA60 are supported")
        return StrategyConfig(**params)


def run_ema_pullback_backtest(
    candles: pd.DataFrame,
    config: StrategyConfig,
    market_data: DuckDBMarketData | None = None,
) -> pd.DataFrame:
    cfg = config.normalized()
    timeframe_delta = pd.to_timedelta(cfg.timeframe)
    df = candles.copy()
    df = df.dropna(subset=["open", "high", "low", "close", "EMA20", "EMA60", "ATR14"])
    if df.empty:
        return pd.DataFrame()

    index = df.index.to_list()
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = df["ATR14"].to_numpy()
    fast = df[f"EMA{cfg.trend_fast_span}"].to_numpy()
    slow = df[f"EMA{cfg.trend_slow_span}"].to_numpy()
    target = df[f"EMA{cfg.ema_span}"].to_numpy()

    start_idx = max(cfg.trend_slow_span + cfg.slope_lookback, cfg.atr_period + 5)
    trades: list[dict[str, object]] = []
    armed = False
    armed_since = -1
    i = start_idx
    n = len(df)

    while i < n - 1:
        if not _trend_ok(close, fast, slow, atr, i, cfg):
            armed = False
            i += 1
            continue

        extension = _extension_ratio(high, low, target, atr, i, cfg.direction)
        if (not armed) and extension >= cfg.min_extension_atr:
            armed = True
            armed_since = i
            i += 1
            continue

        if armed and i > armed_since and _touched_target(close, high, low, target, atr, i, cfg):
            entry_idx = i + 1
            trade = _simulate_trade(
                index=index,
                open_=open_,
                high=high,
                low=low,
                close=close,
                atr=atr,
                target=target,
                signal_idx=i,
                entry_idx=entry_idx,
                timeframe_delta=timeframe_delta,
                config=cfg,
                market_data=market_data,
            )
            if trade is not None:
                trades.append(trade)
                i = trade["exit_idx"] + 1
                armed = False
                continue
            armed = False
        i += 1

    if not trades:
        return pd.DataFrame()
    trades_df = pd.DataFrame(trades).drop(columns=["exit_idx"])
    return trades_df.sort_values("entry_time").reset_index(drop=True)


def _trend_ok(close, fast, slow, atr, i: int, cfg: StrategyConfig) -> bool:
    if atr[i] <= 0:
        return False
    fast_slope = fast[i] - fast[i - cfg.slope_lookback]
    slow_slope = slow[i] - slow[i - cfg.slope_lookback]
    sep_atr = abs(fast[i] - slow[i]) / atr[i]
    if sep_atr < cfg.min_separation_atr:
        return False
    if cfg.direction == "long":
        return fast[i] > slow[i] and fast_slope > 0 and slow_slope >= 0 and close[i] >= slow[i]
    return fast[i] < slow[i] and fast_slope < 0 and slow_slope <= 0 and close[i] <= slow[i]


def _extension_ratio(high, low, target, atr, i: int, direction: str) -> float:
    if direction == "long":
        return (high[i] - target[i]) / atr[i]
    return (target[i] - low[i]) / atr[i]


def _touched_target(close, high, low, target, atr, i: int, cfg: StrategyConfig) -> bool:
    tol = cfg.touch_tolerance_atr * atr[i]
    if cfg.direction == "long":
        return low[i] <= target[i] + tol and close[i] >= target[i]
    return high[i] >= target[i] - tol and close[i] <= target[i]


def _simulate_trade(
    *,
    index,
    open_,
    high,
    low,
    close,
    atr,
    target,
    signal_idx: int,
    entry_idx: int,
    timeframe_delta: pd.Timedelta,
    config: StrategyConfig,
    market_data: DuckDBMarketData | None,
) -> dict[str, object] | None:
    if entry_idx >= len(index):
        return None
    entry_price = float(open_[entry_idx])
    atr_entry = float(atr[signal_idx])
    if atr_entry <= 0:
        return None

    if config.direction == "long":
        stop_atr_price = entry_price - config.stop_atr * atr_entry
        stop_signal_price = float(low[signal_idx] - config.signal_stop_buffer_atr * atr_entry)
        stop_price = min(stop_atr_price, stop_signal_price) if config.stop_mode == "atr_or_signal" else stop_atr_price
        risk = entry_price - stop_price
        target_price = entry_price + risk * config.target_rr
    else:
        stop_atr_price = entry_price + config.stop_atr * atr_entry
        stop_signal_price = float(high[signal_idx] + config.signal_stop_buffer_atr * atr_entry)
        stop_price = max(stop_atr_price, stop_signal_price) if config.stop_mode == "atr_or_signal" else stop_atr_price
        risk = stop_price - entry_price
        target_price = entry_price - risk * config.target_rr

    if risk <= 0:
        return None

    max_exit_idx = min(entry_idx + config.max_hold_bars - 1, len(index) - 1)
    exit_idx = max_exit_idx
    exit_price = float(close[max_exit_idx])
    exit_reason = "time_stop"
    outcome = "time_stop"

    for j in range(entry_idx, max_exit_idx + 1):
        hit_stop = low[j] <= stop_price if config.direction == "long" else high[j] >= stop_price
        hit_target = high[j] >= target_price if config.direction == "long" else low[j] <= target_price
        if not hit_stop and not hit_target:
            continue

        if hit_stop and hit_target:
            first_hit = "stop"
            if market_data is not None:
                first_hit = market_data.resolve_intrabar_order(
                    bar_start=index[j],
                    bar_end=index[j] + timeframe_delta,
                    direction=config.direction,
                    stop_price=stop_price,
                    target_price=target_price,
                )
            if first_hit == "target":
                exit_idx = j
                exit_price = target_price
                exit_reason = "target_hit_tick"
                outcome = "win"
            else:
                exit_idx = j
                exit_price = stop_price
                exit_reason = "stop_hit_tick"
                outcome = "loss"
            break

        if hit_target:
            exit_idx = j
            exit_price = target_price
            exit_reason = "target_hit"
            outcome = "win"
            break

        if hit_stop:
            exit_idx = j
            exit_price = stop_price
            exit_reason = "stop_hit"
            outcome = "loss"
            break

    if config.direction == "long":
        pnl = exit_price - entry_price
        mfe = float(high[entry_idx : exit_idx + 1].max() - entry_price)
        mae = float(low[entry_idx : exit_idx + 1].min() - entry_price)
    else:
        pnl = entry_price - exit_price
        mfe = float(entry_price - low[entry_idx : exit_idx + 1].min())
        mae = float(entry_price - high[entry_idx : exit_idx + 1].max())

    return {
        "timeframe": config.timeframe,
        "ema_span": config.ema_span,
        "direction": config.direction,
        "signal_time": index[signal_idx],
        "entry_time": index[entry_idx],
        "exit_time": index[exit_idx],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk": risk,
        "pnl": pnl,
        "pnl_r": pnl / risk,
        "mfe_r": mfe / risk,
        "mae_r": mae / risk,
        "bars_held": exit_idx - entry_idx + 1,
        "outcome": outcome,
        "exit_reason": exit_reason,
        "signal_atr": atr_entry,
        "signal_ema_price": float(target[signal_idx]),
        "entry_hour": index[entry_idx].hour,
        "exit_idx": exit_idx,
    }
