from __future__ import annotations

import pandas as pd


def normalize_jump_timestamp(value) -> pd.Timestamp:
    return pd.Timestamp(value).floor("min")


def clamp_timestamp(value, start=None, end=None) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if start is not None and ts < start:
        return pd.Timestamp(start)
    if end is not None and ts > end:
        return pd.Timestamp(end)
    return ts


def resolve_chart_target(df: pd.DataFrame, target_dt) -> tuple[int | None, float | None]:
    if df is None or df.empty:
        return None, None

    ts = pd.Timestamp(target_dt)
    if df.index.tz is None and ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    elif df.index.tz is not None and ts.tzinfo is None:
        ts = ts.tz_localize(df.index.tz)

    idx = int(df.index.searchsorted(ts))
    if idx < 0:
        idx = 0
    if idx >= len(df):
        idx = len(df) - 1

    close = float(df["close"].iloc[idx]) if "close" in df.columns else None
    return idx, close
