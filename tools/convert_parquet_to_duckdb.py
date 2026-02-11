import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd


def _should_dropna(timeframe):
    tf = str(timeframe).strip().lower()
    if tf.endswith("s"):
        try:
            int(tf[:-1])
        except ValueError:
            return True
        return False
    if tf.endswith("min"):
        try:
            minutes = int(tf[:-3])
        except ValueError:
            return True
        return minutes <= 60
    if tf.endswith("h"):
        try:
            int(tf[:-1])
        except ValueError:
            return True
        return True
    if tf.endswith("d") or tf.endswith("w") or (tf.endswith("m") and not tf.endswith("min")):
        return True
    return False


def _should_filter_by_calendar(timeframe):
    tf = str(timeframe).strip().lower()
    if tf.endswith("s") or tf.endswith("min") or tf.endswith("h"):
        return True
    return False


def _table_suffix_from_timeframe(timeframe):
    tf = str(timeframe).strip()
    tf_l = tf.lower()
    if tf_l.endswith("s") and tf_l[:-1].isdigit():
        return f"{tf_l[:-1]}s"
    if tf_l.endswith("min") and tf_l[:-3].isdigit():
        return f"{tf_l[:-3]}m"
    if tf_l.endswith("h") and tf_l[:-1].isdigit():
        return f"{tf_l[:-1]}h"
    if tf_l.endswith("d") and tf_l[:-1].isdigit():
        return f"{tf_l[:-1]}D"
    if tf_l.endswith("w") and tf_l[:-1].isdigit():
        return f"{tf_l[:-1]}W"
    if tf.endswith("M") and tf[:-1].isdigit():
        return f"{tf[:-1]}M"
    return tf


def _normalize_timeframe(timeframe):
    tf = str(timeframe).strip()
    # Pandas deprecates "M" for month-end; use "ME"
    if tf.endswith("M") and tf[:-1].isdigit():
        return f"{tf[:-1]}ME"
    return tf


def _build_full_index(start_ts, end_ts, timeframe, origin_ts):
    timeframe = _normalize_timeframe(timeframe)
    if start_ts > end_ts:
        return pd.DatetimeIndex([])
    base = pd.date_range(start=origin_ts, end=end_ts, freq=timeframe)
    if base.empty:
        return base
    pos = base.searchsorted(start_ts, side="right") - 1
    if pos < 0:
        start = base[0]
    else:
        start = base[pos]
    return pd.date_range(start=start, end=end_ts, freq=timeframe)


def _get_calendar():
    try:
        import pandas_market_calendars as mcal
    except Exception:
        return None
    for name in ["CME_FX", "CME"]:
        try:
            return mcal.get_calendar(name)
        except Exception:
            continue
    return None


def _filter_index_by_calendar(full_index, timeframe):
    if full_index is None or full_index.empty:
        return full_index

    cal = _get_calendar()
    if cal is None:
        return full_index

    try:
        offset = pd.tseries.frequencies.to_offset(timeframe)
        freq_delta = offset.delta
    except Exception:
        return full_index
    if freq_delta is None:
        return full_index

    start_date = (full_index[0] - pd.Timedelta(days=2)).date()
    end_date = (full_index[-1] + pd.Timedelta(days=2)).date()
    try:
        schedule = cal.schedule(start_date=start_date, end_date=end_date)
    except Exception:
        return full_index
    if schedule is None or schedule.empty:
        return full_index

    try:
        schedule = schedule.tz_convert("America/New_York")
    except Exception:
        schedule = schedule.tz_localize("America/New_York")

    opens = schedule["market_open"].dt.tz_localize(None)
    closes = schedule["market_close"].dt.tz_localize(None)

    mask = np.zeros(len(full_index), dtype=bool)
    for open_ts, close_ts in zip(opens, closes):
        start = open_ts - freq_delta
        end = close_ts
        left = full_index.searchsorted(start, side="left")
        right = full_index.searchsorted(end, side="left")
        if right > left:
            mask[left:right] = True

    return full_index[mask]


def _calculate_indicators(df):
    for span in [20, 30, 40, 50, 60, 100, 240]:
        df[f"EMA{span}"] = df["close"].ewm(span=span, adjust=False).mean()

    sma20 = df["close"].rolling(window=20).mean()
    std20 = df["close"].rolling(window=20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    for rsi_len in [6, 12, 24]:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / rsi_len, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / rsi_len, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"RSI{rsi_len}"] = 100 - (100 / (1 + rs))

    return df


def build_candles(df_ticks, timeframe):
    timeframe = _normalize_timeframe(timeframe)
    naive_index = df_ticks.index.tz_localize(None)
    price_series = pd.Series(df_ticks["price"].values, index=naive_index)
    vol_series = pd.Series(df_ticks["volume"].values, index=naive_index, name="volume")

    origin_ts = pd.Timestamp("2000-01-01 17:00:00")
    ohlc = price_series.resample(timeframe, closed="left", label="left", origin=origin_ts).ohlc()
    vol = vol_series.resample(timeframe, closed="left", label="left", origin=origin_ts).sum()

    df_candles = pd.concat([ohlc, vol], axis=1)
    if not _should_dropna(timeframe):
        full_index = _build_full_index(naive_index[0], naive_index[-1], timeframe, origin_ts)
        if _should_filter_by_calendar(timeframe):
            full_index = _filter_index_by_calendar(full_index, timeframe)
        df_candles = df_candles.reindex(full_index)

    if _should_dropna(timeframe):
        df_candles.dropna(inplace=True)

    df_candles = _calculate_indicators(df_candles)
    return df_candles


def main():
    parser = argparse.ArgumentParser(description="Convert tick parquet to DuckDB with precomputed candles.")
    parser.add_argument("parquet", help="Tick parquet file path")
    parser.add_argument("--db", default="data/candles.duckdb", help="Output DuckDB file path")
    parser.add_argument(
        "--periods",
        default="30s,1min,2min,3min,5min,10min,15min,20min,30min,45min,90min,1h,2h,3h,4h,6h,8h,12h,1D,1W,1M",
        help="Comma-separated periods",
    )
    args = parser.parse_args()

    parquet_path = args.parquet
    db_path = args.db
    periods = [p.strip() for p in args.periods.split(",") if p.strip()]

    print(f"Loading ticks from {parquet_path} ...")
    df_ticks = pd.read_parquet(parquet_path)
    if df_ticks.index.tz is None:
        df_ticks.index = df_ticks.index.tz_localize("America/New_York")
    else:
        df_ticks.index = df_ticks.index.tz_convert("America/New_York")
    df_ticks.sort_index(inplace=True)

    ticks_naive = df_ticks.copy()
    ticks_naive.index = ticks_naive.index.tz_localize(None)
    ticks_naive = ticks_naive.rename_axis("timestamp").reset_index()

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    import duckdb

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=4")

    print("Writing ticks table ...")
    con.register("ticks_naive_df", ticks_naive)
    con.execute("CREATE OR REPLACE TABLE ticks AS SELECT * FROM ticks_naive_df")
    con.unregister("ticks_naive_df")

    for tf in periods:
        print(f"Building candles for {tf} ...")
        candles = build_candles(df_ticks, tf)
        candles = candles.reset_index()
        if "timestamp" not in candles.columns and "index" in candles.columns:
            candles = candles.rename(columns={"index": "timestamp"})
        if "timestamp" not in candles.columns and "time" in candles.columns:
            candles = candles.rename(columns={"time": "timestamp"})
        if "timestamp" not in candles.columns and "datetime" in candles.columns:
            candles = candles.rename(columns={"datetime": "timestamp"})
        table = f"candles_{_table_suffix_from_timeframe(tf)}"
        con.register("candles_df", candles)
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM candles_df")
        con.unregister("candles_df")

    con.close()
    print(f"Done. DuckDB saved at {db_path}")


if __name__ == "__main__":
    main()
