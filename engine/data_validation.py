from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TICK_TIMESTAMP_ALIASES = ("timestamp", "datetime", "time")
REQUIRED_TICK_COLUMNS = ("price", "volume")
REQUIRED_CANDLE_COLUMNS = ("open", "high", "low", "close", "volume")
CURRENT_CANDLE_INDICATOR_COLUMNS = (
    "EMA20",
    "EMA30",
    "EMA40",
    "EMA50",
    "EMA60",
    "EMA100",
    "EMA240",
    "BB_Upper",
    "BB_Lower",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "RSI6",
    "RSI12",
    "RSI24",
)


class DataValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DuckDBValidationReport:
    tables: set[str]
    candle_tables: set[str]


def _canonicalize_columns(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    lowered = {str(col).lower(): col for col in df.columns}
    for target, names in aliases.items():
        for name in names:
            original = lowered.get(name.lower())
            if original is not None and original != target:
                rename_map[original] = target
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _require_columns(columns, required, source: str) -> None:
    lowered = {str(col).lower() for col in columns}
    missing = [name for name in required if name.lower() not in lowered]
    if missing:
        raise DataValidationError(f"{source} is missing required columns: {', '.join(missing)}")


def _require_any_column(columns, aliases, display_name: str, source: str) -> None:
    lowered = {str(col).lower() for col in columns}
    if not any(name.lower() in lowered for name in aliases):
        raise DataValidationError(f"{source} is missing required column: {display_name}")


def inspect_duckdb_schema(con, source: str) -> DuckDBValidationReport:
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    lowered_tables = {name.lower() for name in tables}
    if "ticks" not in lowered_tables:
        raise DataValidationError(f"{source} is missing required table: ticks")

    tick_columns = [row[0] for row in con.execute("DESCRIBE ticks").fetchall()]
    _require_any_column(tick_columns, TICK_TIMESTAMP_ALIASES, "timestamp", f"{source}::ticks")
    _require_columns(tick_columns, REQUIRED_TICK_COLUMNS, f"{source}::ticks")

    candle_tables = {name for name in tables if name.lower().startswith("candles_")}
    for table_name in candle_tables:
        columns = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
        _require_any_column(columns, TICK_TIMESTAMP_ALIASES, "timestamp", f"{source}::{table_name}")
        _require_columns(columns, REQUIRED_CANDLE_COLUMNS, f"{source}::{table_name}")

    return DuckDBValidationReport(
        tables=tables,
        candle_tables=candle_tables,
    )


def validate_duckdb_ticks_table(con, source: str, table_name: str = "ticks") -> None:
    row_count, null_ts, bad_price_rows, bad_volume_rows, min_ts, max_ts = con.execute(
        f"""
        SELECT
            count(*) AS row_count,
            sum(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END) AS null_ts,
            sum(CASE WHEN price IS NULL OR price <= 0 THEN 1 ELSE 0 END) AS bad_price_rows,
            sum(CASE WHEN volume IS NULL OR volume < 0 THEN 1 ELSE 0 END) AS bad_volume_rows,
            min(timestamp) AS min_ts,
            max(timestamp) AS max_ts
        FROM {table_name}
        """
    ).fetchone()

    if row_count is None or row_count <= 0:
        raise DataValidationError(f"{source}::{table_name} has no tick rows")
    if null_ts:
        raise DataValidationError(f"{source}::{table_name} contains NULL timestamps")
    if bad_price_rows:
        raise DataValidationError(f"{source}::{table_name} contains invalid prices")
    if bad_volume_rows:
        raise DataValidationError(f"{source}::{table_name} contains invalid volumes")
    if min_ts is None or max_ts is None:
        raise DataValidationError(f"{source}::{table_name} has an invalid time range")


def validate_duckdb_candle_table(con, source: str, table_name: str, allow_gap_rows: bool = False) -> None:
    row_count, null_ts, invalid_rows, partial_rows, populated_rows = con.execute(
        f"""
        SELECT
            count(*) AS row_count,
            sum(CASE WHEN timestamp IS NULL THEN 1 ELSE 0 END) AS null_ts,
            sum(
                CASE
                    WHEN open IS NOT NULL AND (
                        high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL OR
                        open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 OR volume < 0 OR
                        high < low OR open > high OR open < low OR close > high OR close < low
                    )
                    THEN 1 ELSE 0
                END
            ) AS invalid_rows,
            sum(
                CASE
                    WHEN open IS NULL AND (
                        high IS NOT NULL OR low IS NOT NULL OR close IS NOT NULL OR volume IS NOT NULL
                    )
                    THEN 1 ELSE 0
                END
            ) AS partial_rows,
            sum(CASE WHEN open IS NOT NULL THEN 1 ELSE 0 END) AS populated_rows
        FROM {table_name}
        """
    ).fetchone()

    if row_count is None or row_count <= 0:
        raise DataValidationError(f"{source}::{table_name} has no candle rows")
    if null_ts:
        raise DataValidationError(f"{source}::{table_name} contains NULL timestamps")
    if invalid_rows:
        raise DataValidationError(f"{source}::{table_name} contains invalid OHLCV rows")
    if partial_rows:
        raise DataValidationError(f"{source}::{table_name} contains partially populated gap rows")
    if populated_rows is None or populated_rows <= 0:
        raise DataValidationError(f"{source}::{table_name} has no populated candles")

    if not allow_gap_rows:
        null_required = con.execute(
            f"""
            SELECT
                sum(CASE WHEN open IS NULL THEN 1 ELSE 0 END) +
                sum(CASE WHEN high IS NULL THEN 1 ELSE 0 END) +
                sum(CASE WHEN low IS NULL THEN 1 ELSE 0 END) +
                sum(CASE WHEN close IS NULL THEN 1 ELSE 0 END) +
                sum(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS null_required
            FROM {table_name}
            """
        ).fetchone()[0]
        if null_required:
            raise DataValidationError(f"{source}::{table_name} contains NULL OHLCV values")

    duplicate_rows = con.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT timestamp
            FROM {table_name}
            GROUP BY 1
            HAVING count(*) > 1
            LIMIT 1
        )
        """
    ).fetchone()[0]
    if duplicate_rows:
        raise DataValidationError(f"{source}::{table_name} contains duplicate timestamps")


def normalize_tick_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = _canonicalize_columns(
        df.copy(),
        {
            "timestamp": TICK_TIMESTAMP_ALIASES,
            "price": ("price", "bid", "close"),
            "volume": ("volume", "vol"),
        },
    )

    _require_columns(df.columns, REQUIRED_TICK_COLUMNS, source)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df.index.name = "timestamp"
    else:
        raise DataValidationError(f"{source} is missing a DatetimeIndex or timestamp column")

    if df.empty:
        raise DataValidationError(f"{source} contains no tick rows")
    if df.index.hasnans:
        raise DataValidationError(f"{source} contains invalid timestamps")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    if df["price"].isna().any() or (df["price"] <= 0).any():
        raise DataValidationError(f"{source} contains invalid prices")
    if df["volume"].isna().any() or (df["volume"] < 0).any():
        raise DataValidationError(f"{source} contains invalid volumes")

    return df.sort_index()


def normalize_candle_dataframe(df: pd.DataFrame, source: str, allow_gap_rows: bool = False) -> pd.DataFrame:
    df = _canonicalize_columns(
        df.copy(),
        {
            "timestamp": TICK_TIMESTAMP_ALIASES,
            "open": ("open",),
            "high": ("high",),
            "low": ("low",),
            "close": ("close",),
            "volume": ("volume",),
        },
    )
    _require_columns(df.columns, REQUIRED_CANDLE_COLUMNS, source)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
        df.index.name = "timestamp"
    else:
        raise DataValidationError(f"{source} is missing a DatetimeIndex or timestamp column")

    if df.empty:
        raise DataValidationError(f"{source} contains no candle rows")
    if df.index.hasnans:
        raise DataValidationError(f"{source} contains invalid timestamps")
    if not df.index.is_unique:
        raise DataValidationError(f"{source} contains duplicate candle timestamps")

    required_cols = list(REQUIRED_CANDLE_COLUMNS)
    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")

    populated = df["open"].notna()
    if not populated.any():
        raise DataValidationError(f"{source} contains no populated candles")

    if allow_gap_rows:
        if df.loc[~populated, ["high", "low", "close", "volume"]].notna().any(axis=1).any():
            raise DataValidationError(f"{source} contains partially populated gap rows")
        work = df.loc[populated, required_cols]
    else:
        if df[required_cols].isna().any(axis=1).any():
            raise DataValidationError(f"{source} contains NULL OHLCV values")
        work = df[required_cols]

    if (work[["open", "high", "low", "close"]] <= 0).any(axis=1).any():
        raise DataValidationError(f"{source} contains non-positive OHLC values")
    if (work["volume"] < 0).any():
        raise DataValidationError(f"{source} contains negative volumes")

    bad_ranges = (
        (work["high"] < work["low"])
        | (work["open"] > work["high"])
        | (work["open"] < work["low"])
        | (work["close"] > work["high"])
        | (work["close"] < work["low"])
    )
    if bad_ranges.any():
        raise DataValidationError(f"{source} contains inconsistent OHLC ranges")

    return df.sort_index()
