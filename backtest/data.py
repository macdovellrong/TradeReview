from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from engine.data_validation import (
    inspect_duckdb_schema,
    normalize_candle_dataframe,
    validate_duckdb_ticks_table,
)


_SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m"}
_ANCHOR_TS = "2000-01-01 17:00:00"


def normalize_timeframe(timeframe: str) -> str:
    tf = str(timeframe).strip().lower()
    mapping = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
    }
    tf = mapping.get(tf, tf)
    if tf not in _SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return tf


def locate_duckdb(data_dir: str | Path = "data") -> Path:
    data_path = Path(data_dir)
    candidates = sorted(data_path.glob("*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No DuckDB file found under {data_path}")
    return candidates[0]


@dataclass
class DuckDBMarketData:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.con = duckdb.connect(str(self.db_path), read_only=True)
        try:
            report = inspect_duckdb_schema(self.con, str(self.db_path))
            validate_duckdb_ticks_table(self.con, str(self.db_path))
            self._tables = report.tables
        except Exception:
            self.con.close()
            raise

    def close(self) -> None:
        self.con.close()

    def load_candles(self, timeframe: str) -> pd.DataFrame:
        tf = normalize_timeframe(timeframe)
        table_name = f"candles_{tf}"
        if table_name in self._tables:
            df = self.con.execute(f"SELECT * FROM {table_name} ORDER BY timestamp").fetchdf()
            df = normalize_candle_dataframe(
                df,
                f"{self.db_path}::{table_name}",
                allow_gap_rows=table_name.endswith("s"),
            )
            df = df[df["open"].notna()]
        else:
            df = self._aggregate_from_ticks(tf)
            df = normalize_candle_dataframe(df, f"{self.db_path}::aggregated_{table_name}")
        return self._ensure_indicators(df)

    def resolve_intrabar_order(
        self,
        bar_start: pd.Timestamp,
        bar_end: pd.Timestamp,
        direction: str,
        stop_price: float,
        target_price: float,
    ) -> str:
        query = """
        SELECT
            min(CASE WHEN price <= ? THEN timestamp END) AS stop_ts,
            min(CASE WHEN price >= ? THEN timestamp END) AS target_ts
        FROM ticks
        WHERE timestamp >= ? AND timestamp < ?
        """
        if direction == "short":
            query = """
            SELECT
                min(CASE WHEN price >= ? THEN timestamp END) AS stop_ts,
                min(CASE WHEN price <= ? THEN timestamp END) AS target_ts
            FROM ticks
            WHERE timestamp >= ? AND timestamp < ?
            """
        stop_ts, target_ts = self.con.execute(
            query,
            [
                stop_price,
                target_price,
                bar_start.to_pydatetime(),
                bar_end.to_pydatetime(),
            ],
        ).fetchone()
        if stop_ts is None and target_ts is None:
            return "stop"
        if stop_ts is None:
            return "target"
        if target_ts is None:
            return "stop"
        return "stop" if stop_ts <= target_ts else "target"

    def _aggregate_from_ticks(self, timeframe: str) -> pd.DataFrame:
        interval_lookup = {"1m": "1 minute", "5m": "5 minutes", "15m": "15 minutes"}
        interval = interval_lookup[timeframe]
        query = f"""
        SELECT
            time_bucket(INTERVAL '{interval}', timestamp, TIMESTAMP '{_ANCHOR_TS}') AS timestamp,
            arg_min(price, timestamp) AS open,
            max(price) AS high,
            min(price) AS low,
            arg_max(price, timestamp) AS close,
            sum(volume) AS volume
        FROM ticks
        GROUP BY 1
        ORDER BY 1
        """
        return self.con.execute(query).fetchdf()

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "EMA20" not in df.columns:
            df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
        if "EMA60" not in df.columns:
            df["EMA60"] = df["close"].ewm(span=60, adjust=False).mean()
        if "ATR14" not in df.columns:
            prev_close = df["close"].shift(1)
            true_range = pd.concat(
                [
                    df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            df["ATR14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
        return df
