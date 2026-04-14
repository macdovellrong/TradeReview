import argparse
from pathlib import Path

import duckdb
import pandas as pd

from convert_parquet_to_duckdb import _table_suffix_from_timeframe, build_candles


DEFAULT_PERIODS = "30s,1min,2min,3min,5min,10min,15min,20min,30min,45min,90min,1h,2h,3h,4h,6h,8h,12h,1D,1W,1M"


def _normalize_input_tz(input_tz):
    if input_tz is None:
        return None
    normalized = str(input_tz).strip()
    if not normalized or normalized.lower() == "none":
        return None
    return normalized


def load_qdm_ticks(csv_path, input_tz="America/New_York"):
    df = pd.read_csv(
        csv_path,
        usecols=["DateTime", "Bid", "Volume"],
        dtype={"Bid": "float64", "Volume": "float64"},
    )
    df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y%m%d %H:%M:%S.%f")
    df = df.rename(columns={"DateTime": "timestamp", "Bid": "price", "Volume": "volume"})
    df["volume"] = df["volume"].fillna(0.0)
    df = df.set_index("timestamp")

    target_tz = _normalize_input_tz(input_tz)
    if target_tz:
        if df.index.tz is None:
            df.index = df.index.tz_localize(target_tz)
        else:
            df.index = df.index.tz_convert(target_tz)

    df.sort_index(inplace=True)
    return df


def write_duckdb(df_ticks, db_path, periods):
    db_path = Path(db_path)
    ticks_naive = df_ticks.copy()
    ticks_naive.index = ticks_naive.index.tz_localize(None)
    ticks_naive = ticks_naive.rename_axis("timestamp").reset_index()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("PRAGMA threads=4")
        print("Writing ticks table ...")
        con.register("ticks_naive_df", ticks_naive)
        con.execute("CREATE OR REPLACE TABLE ticks AS SELECT * FROM ticks_naive_df")
        con.unregister("ticks_naive_df")

        for tf in periods:
            print(f"Building candles for {tf} ...")
            candles = build_candles(df_ticks, tf).reset_index()
            if "timestamp" not in candles.columns and "index" in candles.columns:
                candles = candles.rename(columns={"index": "timestamp"})
            table = f"candles_{_table_suffix_from_timeframe(tf)}"
            con.register("candles_df", candles)
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM candles_df")
            con.unregister("candles_df")
    finally:
        con.close()


def _resolve_output_path(raw_path, output_dir, stem, suffix):
    if raw_path:
        return Path(raw_path)
    return Path(output_dir) / f"{stem}{suffix}"


def _resolve_output_paths(csv_path, args):
    stem = args.stem or csv_path.stem
    output_dir = Path(args.output_dir)
    parquet_path = _resolve_output_path(args.parquet, output_dir, stem, ".parquet")
    db_path = _resolve_output_path(args.db, output_dir, stem, ".duckdb")
    return parquet_path, db_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess QDM tick CSV into parquet and DuckDB for TradeReview."
    )
    parser.add_argument("csv", help="QDM-exported tick CSV path")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Default output directory for generated parquet and DuckDB files",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Override output filename stem when using default output paths",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="Explicit parquet output path (defaults to <output-dir>/<stem>.parquet)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Explicit DuckDB output path (defaults to <output-dir>/<stem>.duckdb)",
    )
    parser.add_argument(
        "--periods",
        default=DEFAULT_PERIODS,
        help="Comma-separated candle periods",
    )
    parser.add_argument(
        "--input-tz",
        default="America/New_York",
        help="Timezone to assign to naive QDM timestamps; use 'none' to keep them naive",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        parser.error(f"CSV file not found: {csv_path}")

    parquet_path, db_path = _resolve_output_paths(csv_path, args)
    periods = [p.strip() for p in args.periods.split(",") if p.strip()]

    print(f"Loading QDM CSV from {csv_path} ...")
    df_ticks = load_qdm_ticks(csv_path, input_tz=args.input_tz)
    print(f"Loaded {len(df_ticks)} ticks.")
    print(f"Time range: {df_ticks.index[0]} -> {df_ticks.index[-1]}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing parquet to {parquet_path} ...")
    df_ticks.to_parquet(parquet_path, compression="zstd")

    print(f"Writing DuckDB to {db_path} ...")
    write_duckdb(df_ticks, db_path, periods)

    print(f"Parquet saved at {parquet_path}")
    print(f"DuckDB saved at {db_path}")
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
