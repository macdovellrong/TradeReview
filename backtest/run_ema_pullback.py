from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.data import DuckDBMarketData, locate_duckdb, normalize_timeframe
from backtest.metrics import summarize_trades
from backtest.strategy import StrategyConfig, run_ema_pullback_backtest


def _parse_csv_arg(raw: str, cast=int) -> list:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(cast(part))
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest first-touch EMA20/EMA60 pullback setups.")
    parser.add_argument("--db", default=None, help="DuckDB path. Defaults to latest file under data/.")
    parser.add_argument("--timeframes", default="1m,5m,15m", help="Comma-separated list: 1m,5m,15m")
    parser.add_argument("--ema-spans", default="20,60", help="Comma-separated EMA spans")
    parser.add_argument("--directions", default="long,short", help="Comma-separated directions")
    parser.add_argument("--target-rr", type=float, default=1.5, help="Target reward/risk multiple")
    parser.add_argument("--stop-atr", type=float, default=0.75, help="ATR multiple for baseline stop")
    parser.add_argument("--max-hold-bars", type=int, default=12, help="Maximum holding bars")
    parser.add_argument("--min-separation-atr", type=float, default=0.25, help="Min EMA20/EMA60 distance in ATR")
    parser.add_argument("--min-extension-atr", type=float, default=0.75, help="Min move away from target EMA before the first touch")
    parser.add_argument("--output-dir", default="backtest/results", help="Directory for CSV outputs")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else locate_duckdb("data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timeframes = [normalize_timeframe(item) for item in _parse_csv_arg(args.timeframes, cast=str)]
    ema_spans = _parse_csv_arg(args.ema_spans, cast=int)
    directions = [item.strip().lower() for item in _parse_csv_arg(args.directions, cast=str)]

    market_data = DuckDBMarketData(db_path)
    try:
        trade_frames: list[pd.DataFrame] = []
        for timeframe in timeframes:
            print(f"Loading candles for {timeframe} from {db_path.name} ...")
            candles = market_data.load_candles(timeframe)
            print(f"  candles: {len(candles):,}")
            for ema_span in ema_spans:
                for direction in directions:
                    config = StrategyConfig(
                        timeframe=timeframe,
                        ema_span=ema_span,
                        direction=direction,
                        target_rr=args.target_rr,
                        stop_atr=args.stop_atr,
                        max_hold_bars=args.max_hold_bars,
                        min_separation_atr=args.min_separation_atr,
                        min_extension_atr=args.min_extension_atr,
                    )
                    print(
                        f"Running {timeframe} EMA{ema_span} {direction} "
                        f"(RR={args.target_rr}, stopATR={args.stop_atr}) ..."
                    )
                    trades = run_ema_pullback_backtest(
                        candles=candles,
                        config=config,
                        market_data=market_data,
                    )
                    if trades.empty:
                        continue
                    trade_frames.append(trades)

        all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        summary = summarize_trades(all_trades, ["timeframe", "ema_span", "direction"])
        hourly = summarize_trades(all_trades, ["timeframe", "ema_span", "direction", "entry_hour"])

        trades_path = output_dir / "ema_pullback_trades.csv"
        summary_path = output_dir / "ema_pullback_summary.csv"
        hourly_path = output_dir / "ema_pullback_hourly.csv"

        all_trades.to_csv(trades_path, index=False)
        summary.to_csv(summary_path, index=False)
        hourly.to_csv(hourly_path, index=False)

        print("\nSummary")
        if summary.empty:
            print("No trades found with the current rules.")
        else:
            display_cols = [
                "timeframe",
                "ema_span",
                "direction",
                "trades",
                "win_rate",
                "avg_r",
                "profit_factor",
                "avg_bars_held",
            ]
            with pd.option_context("display.max_rows", None, "display.width", 140):
                print(summary[display_cols].to_string(index=False))

        print(f"\nSaved trades to   {trades_path}")
        print(f"Saved summary to  {summary_path}")
        print(f"Saved hourly to   {hourly_path}")
    finally:
        market_data.close()


if __name__ == "__main__":
    main()
