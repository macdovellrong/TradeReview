from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def summarize_trades(trades: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    group_cols = list(group_cols)
    if trades.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "trades",
                "win_rate",
                "avg_r",
                "median_r",
                "expectancy_r",
                "profit_factor",
                "avg_mfe_r",
                "avg_mae_r",
                "avg_bars_held",
            ]
        )

    rows: list[dict[str, float | str | int]] = []
    for keys, group in trades.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        wins = group["pnl_r"] > 0
        loss_r_sum = group.loc[group["pnl_r"] < 0, "pnl_r"].abs().sum()
        profit_factor = np.nan
        if loss_r_sum > 0:
            profit_factor = group.loc[group["pnl_r"] > 0, "pnl_r"].sum() / loss_r_sum
        rows.append(
            {
                **key_map,
                "trades": int(len(group)),
                "win_rate": float(wins.mean()),
                "avg_r": float(group["pnl_r"].mean()),
                "median_r": float(group["pnl_r"].median()),
                "expectancy_r": float(group["pnl_r"].mean()),
                "profit_factor": float(profit_factor) if profit_factor == profit_factor else np.nan,
                "avg_mfe_r": float(group["mfe_r"].mean()),
                "avg_mae_r": float(group["mae_r"].mean()),
                "avg_bars_held": float(group["bars_held"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
