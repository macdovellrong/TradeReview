import pandas as pd


def choose_lod_period(
    requested_period,
    view_start,
    view_end,
    pixel_width=1600,
    max_points_per_pixel=2.0,
):
    span = pd.Timestamp(view_end) - pd.Timestamp(view_start)
    if span <= pd.Timedelta(days=2):
        return requested_period
    if span <= pd.Timedelta(days=30):
        return "5min"
    if span <= pd.Timedelta(days=180):
        return "1h"
    if span <= pd.Timedelta(days=730):
        return "4h"
    return "1D"
