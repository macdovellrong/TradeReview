import pandas as pd


def _period_seconds(period):
    tf = str(period).strip().lower()
    try:
        if tf.endswith("min"):
            return int(tf[:-3]) * 60
        if tf.endswith("s"):
            return int(tf[:-1])
        if tf.endswith("h"):
            return int(tf[:-1]) * 60 * 60
        if tf.endswith("d"):
            return int(tf[:-1]) * 24 * 60 * 60
        if tf.endswith("w"):
            return int(tf[:-1]) * 7 * 24 * 60 * 60
        if tf.endswith("m") and not tf.endswith("min"):
            return int(tf[:-1]) * 30 * 24 * 60 * 60
    except ValueError:
        return None
    return None


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
        lod_period = "5min"
    elif span <= pd.Timedelta(days=180):
        lod_period = "1h"
    elif span <= pd.Timedelta(days=730):
        lod_period = "4h"
    else:
        lod_period = "1D"

    requested_seconds = _period_seconds(requested_period)
    lod_seconds = _period_seconds(lod_period)
    if requested_seconds is not None and lod_seconds is not None and requested_seconds >= lod_seconds:
        return requested_period
    return lod_period
