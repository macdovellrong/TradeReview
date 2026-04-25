import pandas as pd


def _span(view_start, view_end):
    span = pd.Timestamp(view_end) - pd.Timestamp(view_start)
    if span <= pd.Timedelta(0):
        return pd.Timedelta(minutes=1)
    return span


def build_query_window(view_start, view_end, buffer_multiplier=2):
    view_start = pd.Timestamp(view_start)
    view_end = pd.Timestamp(view_end)
    buffer = _span(view_start, view_end) * buffer_multiplier
    return view_start - buffer, view_end + buffer


def is_view_inside_loaded_window(view_start, view_end, loaded_start, loaded_end):
    return (
        pd.Timestamp(loaded_start) <= pd.Timestamp(view_start)
        and pd.Timestamp(view_end) <= pd.Timestamp(loaded_end)
    )


def should_prefetch_window(
    view_start,
    view_end,
    loaded_start,
    loaded_end,
    edge_fraction=0.5,
):
    view_start = pd.Timestamp(view_start)
    view_end = pd.Timestamp(view_end)
    loaded_start = pd.Timestamp(loaded_start)
    loaded_end = pd.Timestamp(loaded_end)
    margin = _span(view_start, view_end) * edge_fraction
    return view_start <= loaded_start + margin or view_end >= loaded_end - margin
