def build_visible_slice_window(view_min, view_max, total_len, padding=1000):
    slice_start = max(0, int(view_min) - padding)
    slice_end = min(total_len, int(view_max) + padding)
    return slice_start, slice_end


def should_refresh_visible_slice(
    view_min,
    view_max,
    total_len,
    last_slice_start,
    last_slice_end,
    padding=1000,
):
    if total_len <= 0:
        return False
    if last_slice_start < 0 or last_slice_end <= last_slice_start:
        return True

    if view_max < last_slice_start or view_min > last_slice_end:
        return True

    edge_margin = max(100, padding // 4)

    near_left_edge = last_slice_start > 0 and view_min <= (last_slice_start + edge_margin)
    near_right_edge = last_slice_end < total_len and view_max >= (last_slice_end - edge_margin)
    return near_left_edge or near_right_edge


def get_crosshair_sync_targets(charts, source_chart):
    return [chart for chart in charts if chart is not source_chart and not getattr(chart, "is_detached", False)]
