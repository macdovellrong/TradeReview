from __future__ import annotations

import pyqtgraph as pg
from PyQt6.QtCore import Qt

from ui.drawings.fib_math import build_extension_levels, build_retracement_levels
from ui.drawings.specs import normalize_drawing_spec


def build_render_plan(spec: dict) -> dict:
    normalized = normalize_drawing_spec(spec)
    levels = list(normalized.get("config_snapshot", {}).get("levels", []))
    points = normalized["points"]
    rows = []

    if normalized["type"] == "fib" and len(points) >= 2:
        rows = [
            {"ratio": level.ratio, "price": level.price}
            for level in build_retracement_levels(points[0]["price"], points[1]["price"], levels)
        ]
    elif normalized["type"] == "fib_ext" and len(points) >= 3:
        rows = [
            {"ratio": level.ratio, "price": level.price}
            for level in build_extension_levels(
                points[0]["price"],
                points[1]["price"],
                points[2]["price"],
                levels,
            )
        ]

    return {
        "id": normalized.get("id"),
        "type": normalized["type"],
        "points": points,
        "levels": levels,
        "rows": rows,
    }


def _tag_drawing_item(item, drawing_id: int | None) -> None:
    item._is_drawing = True
    item._drawing_id = drawing_id


def _line_pen(preview: bool = False, dashed: bool = False, color: str | None = None, width: float = 2.0):
    pen_color = color or ("#00E5FF" if preview else "#FFD54A")
    style = Qt.PenStyle.DashLine if dashed or preview else Qt.PenStyle.SolidLine
    return pg.mkPen(pen_color, width=width, style=style)


def render_spec_items(plot_item, spec: dict, x_from_datetime, preview: bool = False) -> list[object]:
    normalized = normalize_drawing_spec(spec)
    drawing_id = normalized.get("id")
    points = normalized["points"]
    dtype = normalized["type"]
    items: list[object] = []

    def add_item(item, ignore_bounds: bool = False):
        _tag_drawing_item(item, drawing_id)
        plot_item.addItem(item, ignoreBounds=ignore_bounds)
        items.append(item)

    if dtype == "hline" and len(points) >= 1:
        line = pg.InfiniteLine(angle=0, pos=points[0]["price"], pen=_line_pen(preview=False, color="#FF4444", width=1.0))
        add_item(line)
        return items

    if dtype == "vline" and len(points) >= 1:
        x = x_from_datetime(points[0]["dt"])
        if x is None:
            return []
        line = pg.InfiniteLine(angle=90, pos=x, pen=_line_pen(preview=False, color="#FF4444", width=1.0))
        add_item(line)
        return items

    if dtype == "line" and len(points) >= 2:
        x_values = [x_from_datetime(point["dt"]) for point in points[:2]]
        if any(x is None for x in x_values):
            return []
        curve = pg.PlotCurveItem(
            x=x_values,
            y=[points[0]["price"], points[1]["price"]],
            pen=_line_pen(preview=preview, color="#00E5FF", width=2.5 if not preview else 2.0),
        )
        add_item(curve)
        return items

    plan = build_render_plan(normalized)

    if dtype == "fib" and len(points) >= 2:
        x_values = [x_from_datetime(point["dt"]) for point in points[:2]]
        if any(x is None for x in x_values):
            return []
        x_left = min(x_values)
        x_right = max(x_values)

        for point in points[:2]:
            edge = pg.PlotCurveItem(
                x=[x_left, x_right],
                y=[point["price"], point["price"]],
                pen=_line_pen(preview=preview, dashed=False, width=2.0),
            )
            add_item(edge)

        for row in plan["rows"]:
            level = pg.PlotCurveItem(
                x=[x_left, x_right],
                y=[row["price"], row["price"]],
                pen=_line_pen(preview=preview, dashed=True, width=2.0),
            )
            add_item(level)
            if preview:
                continue
            label = pg.TextItem(
                text=f'{row["ratio"]:.3f}  {row["price"]:.3f}',
                color="#FFD54A",
                fill=pg.mkBrush(20, 20, 20, 220),
                anchor=(0, 0.5),
            )
            label.setPos(x_right, row["price"])
            label.setZValue(21)
            add_item(label, ignore_bounds=True)
        return items

    if dtype == "fib_ext" and len(points) >= 3:
        x_a = x_from_datetime(points[0]["dt"])
        x_b = x_from_datetime(points[1]["dt"])
        x_c = x_from_datetime(points[2]["dt"])
        if any(x is None for x in (x_a, x_b, x_c)):
            return []

        guide_pen = _line_pen(preview=preview, dashed=True, color="#B38F00", width=1.6)
        ab = pg.PlotCurveItem(x=[x_a, x_b], y=[points[0]["price"], points[1]["price"]], pen=guide_pen)
        bc = pg.PlotCurveItem(x=[x_b, x_c], y=[points[1]["price"], points[2]["price"]], pen=guide_pen)
        add_item(ab)
        add_item(bc)

        projection_span = max(abs(x_b - x_a), 1.0)
        x_right = x_c + projection_span
        for row in plan["rows"]:
            level = pg.PlotCurveItem(
                x=[x_c, x_right],
                y=[row["price"], row["price"]],
                pen=_line_pen(preview=preview, dashed=True, width=2.0),
            )
            add_item(level)
            if preview:
                continue
            label = pg.TextItem(
                text=f'{row["ratio"]:.3f}  {row["price"]:.3f}',
                color="#FFD54A",
                fill=pg.mkBrush(20, 20, 20, 220),
                anchor=(0, 0.5),
            )
            label.setPos(x_right, row["price"])
            label.setZValue(21)
            add_item(label, ignore_bounds=True)
        return items

    return items
