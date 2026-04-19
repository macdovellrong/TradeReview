from __future__ import annotations

import pandas as pd


SUPPORTED_DRAWING_TYPES = ("hline", "vline", "line", "fib", "fib_ext", "rect")


def _normalize_point(point: dict) -> dict:
    return {
        "dt": pd.Timestamp(point.get("dt")),
        "price": float(point.get("price")),
    }


def _point_from_legacy(spec: dict, prefix: str) -> dict | None:
    dt_key = f"{prefix}_dt"
    price_key = f"{prefix}_price"
    if dt_key not in spec or price_key not in spec:
        return None
    return _normalize_point({"dt": spec.get(dt_key), "price": spec.get(price_key)})


def normalize_drawing_spec(spec: dict) -> dict:
    if "points" in spec:
        normalized = dict(spec)
        normalized["points"] = [_normalize_point(point) for point in spec["points"]]
        return normalized

    points = []
    for prefix in ("p1", "p2", "p3"):
        point = _point_from_legacy(spec, prefix)
        if point is not None:
            points.append(point)

    normalized = dict(spec)
    normalized["points"] = points
    return normalized


def serialize_drawing_spec(spec: dict) -> dict:
    normalized = normalize_drawing_spec(spec)
    payload = dict(normalized)
    payload["points"] = [
        {
            "dt": pd.Timestamp(point["dt"]).isoformat(),
            "price": float(point["price"]),
        }
        for point in normalized.get("points", [])
    ]
    if "config_snapshot" in normalized and normalized["config_snapshot"] is not None:
        payload["config_snapshot"] = dict(normalized["config_snapshot"])
    return payload


def deserialize_drawing_spec(payload: dict) -> dict:
    spec_type = payload.get("type")
    if spec_type not in SUPPORTED_DRAWING_TYPES:
        raise ValueError(f"Unsupported drawing type: {spec_type!r}")
    return normalize_drawing_spec(payload)
