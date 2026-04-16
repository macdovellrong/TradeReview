from __future__ import annotations


def _point_from_legacy(spec: dict, prefix: str) -> dict | None:
    dt_key = f"{prefix}_dt"
    price_key = f"{prefix}_price"
    if dt_key not in spec or price_key not in spec:
        return None
    return {"dt": spec.get(dt_key), "price": float(spec.get(price_key))}


def normalize_drawing_spec(spec: dict) -> dict:
    if "points" in spec:
        normalized = dict(spec)
        normalized["points"] = [
            {"dt": point.get("dt"), "price": float(point.get("price"))}
            for point in spec["points"]
        ]
        return normalized

    points = []
    for prefix in ("p1", "p2", "p3"):
        point = _point_from_legacy(spec, prefix)
        if point is not None:
            points.append(point)

    normalized = dict(spec)
    normalized["points"] = points
    return normalized
