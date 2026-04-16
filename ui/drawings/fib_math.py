from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FibLevel:
    ratio: float
    price: float


def build_retracement_levels(start_price: float, end_price: float, levels: list[float]) -> list[FibLevel]:
    return [
        FibLevel(ratio=float(level), price=float(end_price + (start_price - end_price) * level))
        for level in levels
    ]


def build_extension_levels(a_price: float, b_price: float, c_price: float, levels: list[float]) -> list[FibLevel]:
    delta = b_price - a_price
    return [
        FibLevel(ratio=float(level), price=float(c_price + delta * level))
        for level in levels
    ]
