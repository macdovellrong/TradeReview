from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolDefinition:
    tool_id: str
    point_count: int


TOOL_DEFINITIONS = {
    "hline": ToolDefinition(tool_id="hline", point_count=1),
    "vline": ToolDefinition(tool_id="vline", point_count=1),
    "line": ToolDefinition(tool_id="line", point_count=2),
    "fib": ToolDefinition(tool_id="fib", point_count=2),
    "fib_ext": ToolDefinition(tool_id="fib_ext", point_count=3),
    "rect": ToolDefinition(tool_id="rect", point_count=2),
}


@dataclass
class DrawingSession:
    tool: ToolDefinition
    config_snapshot: dict | None = None
    points: list[dict] = field(default_factory=list)

    def add_point(self, dt, price: float) -> dict | None:
        self.points.append({"dt": dt, "price": float(price)})
        if len(self.points) < self.tool.point_count:
            return None

        spec = {"type": self.tool.tool_id, "points": list(self.points)}
        if self.config_snapshot is not None:
            spec["config_snapshot"] = dict(self.config_snapshot)
        return spec

    def build_preview_spec(self, dt, price: float) -> dict | None:
        if not self.points:
            return None

        preview_points = [*self.points, {"dt": dt, "price": float(price)}]
        if self.tool.point_count == 2 and len(self.points) == 1:
            spec = {"type": self.tool.tool_id, "points": preview_points}
        elif self.tool.tool_id == "fib_ext" and len(self.points) == 1:
            spec = {"type": "line", "points": preview_points}
        elif self.tool.tool_id == "fib_ext" and len(self.points) == 2:
            spec = {"type": "fib_ext", "points": preview_points}
        else:
            return None

        if self.config_snapshot is not None:
            spec["config_snapshot"] = dict(self.config_snapshot)
        return spec
