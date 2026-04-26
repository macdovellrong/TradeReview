#pragma once

#include "tradereview/drawing/FibSettings.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace tradereview::drawing {

enum class DrawingType {
    HorizontalLine,
    VerticalLine,
    Line,
    FibRetracement,
    FibExtension,
};

struct DrawingPoint {
    std::int64_t timestamp_ns = 0;
    double price = 0.0;
};

struct FibConfigSnapshot {
    std::vector<double> levels;
};

struct DrawingSpec {
    DrawingType type = DrawingType::Line;
    std::vector<DrawingPoint> points;
    std::optional<FibConfigSnapshot> fib_snapshot;
};

[[nodiscard]] std::size_t required_point_count(DrawingType type);
[[nodiscard]] bool is_fib_drawing_type(DrawingType type);
[[nodiscard]] DrawingSpec normalize_drawing_spec(
    DrawingType type,
    const std::vector<DrawingPoint>& points);
[[nodiscard]] DrawingSpec create_drawing_spec(
    DrawingType type,
    const std::vector<DrawingPoint>& points,
    const FibSettings& fib_settings);

} // namespace tradereview::drawing
