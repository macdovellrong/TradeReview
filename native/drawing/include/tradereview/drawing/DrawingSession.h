#pragma once

#include "tradereview/drawing/DrawingSpec.h"
#include "tradereview/drawing/FibSettings.h"

#include <optional>
#include <vector>

namespace tradereview::drawing {

class DrawingSession final {
public:
    static DrawingSession for_type(DrawingType type, const FibSettings& fib_settings);

    DrawingSession(DrawingType type, std::optional<FibConfigSnapshot> snapshot = std::nullopt);

    [[nodiscard]] DrawingType type() const;
    [[nodiscard]] const std::vector<DrawingPoint>& points() const;
    [[nodiscard]] std::optional<DrawingSpec> build_preview(DrawingPoint point) const;
    [[nodiscard]] bool complete() const;

    std::optional<DrawingSpec> add_point(DrawingPoint point);

private:
    [[nodiscard]] DrawingSpec spec_for(DrawingType type, const std::vector<DrawingPoint>& points) const;

    DrawingType type_ = DrawingType::Line;
    std::optional<FibConfigSnapshot> fib_snapshot_;
    std::vector<DrawingPoint> points_;
};

} // namespace tradereview::drawing
