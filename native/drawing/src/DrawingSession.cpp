#include "tradereview/drawing/DrawingSession.h"

#include <utility>

namespace tradereview::drawing {
namespace {

std::optional<FibConfigSnapshot> snapshot_for(DrawingType type, const FibSettings& fib_settings)
{
    if (type == DrawingType::FibRetracement) {
        return FibConfigSnapshot{fib_settings.retracement.effective_levels()};
    }
    if (type == DrawingType::FibExtension) {
        return FibConfigSnapshot{fib_settings.extension.effective_levels()};
    }
    return std::nullopt;
}

} // namespace

DrawingSession DrawingSession::for_type(DrawingType type, const FibSettings& fib_settings)
{
    return DrawingSession(type, snapshot_for(type, fib_settings));
}

DrawingSession::DrawingSession(DrawingType type, std::optional<FibConfigSnapshot> snapshot)
    : type_(type)
    , fib_snapshot_(std::move(snapshot))
{
}

DrawingType DrawingSession::type() const
{
    return type_;
}

const std::vector<DrawingPoint>& DrawingSession::points() const
{
    return points_;
}

std::optional<DrawingSpec> DrawingSession::build_preview(DrawingPoint point) const
{
    if (points_.empty()) {
        return std::nullopt;
    }

    auto preview_points = points_;
    preview_points.push_back(point);
    if (type_ == DrawingType::FibExtension && points_.size() == 1) {
        return spec_for(DrawingType::Line, preview_points);
    }
    if (preview_points.size() >= required_point_count(type_)) {
        return spec_for(type_, preview_points);
    }
    return std::nullopt;
}

bool DrawingSession::complete() const
{
    return points_.size() >= required_point_count(type_);
}

std::optional<DrawingSpec> DrawingSession::add_point(DrawingPoint point)
{
    points_.push_back(point);
    if (!complete()) {
        return std::nullopt;
    }
    return spec_for(type_, points_);
}

DrawingSpec DrawingSession::spec_for(DrawingType type, const std::vector<DrawingPoint>& points) const
{
    auto spec = normalize_drawing_spec(type, points);
    if (fib_snapshot_.has_value()) {
        spec.fib_snapshot = *fib_snapshot_;
    }
    return spec;
}

} // namespace tradereview::drawing
