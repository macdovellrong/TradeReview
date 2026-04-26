#include "tradereview/drawing/DrawingSpec.h"

#include <cmath>
#include <stdexcept>

namespace tradereview::drawing {
namespace {

void validate_points(const std::vector<DrawingPoint>& points, std::size_t count)
{
    if (points.size() < count) {
        throw std::invalid_argument("Drawing spec does not have enough points");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(points[index].price)) {
            throw std::invalid_argument("Drawing point price must be finite");
        }
    }
}

} // namespace

std::size_t required_point_count(DrawingType type)
{
    switch (type) {
    case DrawingType::HorizontalLine:
    case DrawingType::VerticalLine:
        return 1;
    case DrawingType::Line:
    case DrawingType::FibRetracement:
        return 2;
    case DrawingType::FibExtension:
        return 3;
    }
    throw std::invalid_argument("Unsupported drawing type");
}

const char* drawing_type_name(DrawingType type)
{
    switch (type) {
    case DrawingType::HorizontalLine:
        return "hline";
    case DrawingType::VerticalLine:
        return "vline";
    case DrawingType::Line:
        return "line";
    case DrawingType::FibRetracement:
        return "fib";
    case DrawingType::FibExtension:
        return "fib_ext";
    }
    return "unknown";
}

std::ostream& operator<<(std::ostream& out, DrawingType type)
{
    out << drawing_type_name(type);
    return out;
}

bool is_fib_drawing_type(DrawingType type)
{
    return type == DrawingType::FibRetracement || type == DrawingType::FibExtension;
}

DrawingSpec normalize_drawing_spec(DrawingType type, const std::vector<DrawingPoint>& points)
{
    const auto count = required_point_count(type);
    validate_points(points, count);

    DrawingSpec spec;
    spec.type = type;
    spec.points.assign(points.begin(), points.begin() + static_cast<std::ptrdiff_t>(count));
    return spec;
}

DrawingSpec create_drawing_spec(
    DrawingType type,
    const std::vector<DrawingPoint>& points,
    const FibSettings& fib_settings)
{
    auto spec = normalize_drawing_spec(type, points);
    if (type == DrawingType::FibRetracement) {
        spec.fib_snapshot = FibConfigSnapshot{fib_settings.retracement.effective_levels()};
    } else if (type == DrawingType::FibExtension) {
        spec.fib_snapshot = FibConfigSnapshot{fib_settings.extension.effective_levels()};
    }
    return spec;
}

} // namespace tradereview::drawing
