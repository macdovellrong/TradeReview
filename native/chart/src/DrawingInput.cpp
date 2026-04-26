#include "tradereview/chart/DrawingInput.h"

#include <algorithm>
#include <cmath>

namespace tradereview::chart {
namespace {

[[nodiscard]] bool finite(double value)
{
    return std::isfinite(value);
}

} // namespace

std::optional<drawing::DrawingPoint> drawing_point_from_widget_position(
    const ChartIndexMapper& mapper,
    DenseRange visible_dense_range,
    PaneRect price_pane,
    int widget_width,
    int widget_height,
    WidgetPosition position,
    double min_price,
    double max_price)
{
    if (mapper.empty() || widget_width <= 0 || widget_height <= 0) {
        return std::nullopt;
    }
    if (!finite(position.x) || !finite(position.y) || !finite(min_price) || !finite(max_price)) {
        return std::nullopt;
    }
    if (position.x < 0.0 || position.x > static_cast<double>(widget_width)) {
        return std::nullopt;
    }

    const auto normalized_device_y = 1.0 - (2.0 * (position.y / static_cast<double>(widget_height)));
    if (normalized_device_y < static_cast<double>(price_pane.bottom)
        || normalized_device_y > static_cast<double>(price_pane.top)) {
        return std::nullopt;
    }

    auto normalized_range = visible_dense_range;
    if (!finite(normalized_range.start_x) || !finite(normalized_range.end_x)) {
        normalized_range = {0.0, 1.0};
    }
    if (normalized_range.end_x < normalized_range.start_x) {
        std::swap(normalized_range.start_x, normalized_range.end_x);
    }
    if (normalized_range.end_x <= normalized_range.start_x) {
        normalized_range.end_x = normalized_range.start_x + 1.0;
    }

    if (max_price <= min_price) {
        max_price = min_price + 1.0;
    }

    const auto x_fraction = std::clamp(position.x / static_cast<double>(widget_width), 0.0, 1.0);
    const auto dense_x = normalized_range.start_x + (normalized_range.span() * x_fraction);
    const auto pane_fraction =
        (normalized_device_y - static_cast<double>(price_pane.bottom))
        / static_cast<double>(std::max(price_pane.height(), 0.000001F));

    return drawing::DrawingPoint{
        mapper.timestamp_from_x(dense_x),
        min_price + ((max_price - min_price) * pane_fraction),
    };
}

} // namespace tradereview::chart
