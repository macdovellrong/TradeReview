#include "tradereview/chart/ChartOverlayGeometry.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

namespace tradereview::chart {
namespace {

[[nodiscard]] DenseRange normalized_visible_range(DenseRange range)
{
    if (!std::isfinite(range.start_x) || !std::isfinite(range.end_x)) {
        return {0.0, 1.0};
    }
    if (range.end_x < range.start_x) {
        std::swap(range.start_x, range.end_x);
    }
    if (range.end_x <= range.start_x) {
        range.end_x = range.start_x + 1.0;
    }
    return range;
}

} // namespace

double widget_x_for_dense_x(DenseRange visible_dense_range, int widget_width, double dense_x)
{
    if (widget_width <= 0 || !std::isfinite(dense_x)) {
        return 0.0;
    }

    const auto range = normalized_visible_range(visible_dense_range);
    const auto fraction = (dense_x - range.start_x) / std::max(range.span(), 1.0);
    return fraction * static_cast<double>(widget_width);
}

double widget_y_for_normalized_device_y(float normalized_device_y, int widget_height)
{
    if (widget_height <= 0 || !std::isfinite(normalized_device_y)) {
        return 0.0;
    }
    return (1.0 - static_cast<double>(normalized_device_y)) * 0.5 * static_cast<double>(widget_height);
}

std::optional<double> widget_y_for_price(
    PaneRect pane,
    int widget_height,
    double min_price,
    double max_price,
    double price)
{
    if (widget_height <= 0 || !std::isfinite(min_price) || !std::isfinite(max_price) || !std::isfinite(price)) {
        return std::nullopt;
    }
    if (max_price <= min_price) {
        max_price = min_price + 1.0;
    }

    const auto price_fraction = (price - min_price) / (max_price - min_price);
    const auto normalized_device_y = pane.bottom + (static_cast<float>(price_fraction) * pane.height());
    return widget_y_for_normalized_device_y(normalized_device_y, widget_height);
}

std::string format_fib_ratio_label(double ratio)
{
    if (!std::isfinite(ratio)) {
        return {};
    }

    const auto percentage = ratio * 100.0;
    const auto rounded_to_tenth = std::round(percentage * 10.0) / 10.0;
    const auto rounded_to_integer = std::round(rounded_to_tenth);

    std::ostringstream out;
    if (std::abs(rounded_to_tenth - rounded_to_integer) < 0.000001) {
        out << static_cast<long long>(rounded_to_integer);
    } else {
        out << std::fixed << std::setprecision(1) << rounded_to_tenth;
    }
    out << '%';
    return out.str();
}

} // namespace tradereview::chart
