#include "tradereview/chart/ChartInteractionController.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>

namespace tradereview::chart {
namespace {

constexpr double kMinimumVisibleBars = 1.0;
constexpr double kWheelZoomBase = 0.8;

std::int64_t saturating_add(std::int64_t value, std::int64_t amount)
{
    const auto max_value = std::numeric_limits<std::int64_t>::max();
    if (amount > 0 && value > max_value - amount) {
        return max_value;
    }
    return value + amount;
}

std::int64_t saturating_subtract(std::int64_t value, std::int64_t amount)
{
    const auto min_value = std::numeric_limits<std::int64_t>::min();
    if (amount > 0 && value < min_value + amount) {
        return min_value;
    }
    return value - amount;
}

std::int64_t clamped_fraction_span(core::TimeRange range, double fraction)
{
    const auto span = static_cast<long double>(range.end_ns) - static_cast<long double>(range.start_ns);
    if (span <= 0.0L) {
        return 0;
    }
    const auto scaled = span * static_cast<long double>(std::clamp(fraction, 0.0, 1.0));
    const auto max_value = static_cast<long double>(std::numeric_limits<std::int64_t>::max());
    if (scaled >= max_value) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return static_cast<std::int64_t>(scaled);
}

} // namespace

double DenseRange::span() const
{
    return end_x - start_x;
}

ChartInteractionController::ChartInteractionController(int right_padding_bars)
    : right_padding_bars_(std::max(0, right_padding_bars))
{
}

void ChartInteractionController::reset_for_row_count(std::size_t row_count)
{
    if (row_count == 0) {
        visible_dense_range_ = {0.0, 0.0};
        return;
    }

    visible_dense_range_ = {
        0.0,
        static_cast<double>(row_count - 1) + static_cast<double>(right_padding_bars_),
    };
}

void ChartInteractionController::reset_for_visible_time_range(const ChartIndexMapper& mapper, core::TimeRange visible_range)
{
    if (mapper.empty() || visible_range.end_ns <= visible_range.start_ns) {
        reset_for_row_count(mapper.row_count());
        return;
    }

    auto start_x = static_cast<double>(mapper.nearest_dense_x(visible_range.start_ns));
    auto end_x = static_cast<double>(mapper.nearest_dense_x(visible_range.end_ns));
    const auto last_dense_x = static_cast<double>(mapper.row_count() - 1);
    try {
        if (visible_range.end_ns >= mapper.timestamp_at_dense_x(static_cast<int>(last_dense_x))) {
            end_x = last_dense_x + static_cast<double>(right_padding_bars_);
        }
    } catch (const std::exception&) {
        end_x = std::max(end_x, last_dense_x);
    }
    set_visible_dense_range({start_x, end_x});
}

void ChartInteractionController::set_visible_dense_range(DenseRange range)
{
    visible_dense_range_ = normalized_range(range);
}

void ChartInteractionController::pan_by_pixels(double pixel_delta_x, int viewport_width)
{
    if (!std::isfinite(pixel_delta_x) || viewport_width <= 0 || visible_dense_range_.span() <= 0.0) {
        return;
    }

    const auto dense_delta = (pixel_delta_x / static_cast<double>(viewport_width)) * visible_dense_range_.span();
    set_visible_dense_range({
        visible_dense_range_.start_x - dense_delta,
        visible_dense_range_.end_x - dense_delta,
    });
}

void ChartInteractionController::zoom_at_pixel(double pixel_x, int viewport_width, double scale_factor)
{
    if (!std::isfinite(pixel_x) || !std::isfinite(scale_factor) || scale_factor <= 0.0 || viewport_width <= 0) {
        return;
    }

    const auto span = visible_dense_range_.span();
    if (span <= 0.0) {
        return;
    }

    const auto anchor_fraction = std::clamp(pixel_x / static_cast<double>(viewport_width), 0.0, 1.0);
    const auto anchor_x = visible_dense_range_.start_x + (span * anchor_fraction);
    set_visible_dense_range({
        anchor_x - ((anchor_x - visible_dense_range_.start_x) * scale_factor),
        anchor_x + ((visible_dense_range_.end_x - anchor_x) * scale_factor),
    });
}

DenseRange ChartInteractionController::visible_dense_range() const
{
    return visible_dense_range_;
}

int ChartInteractionController::right_padding_bars() const
{
    return right_padding_bars_;
}

double ChartInteractionController::zoom_scale_from_wheel_delta(int angle_delta_y, int pixel_delta_y) const
{
    auto steps = 0.0;
    if (angle_delta_y != 0) {
        steps = static_cast<double>(angle_delta_y) / 120.0;
    } else if (pixel_delta_y != 0) {
        steps = static_cast<double>(pixel_delta_y) / 120.0;
    }

    if (steps == 0.0) {
        return 1.0;
    }
    steps = std::clamp(steps, -10.0, 10.0);
    return std::pow(kWheelZoomBase, steps);
}

core::TimeRange ChartInteractionController::visible_time_range(const ChartIndexMapper& mapper) const
{
    if (mapper.empty()) {
        return {};
    }

    return core::TimeRange::normalized(
        mapper.timestamp_from_x(visible_dense_range_.start_x),
        mapper.timestamp_from_x(visible_dense_range_.end_x));
}

ReloadDecision ChartInteractionController::reload_decision(const ChartIndexMapper& mapper, core::TimeRange loaded_range) const
{
    ReloadDecision decision;
    if (mapper.empty() || loaded_range.end_ns <= loaded_range.start_ns) {
        return decision;
    }

    decision.visible_range = visible_time_range(mapper);
    const auto padded_loaded = padded_loaded_range(mapper, loaded_range);
    if (decision.visible_range.start_ns < padded_loaded.start_ns || decision.visible_range.end_ns > padded_loaded.end_ns) {
        decision.requested = true;
        return decision;
    }

    const auto edge_margin = clamped_fraction_span(decision.visible_range, 0.5);
    const auto left_edge = saturating_add(loaded_range.start_ns, edge_margin);
    if (decision.visible_range.start_ns <= left_edge) {
        decision.requested = true;
        return decision;
    }

    const auto right_edge = saturating_subtract(loaded_range.end_ns, edge_margin);
    if (decision.visible_range.end_ns <= loaded_range.end_ns && decision.visible_range.end_ns >= right_edge) {
        decision.requested = true;
    }
    return decision;
}

DenseRange ChartInteractionController::normalized_range(DenseRange range) const
{
    if (!std::isfinite(range.start_x) || !std::isfinite(range.end_x)) {
        return visible_dense_range_;
    }
    if (range.end_x < range.start_x) {
        std::swap(range.start_x, range.end_x);
    }
    if (range.end_x - range.start_x < kMinimumVisibleBars) {
        const auto midpoint = (range.start_x + range.end_x) * 0.5;
        range.start_x = midpoint - (kMinimumVisibleBars * 0.5);
        range.end_x = midpoint + (kMinimumVisibleBars * 0.5);
    }
    return range;
}

core::TimeRange ChartInteractionController::padded_loaded_range(const ChartIndexMapper& mapper, core::TimeRange loaded_range) const
{
    if (mapper.empty()) {
        return loaded_range;
    }

    try {
        const auto last_dense_x = static_cast<double>(mapper.row_count() - 1);
        const auto padded_end = mapper.timestamp_from_x(last_dense_x + static_cast<double>(right_padding_bars_));
        return {
            loaded_range.start_ns,
            std::max(loaded_range.end_ns, padded_end),
        };
    } catch (const std::exception&) {
        return loaded_range;
    }
}

} // namespace tradereview::chart
