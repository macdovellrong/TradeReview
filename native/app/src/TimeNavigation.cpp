#include "tradereview/app/TimeNavigation.h"

#include "tradereview/core/Period.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tradereview::app {
namespace {

constexpr std::int64_t kNanosecondsPerSecond = 1000LL * 1000LL * 1000LL;
constexpr std::int64_t kNanosecondsPerMinute = 60LL * kNanosecondsPerSecond;

[[nodiscard]] std::int64_t positive_width(core::TimeRange range)
{
    return std::max<std::int64_t>(0, range.end_ns - range.start_ns);
}

[[nodiscard]] std::int64_t minimum_period_width_ns(std::int64_t period_seconds, int minimum_bars)
{
    if (period_seconds <= 0 || minimum_bars <= 0) {
        return 0;
    }

    const auto width = static_cast<long double>(period_seconds)
        * static_cast<long double>(kNanosecondsPerSecond)
        * static_cast<long double>(minimum_bars);
    const auto max_value = static_cast<long double>(std::numeric_limits<std::int64_t>::max());
    if (width >= max_value) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return static_cast<std::int64_t>(width);
}

} // namespace

std::int64_t normalize_jump_timestamp_ns(std::int64_t timestamp_ns)
{
    const auto remainder = timestamp_ns % kNanosecondsPerMinute;
    auto normalized = timestamp_ns - remainder;
    if (remainder < 0) {
        normalized -= kNanosecondsPerMinute;
    }
    return normalized;
}

std::int64_t clamp_jump_timestamp_ns(std::int64_t timestamp_ns, core::TimeRange dataset_range)
{
    const auto normalized_range = core::TimeRange::normalized(dataset_range.start_ns, dataset_range.end_ns);
    return std::clamp(timestamp_ns, normalized_range.start_ns, normalized_range.end_ns);
}

std::optional<ChartTargetRow> resolve_chart_target_row(
    const data::CandleWindow& window,
    std::int64_t target_ns)
{
    if (window.timestamp_ns.empty() || window.close.size() != window.timestamp_ns.size()) {
        return std::nullopt;
    }

    auto right = std::lower_bound(window.timestamp_ns.begin(), window.timestamp_ns.end(), target_ns);
    if (right == window.timestamp_ns.end()) {
        --right;
    }

    const auto row = static_cast<std::size_t>(std::distance(window.timestamp_ns.begin(), right));
    const auto close = window.close[row];
    if (!std::isfinite(close)) {
        return std::nullopt;
    }
    return ChartTargetRow{row, close};
}

core::TimeRange centered_visible_range(
    std::int64_t center_ns,
    core::TimeRange dataset_range,
    std::int64_t width_ns)
{
    const auto normalized_range = core::TimeRange::normalized(dataset_range.start_ns, dataset_range.end_ns);
    const auto dataset_width = positive_width(normalized_range);
    if (dataset_width <= 0) {
        return normalized_range;
    }

    const auto target_width = std::clamp<std::int64_t>(width_ns, 1, dataset_width);
    const auto clamped_center = clamp_jump_timestamp_ns(center_ns, normalized_range);
    auto start = clamped_center - (target_width / 2);
    auto end = start + target_width;

    if (start < normalized_range.start_ns) {
        start = normalized_range.start_ns;
        end = start + target_width;
    }
    if (end > normalized_range.end_ns) {
        end = normalized_range.end_ns;
        start = end - target_width;
    }

    return {start, end};
}

core::TimeRange adjusted_visible_range_for_period(
    core::TimeRange visible_range,
    core::TimeRange dataset_range,
    std::string_view period,
    int minimum_bars)
{
    const auto period_seconds = core::try_period_seconds(period);
    if (!period_seconds.has_value()) {
        return visible_range;
    }

    const auto minimum_width = minimum_period_width_ns(*period_seconds, minimum_bars);
    if (minimum_width <= 0 || positive_width(visible_range) >= minimum_width) {
        return visible_range;
    }

    return centered_visible_range(
        visible_range.start_ns + (positive_width(visible_range) / 2),
        dataset_range,
        minimum_width);
}

} // namespace tradereview::app
