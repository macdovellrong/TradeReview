#include "tradereview/chart/Windowing.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace tradereview::chart {
namespace {

constexpr std::int64_t kMinimumPositiveSpanNs = 60LL * 1'000'000'000LL;

std::int64_t positive_span(core::TimeRange range)
{
    if (range.end_ns <= range.start_ns) {
        return kMinimumPositiveSpanNs;
    }

    const auto span = static_cast<long double>(range.end_ns) - static_cast<long double>(range.start_ns);
    const auto max_value = static_cast<long double>(std::numeric_limits<std::int64_t>::max());
    if (span >= max_value) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return static_cast<std::int64_t>(span);
}

std::int64_t clamped_buffer(std::int64_t span, double multiplier)
{
    if (!std::isfinite(multiplier) || multiplier < 0.0) {
        multiplier = 0.0;
    }

    const auto buffer = static_cast<long double>(span) * static_cast<long double>(multiplier);
    const auto max_value = static_cast<long double>(std::numeric_limits<std::int64_t>::max());
    if (buffer >= max_value) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return static_cast<std::int64_t>(buffer);
}

std::int64_t saturating_subtract(std::int64_t value, std::int64_t amount)
{
    const auto min_value = std::numeric_limits<std::int64_t>::min();
    if (amount > 0 && value < min_value + amount) {
        return min_value;
    }
    return value - amount;
}

std::int64_t saturating_add(std::int64_t value, std::int64_t amount)
{
    const auto max_value = std::numeric_limits<std::int64_t>::max();
    if (amount > 0 && value > max_value - amount) {
        return max_value;
    }
    return value + amount;
}

} // namespace

core::TimeRange build_query_window(core::TimeRange visible_range, double buffer_multiplier)
{
    const auto span = positive_span(visible_range);
    const auto buffer = clamped_buffer(span, buffer_multiplier);
    return core::TimeRange{
        saturating_subtract(visible_range.start_ns, buffer),
        saturating_add(visible_range.end_ns, buffer),
    };
}

bool is_view_inside_loaded_window(core::TimeRange visible_range, core::TimeRange loaded_range)
{
    return loaded_range.start_ns <= visible_range.start_ns && visible_range.end_ns <= loaded_range.end_ns;
}

bool should_prefetch_window(core::TimeRange visible_range, core::TimeRange loaded_range, double edge_fraction)
{
    if (!is_view_inside_loaded_window(visible_range, loaded_range)) {
        return true;
    }

    const auto visible_span = positive_span(visible_range);
    const auto clamped_fraction = std::clamp(edge_fraction, 0.0, 1.0);
    const auto edge_span = clamped_buffer(visible_span, clamped_fraction);
    return visible_range.start_ns <= saturating_add(loaded_range.start_ns, edge_span)
        || visible_range.end_ns >= saturating_subtract(loaded_range.end_ns, edge_span);
}

} // namespace tradereview::chart
