#include "tradereview/chart/Windowing.h"

#include <algorithm>
#include <cstdint>

namespace tradereview::chart {
namespace {

constexpr std::int64_t kMinimumPositiveSpanNs = 60LL * 1'000'000'000LL;

std::int64_t positive_span(core::TimeRange range)
{
    const auto span = range.span_ns();
    if (span > 0) {
        return span;
    }
    return kMinimumPositiveSpanNs;
}

} // namespace

core::TimeRange build_query_window(core::TimeRange visible_range, double buffer_multiplier)
{
    const auto span = positive_span(visible_range);
    const auto clamped_multiplier = std::max(0.0, buffer_multiplier);
    const auto buffer = static_cast<std::int64_t>(static_cast<double>(span) * clamped_multiplier);
    return core::TimeRange{visible_range.start_ns - buffer, visible_range.end_ns + buffer};
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

    const auto loaded_span = positive_span(loaded_range);
    const auto clamped_fraction = std::clamp(edge_fraction, 0.0, 1.0);
    const auto edge_span = static_cast<std::int64_t>(static_cast<double>(loaded_span) * clamped_fraction);
    return visible_range.start_ns <= loaded_range.start_ns + edge_span
        || visible_range.end_ns >= loaded_range.end_ns - edge_span;
}

} // namespace tradereview::chart
