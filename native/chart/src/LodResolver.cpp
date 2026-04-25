#include "tradereview/chart/LodResolver.h"

#include "tradereview/core/Period.h"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace tradereview::chart {
namespace {

constexpr double kNanosecondsPerSecond = 1'000'000'000.0;

double bar_count(core::TimeRange range, std::int64_t period_seconds)
{
    const auto span_ns = std::max<std::int64_t>(0, range.span_ns());
    return static_cast<double>(span_ns) / (static_cast<double>(period_seconds) * kNanosecondsPerSecond);
}

bool fits_density(core::TimeRange range, std::int64_t period_seconds, int pixel_width, double max_bars_per_pixel)
{
    if (pixel_width <= 0 || max_bars_per_pixel <= 0.0) {
        return false;
    }
    return bar_count(range, period_seconds) <= static_cast<double>(pixel_width) * max_bars_per_pixel;
}

struct AvailablePeriod {
    std::string period;
    std::int64_t seconds = 0;
};

std::vector<AvailablePeriod> parse_available_periods(const std::vector<std::string>& available_periods)
{
    std::vector<AvailablePeriod> parsed;
    parsed.reserve(available_periods.size());
    for (const auto& period : available_periods) {
        const auto seconds = core::try_period_seconds(period);
        if (seconds.has_value()) {
            parsed.push_back(AvailablePeriod{period, *seconds});
        }
    }
    std::sort(parsed.begin(), parsed.end(), [](const AvailablePeriod& lhs, const AvailablePeriod& rhs) {
        return lhs.seconds < rhs.seconds;
    });
    return parsed;
}

} // namespace

std::string choose_lod_period(
    const std::string& requested_period,
    core::TimeRange visible_range,
    int pixel_width,
    const std::vector<std::string>& available_periods,
    double max_bars_per_pixel)
{
    const auto parsed_available_periods = parse_available_periods(available_periods);
    const auto maybe_requested_seconds = core::try_period_seconds(requested_period);
    if (!maybe_requested_seconds.has_value()) {
        if (!parsed_available_periods.empty()) {
            return parsed_available_periods.front().period;
        }
        return requested_period;
    }

    const auto requested_seconds = *maybe_requested_seconds;
    if (parsed_available_periods.empty()) {
        return requested_period;
    }

    const auto requested_is_available =
        std::any_of(parsed_available_periods.begin(), parsed_available_periods.end(), [&](const AvailablePeriod& available) {
            return available.period == requested_period;
        });
    if (requested_is_available && fits_density(visible_range, requested_seconds, pixel_width, max_bars_per_pixel)) {
        return requested_period;
    }

    std::optional<AvailablePeriod> coarsest_not_finer;
    for (const auto& available : parsed_available_periods) {
        if (available.seconds < requested_seconds) {
            continue;
        }

        coarsest_not_finer = available;
        if (fits_density(visible_range, available.seconds, pixel_width, max_bars_per_pixel)) {
            return available.period;
        }
    }

    if (coarsest_not_finer.has_value()) {
        return coarsest_not_finer->period;
    }
    return parsed_available_periods.back().period;
}

} // namespace tradereview::chart
