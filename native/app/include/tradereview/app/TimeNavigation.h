#pragma once

#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace tradereview::app {

struct ChartTargetRow {
    std::size_t row = 0;
    double close = 0.0;
};

[[nodiscard]] std::int64_t normalize_jump_timestamp_ns(std::int64_t timestamp_ns);
[[nodiscard]] std::int64_t clamp_jump_timestamp_ns(std::int64_t timestamp_ns, core::TimeRange dataset_range);
[[nodiscard]] std::optional<ChartTargetRow> resolve_chart_target_row(
    const data::CandleWindow& window,
    std::int64_t target_ns);
[[nodiscard]] core::TimeRange centered_visible_range(
    std::int64_t center_ns,
    core::TimeRange dataset_range,
    std::int64_t width_ns);
[[nodiscard]] core::TimeRange adjusted_visible_range_for_period(
    core::TimeRange visible_range,
    core::TimeRange dataset_range,
    std::string_view period,
    int minimum_bars);

} // namespace tradereview::app
