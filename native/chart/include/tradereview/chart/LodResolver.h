#pragma once

#include "tradereview/core/TimeRange.h"

#include <string>
#include <vector>

namespace tradereview::chart {

std::string choose_lod_period(
    const std::string& requested_period,
    core::TimeRange visible_range,
    int pixel_width,
    const std::vector<std::string>& available_periods,
    double max_bars_per_pixel = 2.0);

} // namespace tradereview::chart
