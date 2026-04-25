#pragma once

#include "tradereview/core/TimeRange.h"

namespace tradereview::chart {

core::TimeRange build_query_window(core::TimeRange visible_range, double buffer_multiplier);
bool is_view_inside_loaded_window(core::TimeRange visible_range, core::TimeRange loaded_range);
bool should_prefetch_window(core::TimeRange visible_range, core::TimeRange loaded_range, double edge_fraction);

} // namespace tradereview::chart
