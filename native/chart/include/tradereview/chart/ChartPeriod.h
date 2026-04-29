#pragma once

#include <string>
#include <string_view>

namespace tradereview::chart {

[[nodiscard]] std::string canonical_chart_period(std::string_view period);
[[nodiscard]] std::string toolbar_chart_period(std::string_view period);

} // namespace tradereview::chart
