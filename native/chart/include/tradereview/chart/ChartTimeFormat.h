#pragma once

#include <cstdint>
#include <string_view>

#include <QString>

namespace tradereview::chart {

[[nodiscard]] QString format_axis_timestamp_label(std::int64_t timestamp_ns, std::string_view period);

} // namespace tradereview::chart
