#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tradereview/core/TimeRange.h"

namespace tradereview::data {

struct DataSetInfo {
    std::string path;
    int64_t tick_count = 0;
    core::TimeRange tick_range;
    std::vector<std::string> periods;
    std::vector<std::string> indicators;
    std::string schema_version;
    std::string indicator_version;
};

} // namespace tradereview::data
