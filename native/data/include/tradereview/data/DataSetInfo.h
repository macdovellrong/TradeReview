#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tradereview/core/TimeRange.h"

namespace tradereview::data {

struct DataSetInfo {
    std::string dataset_path;
    int64_t tick_count = 0;
    core::TimeRange tick_range;
    std::vector<std::string> available_periods;
    std::vector<std::string> available_indicators;
    std::string schema_version;
    std::string indicator_version;
    bool metadata_only = false;
};

} // namespace tradereview::data
