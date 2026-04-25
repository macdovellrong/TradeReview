#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tradereview/core/TimeRange.h"

namespace tradereview::data {

struct CandleWindow {
    uint64_t chart_id = 0;
    uint64_t generation = 0;
    std::string requested_period;
    std::string actual_period;
    core::TimeRange loaded_range;
    core::TimeRange visible_range;
    std::vector<int64_t> timestamp_ns;
    std::vector<double> open;
    std::vector<double> high;
    std::vector<double> low;
    std::vector<double> close;
    std::vector<double> volume;
    std::unordered_map<std::string, std::vector<double>> indicators;
    bool from_cache = false;

    [[nodiscard]] size_t row_count() const;
    [[nodiscard]] bool empty() const;
    [[nodiscard]] bool has_consistent_columns() const;
};

} // namespace tradereview::data
