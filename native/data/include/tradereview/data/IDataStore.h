#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"
#include "tradereview/data/DataSetInfo.h"

namespace tradereview::data {

struct CandleWindowRequest {
    uint64_t chart_id = 0;
    uint64_t generation = 0;
    std::string requested_period;
    core::TimeRange visible_range;
    int pixel_width = 0;
    double buffer_multiplier = 2.0;
    bool include_indicators = true;
    int warmup_bars = 0;
};

struct TickSlice {
    std::vector<int64_t> timestamp_ns;
    std::vector<double> price;
    std::vector<double> volume;
};

struct ReplayChunk {
    TickSlice ticks;
    bool reached_end = false;
};

class IDataStore {
public:
    virtual ~IDataStore() = default;
    virtual DataSetInfo open_readonly(const std::string& path) = 0;
    virtual CandleWindow query_candles(const CandleWindowRequest& request) = 0;
    virtual TickSlice query_ticks(core::TimeRange range, size_t max_rows) = 0;
    virtual ReplayChunk query_replay_ticks(int64_t from_ns, int64_t to_ns, size_t max_ticks) = 0;
};

} // namespace tradereview::data
