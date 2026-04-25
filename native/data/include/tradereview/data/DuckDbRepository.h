#pragma once

#include <memory>
#include <string>

#include "tradereview/data/IDataStore.h"

namespace tradereview::data {

class DuckDbRepository final : public IDataStore {
public:
    DuckDbRepository();
    ~DuckDbRepository() override;

    DuckDbRepository(const DuckDbRepository&) = delete;
    DuckDbRepository& operator=(const DuckDbRepository&) = delete;
    DuckDbRepository(DuckDbRepository&&) noexcept;
    DuckDbRepository& operator=(DuckDbRepository&&) noexcept;

    DataSetInfo open_readonly(const std::string& path) override;
    CandleWindow query_candles(const CandleWindowRequest& request) override;
    TickSlice query_ticks(core::TimeRange range, size_t max_rows) override;
    ReplayChunk query_replay_ticks(int64_t from_ns, int64_t to_ns, size_t max_ticks) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tradereview::data
