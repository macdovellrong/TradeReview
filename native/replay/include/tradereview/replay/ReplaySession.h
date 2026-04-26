#pragma once

#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"
#include "tradereview/data/IDataStore.h"
#include "tradereview/replay/BarBuilder.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tradereview::replay {

struct ReplayConfig {
    core::TimeRange dataset_range;
    std::vector<std::string> periods;
    std::int64_t start_time_ns = 0;
    std::size_t max_ticks_per_frame = 20000;
    std::size_t max_bars_per_period = 1000;
};

struct ReplayAdvanceResult {
    std::int64_t current_time_ns = 0;
    std::size_t ticks_consumed = 0;
    bool reached_end = false;
};

class ReplaySession final {
public:
    explicit ReplaySession(std::shared_ptr<data::IDataStore> store);

    void configure(ReplayConfig config);
    void set_enabled(bool enabled);
    void set_playing(bool playing);
    [[nodiscard]] bool enabled() const;
    [[nodiscard]] bool playing() const;
    [[nodiscard]] bool toggle_playing();
    void set_speed(int speed);
    [[nodiscard]] int speed() const;
    void seek(std::int64_t timestamp_ns);

    [[nodiscard]] ReplayAdvanceResult advance_by(std::int64_t delta_ns);
    [[nodiscard]] ReplayAdvanceResult advance_to(std::int64_t target_time_ns);
    [[nodiscard]] std::optional<data::CandleWindow> window_for_period(
        std::string_view period,
        std::uint64_t chart_id,
        std::uint64_t generation,
        core::TimeRange visible_range) const;
    [[nodiscard]] std::int64_t current_time_ns() const;
    [[nodiscard]] bool reached_end() const;

private:
    void rebuild_builders();
    [[nodiscard]] std::int64_t clamp_to_dataset(std::int64_t timestamp_ns) const;
    [[nodiscard]] ReplayAdvanceResult current_result(std::size_t ticks_consumed = 0) const;

    std::shared_ptr<data::IDataStore> store_;
    ReplayConfig config_;
    std::map<std::string, BarBuilder> builders_;
    std::int64_t current_time_ns_ = 0;
    std::int64_t cursor_ns_ = 0;
    bool configured_ = false;
    bool enabled_ = false;
    bool playing_ = false;
    bool reached_end_ = false;
    int speed_ = 60;
};

} // namespace tradereview::replay
