#pragma once

#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tradereview::replay {

class BarBuilder final {
public:
    explicit BarBuilder(std::string period, std::size_t max_bars = 1000);

    void reset();
    void add_tick(std::int64_t timestamp_ns, double price, double volume);

    [[nodiscard]] bool empty() const;
    [[nodiscard]] const std::string& period() const;
    [[nodiscard]] std::int64_t period_ns() const;
    [[nodiscard]] data::CandleWindow to_window(
        std::uint64_t chart_id,
        std::uint64_t generation,
        core::TimeRange visible_range) const;

private:
    struct Bar {
        std::int64_t timestamp_ns = 0;
        double open = 0.0;
        double high = 0.0;
        double low = 0.0;
        double close = 0.0;
        double volume = 0.0;
    };

    [[nodiscard]] std::int64_t bucket_start(std::int64_t timestamp_ns) const;
    void append_completed(Bar bar);
    void trim_completed();
    [[nodiscard]] Bar gap_bar(std::int64_t timestamp_ns) const;

    std::string period_;
    std::int64_t period_ns_ = 0;
    std::size_t max_bars_ = 1;
    std::vector<Bar> completed_;
    std::optional<Bar> current_;
};

} // namespace tradereview::replay
