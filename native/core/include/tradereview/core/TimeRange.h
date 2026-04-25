#pragma once

#include <cstdint>

namespace tradereview::core {

struct TimeRange {
    std::int64_t start_ns = 0;
    std::int64_t end_ns = 0;

    std::int64_t span_ns() const;
    bool contains(std::int64_t timestamp_ns) const;

    static TimeRange normalized(std::int64_t first_ns, std::int64_t second_ns);
};

} // namespace tradereview::core
