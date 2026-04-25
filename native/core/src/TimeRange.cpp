#include "tradereview/core/TimeRange.h"

#include <algorithm>

namespace tradereview::core {

std::int64_t TimeRange::span_ns() const
{
    return end_ns - start_ns;
}

bool TimeRange::contains(std::int64_t timestamp_ns) const
{
    return start_ns <= timestamp_ns && timestamp_ns <= end_ns;
}

TimeRange TimeRange::normalized(std::int64_t first_ns, std::int64_t second_ns)
{
    return TimeRange{
        std::min(first_ns, second_ns),
        std::max(first_ns, second_ns),
    };
}

} // namespace tradereview::core
