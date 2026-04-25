#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tradereview::chart {

class ChartIndexMapper {
public:
    void set_timestamps(std::vector<std::int64_t> timestamps);

    bool empty() const;
    std::size_t row_count() const;

    int nearest_dense_x(std::int64_t timestamp_ns) const;
    std::int64_t timestamp_at_dense_x(int dense_x) const;
    std::int64_t timestamp_from_x(double x) const;

private:
    std::int64_t step_ns() const;

    std::vector<std::int64_t> timestamps_;
};

} // namespace tradereview::chart
