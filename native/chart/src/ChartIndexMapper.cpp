#include "tradereview/chart/ChartIndexMapper.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tradereview::chart {
namespace {

constexpr std::int64_t kDefaultStepNs = 60LL * 1'000'000'000LL;

} // namespace

void ChartIndexMapper::set_timestamps(std::vector<std::int64_t> timestamps)
{
    std::sort(timestamps.begin(), timestamps.end());
    timestamps_ = std::move(timestamps);
}

bool ChartIndexMapper::empty() const
{
    return timestamps_.empty();
}

std::size_t ChartIndexMapper::row_count() const
{
    return timestamps_.size();
}

int ChartIndexMapper::nearest_dense_x(std::int64_t timestamp_ns) const
{
    if (timestamps_.empty()) {
        throw std::runtime_error("cannot map timestamp in empty ChartIndexMapper");
    }

    const auto right = std::lower_bound(timestamps_.begin(), timestamps_.end(), timestamp_ns);
    if (right == timestamps_.begin()) {
        return 0;
    }
    if (right == timestamps_.end()) {
        return static_cast<int>(timestamps_.size() - 1);
    }

    const auto left = right - 1;
    const auto left_distance = timestamp_ns - *left;
    const auto right_distance = *right - timestamp_ns;
    if (left_distance <= right_distance) {
        return static_cast<int>(std::distance(timestamps_.begin(), left));
    }
    return static_cast<int>(std::distance(timestamps_.begin(), right));
}

std::int64_t ChartIndexMapper::timestamp_at_dense_x(int dense_x) const
{
    if (dense_x < 0 || static_cast<std::size_t>(dense_x) >= timestamps_.size()) {
        throw std::out_of_range("dense x is outside ChartIndexMapper range");
    }
    return timestamps_[static_cast<std::size_t>(dense_x)];
}

std::int64_t ChartIndexMapper::timestamp_from_x(double x) const
{
    if (timestamps_.empty()) {
        throw std::runtime_error("cannot map x in empty ChartIndexMapper");
    }

    const auto dense_x = static_cast<int>(std::llround(x));
    if (dense_x >= 0 && static_cast<std::size_t>(dense_x) < timestamps_.size()) {
        return timestamps_[static_cast<std::size_t>(dense_x)];
    }

    const auto step = step_ns();
    if (dense_x < 0) {
        return timestamps_.front() + (static_cast<std::int64_t>(dense_x) * step);
    }

    const auto last_dense_x = static_cast<int>(timestamps_.size() - 1);
    return timestamps_.back() + (static_cast<std::int64_t>(dense_x - last_dense_x) * step);
}

std::int64_t ChartIndexMapper::step_ns() const
{
    if (timestamps_.size() < 2) {
        return kDefaultStepNs;
    }

    std::vector<std::int64_t> steps;
    steps.reserve(timestamps_.size() - 1);
    for (std::size_t index = 1; index < timestamps_.size(); ++index) {
        steps.push_back(timestamps_[index] - timestamps_[index - 1]);
    }

    std::sort(steps.begin(), steps.end());
    const auto median_index = steps.size() / 2;
    if ((steps.size() % 2) == 1) {
        return steps[median_index];
    }

    const auto lower = steps[median_index - 1];
    const auto upper = steps[median_index];
    return lower + ((upper - lower) / 2);
}

} // namespace tradereview::chart
