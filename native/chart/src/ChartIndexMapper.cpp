#include "tradereview/chart/ChartIndexMapper.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tradereview::chart {
namespace {

constexpr std::int64_t kDefaultStepNs = 60LL * 1'000'000'000LL;

int round_half_left_to_int(double x)
{
    if (!std::isfinite(x)) {
        throw std::invalid_argument("x must be finite");
    }

    const auto min_dense_x = static_cast<double>(std::numeric_limits<int>::min());
    const auto max_dense_x = static_cast<double>(std::numeric_limits<int>::max());
    if (x <= min_dense_x) {
        return std::numeric_limits<int>::min();
    }
    if (x >= max_dense_x) {
        return std::numeric_limits<int>::max();
    }

    const auto lower = std::floor(x);
    const auto rounded = ((x - lower) == 0.5) ? lower : std::floor(x + 0.5);
    return static_cast<int>(rounded);
}

} // namespace

void ChartIndexMapper::set_timestamps(std::vector<std::int64_t> timestamps)
{
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

double ChartIndexMapper::dense_x_from_timestamp(std::int64_t timestamp_ns) const
{
    if (timestamps_.empty()) {
        throw std::runtime_error("cannot map timestamp in empty ChartIndexMapper");
    }

    const auto right = std::lower_bound(timestamps_.begin(), timestamps_.end(), timestamp_ns);
    if (right == timestamps_.begin()) {
        const auto step = static_cast<double>(step_ns());
        return static_cast<double>(timestamp_ns - timestamps_.front()) / step;
    }
    if (right == timestamps_.end()) {
        const auto step = static_cast<double>(step_ns());
        return static_cast<double>(timestamps_.size() - 1) + (static_cast<double>(timestamp_ns - timestamps_.back()) / step);
    }
    if (*right == timestamp_ns) {
        return static_cast<double>(std::distance(timestamps_.begin(), right));
    }

    const auto left = right - 1;
    const auto interval = *right - *left;
    if (interval <= 0) {
        return static_cast<double>(nearest_dense_x(timestamp_ns));
    }

    const auto left_x = static_cast<double>(std::distance(timestamps_.begin(), left));
    const auto fraction = static_cast<double>(timestamp_ns - *left) / static_cast<double>(interval);
    return left_x + fraction;
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

    const auto dense_x = round_half_left_to_int(x);
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
        const auto step = timestamps_[index] - timestamps_[index - 1];
        if (step > 0) {
            steps.push_back(step);
        }
    }

    if (steps.empty()) {
        return kDefaultStepNs;
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
