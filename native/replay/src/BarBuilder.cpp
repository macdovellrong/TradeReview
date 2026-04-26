#include "tradereview/replay/BarBuilder.h"

#include "tradereview/core/Period.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tradereview::replay {
namespace {

constexpr std::int64_t kNanosecondsPerSecond = 1000LL * 1000LL * 1000LL;

[[nodiscard]] std::int64_t to_period_ns(const std::string& period)
{
    const auto seconds = core::period_seconds(period);
    if (seconds <= 0 || seconds > std::numeric_limits<std::int64_t>::max() / kNanosecondsPerSecond) {
        throw std::invalid_argument("invalid replay period");
    }
    return seconds * kNanosecondsPerSecond;
}

[[nodiscard]] double quiet_nan()
{
    return std::numeric_limits<double>::quiet_NaN();
}

} // namespace

BarBuilder::BarBuilder(std::string period, std::size_t max_bars)
    : period_(std::move(period))
    , period_ns_(to_period_ns(period_))
    , max_bars_(std::max<std::size_t>(1, max_bars))
{
}

void BarBuilder::reset()
{
    completed_.clear();
    current_.reset();
}

void BarBuilder::add_tick(std::int64_t timestamp_ns, double price, double volume)
{
    if (!std::isfinite(price) || !std::isfinite(volume)) {
        return;
    }

    const auto bucket = bucket_start(timestamp_ns);
    if (!current_.has_value()) {
        current_ = Bar{bucket, price, price, price, price, volume};
        trim_completed();
        return;
    }

    if (bucket < current_->timestamp_ns) {
        return;
    }

    if (bucket == current_->timestamp_ns) {
        current_->high = std::max(current_->high, price);
        current_->low = std::min(current_->low, price);
        current_->close = price;
        current_->volume += volume;
        return;
    }

    append_completed(*current_);
    for (auto gap = current_->timestamp_ns + period_ns_; gap < bucket; gap += period_ns_) {
        append_completed(gap_bar(gap));
    }
    current_ = Bar{bucket, price, price, price, price, volume};
    trim_completed();
}

bool BarBuilder::empty() const
{
    return completed_.empty() && !current_.has_value();
}

const std::string& BarBuilder::period() const
{
    return period_;
}

std::int64_t BarBuilder::period_ns() const
{
    return period_ns_;
}

data::CandleWindow BarBuilder::to_window(
    std::uint64_t chart_id,
    std::uint64_t generation,
    core::TimeRange visible_range) const
{
    data::CandleWindow window;
    window.chart_id = chart_id;
    window.generation = generation;
    window.requested_period = period_;
    window.actual_period = period_;
    window.visible_range = visible_range;

    const auto row_count = completed_.size() + (current_.has_value() ? 1U : 0U);
    window.timestamp_ns.reserve(row_count);
    window.open.reserve(row_count);
    window.high.reserve(row_count);
    window.low.reserve(row_count);
    window.close.reserve(row_count);
    window.volume.reserve(row_count);

    auto append_bar = [&window](const Bar& bar) {
        window.timestamp_ns.push_back(bar.timestamp_ns);
        window.open.push_back(bar.open);
        window.high.push_back(bar.high);
        window.low.push_back(bar.low);
        window.close.push_back(bar.close);
        window.volume.push_back(bar.volume);
    };

    for (const auto& bar : completed_) {
        append_bar(bar);
    }
    if (current_.has_value()) {
        append_bar(*current_);
    }

    if (!window.timestamp_ns.empty()) {
        window.loaded_range.start_ns = window.timestamp_ns.front();
        window.loaded_range.end_ns = window.timestamp_ns.back();
        if (!window.has_visible_range()) {
            window.visible_range = window.loaded_range;
        }
    }
    return window;
}

std::int64_t BarBuilder::bucket_start(std::int64_t timestamp_ns) const
{
    auto quotient = timestamp_ns / period_ns_;
    const auto remainder = timestamp_ns % period_ns_;
    if (remainder < 0) {
        --quotient;
    }
    return quotient * period_ns_;
}

void BarBuilder::append_completed(Bar bar)
{
    completed_.push_back(std::move(bar));
    trim_completed();
}

void BarBuilder::trim_completed()
{
    const auto current_rows = current_.has_value() ? 1U : 0U;
    const auto max_completed = max_bars_ > current_rows ? max_bars_ - current_rows : 0U;
    if (completed_.size() <= max_completed) {
        return;
    }
    completed_.erase(completed_.begin(), completed_.begin() + static_cast<std::ptrdiff_t>(completed_.size() - max_completed));
}

BarBuilder::Bar BarBuilder::gap_bar(std::int64_t timestamp_ns) const
{
    const auto nan = quiet_nan();
    return Bar{timestamp_ns, nan, nan, nan, nan, 0.0};
}

} // namespace tradereview::replay
