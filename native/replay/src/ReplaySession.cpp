#include "tradereview/replay/ReplaySession.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tradereview::replay {

ReplaySession::ReplaySession(std::shared_ptr<data::IDataStore> store)
    : store_(std::move(store))
{
    if (!store_) {
        throw std::invalid_argument("ReplaySession requires an IDataStore");
    }
}

void ReplaySession::configure(ReplayConfig config)
{
    if (config.dataset_range.end_ns <= config.dataset_range.start_ns) {
        throw std::invalid_argument("ReplaySession requires a non-empty dataset range");
    }
    if (config.periods.empty()) {
        throw std::invalid_argument("ReplaySession requires at least one period");
    }

    config.max_ticks_per_frame = std::max<std::size_t>(1, config.max_ticks_per_frame);
    config.max_bars_per_period = std::max<std::size_t>(1, config.max_bars_per_period);
    config_ = std::move(config);
    configured_ = true;
    reached_end_ = false;
    playing_ = false;
    current_time_ns_ = clamp_to_dataset(config_.start_time_ns);
    cursor_ns_ = current_time_ns_;
    rebuild_builders();
}

void ReplaySession::set_enabled(bool enabled)
{
    enabled_ = enabled;
    if (!enabled_) {
        playing_ = false;
    }
}

void ReplaySession::set_playing(bool playing)
{
    playing_ = enabled_ && configured_ && !reached_end_ && playing;
}

bool ReplaySession::enabled() const
{
    return enabled_;
}

bool ReplaySession::playing() const
{
    return playing_;
}

bool ReplaySession::toggle_playing()
{
    set_playing(!playing_);
    return playing_;
}

void ReplaySession::set_speed(int speed)
{
    speed_ = std::max(1, speed);
}

int ReplaySession::speed() const
{
    return speed_;
}

void ReplaySession::seek(std::int64_t timestamp_ns)
{
    if (!configured_) {
        return;
    }

    current_time_ns_ = clamp_to_dataset(timestamp_ns);
    cursor_ns_ = current_time_ns_;
    reached_end_ = current_time_ns_ >= config_.dataset_range.end_ns;
    if (reached_end_) {
        playing_ = false;
    }
    rebuild_builders();
}

ReplayAdvanceResult ReplaySession::advance_by(std::int64_t delta_ns)
{
    return advance_to(current_time_ns_ + delta_ns);
}

ReplayAdvanceResult ReplaySession::advance_to(std::int64_t target_time_ns)
{
    if (!configured_ || !enabled_) {
        return current_result();
    }
    if (reached_end_) {
        playing_ = false;
        return current_result();
    }

    const auto target = clamp_to_dataset(target_time_ns);
    if (target < current_time_ns_) {
        seek(target);
        return current_result();
    }
    if (target == current_time_ns_) {
        return current_result();
    }

    const auto chunk = store_->query_replay_ticks(cursor_ns_, target, config_.max_ticks_per_frame);
    const auto rows = std::min(
        {chunk.ticks.timestamp_ns.size(), chunk.ticks.price.size(), chunk.ticks.volume.size()});

    for (std::size_t row = 0; row < rows; ++row) {
        const auto timestamp = chunk.ticks.timestamp_ns[row];
        for (auto& [period, builder] : builders_) {
            (void)period;
            builder.add_tick(timestamp, chunk.ticks.price[row], chunk.ticks.volume[row]);
        }
        cursor_ns_ = timestamp;
        current_time_ns_ = timestamp;
    }

    if (rows == 0) {
        current_time_ns_ = target;
        cursor_ns_ = target;
    }

    reached_end_ = target >= config_.dataset_range.end_ns &&
        (chunk.reached_end || current_time_ns_ >= config_.dataset_range.end_ns);
    if (reached_end_) {
        playing_ = false;
    }

    return current_result(rows);
}

std::optional<data::CandleWindow> ReplaySession::window_for_period(
    std::string_view period,
    std::uint64_t chart_id,
    std::uint64_t generation,
    core::TimeRange visible_range) const
{
    const auto found = builders_.find(std::string{period});
    if (found == builders_.end() || found->second.empty()) {
        return std::nullopt;
    }
    return found->second.to_window(chart_id, generation, visible_range);
}

std::int64_t ReplaySession::current_time_ns() const
{
    return current_time_ns_;
}

bool ReplaySession::reached_end() const
{
    return reached_end_;
}

void ReplaySession::rebuild_builders()
{
    builders_.clear();
    for (const auto& period : config_.periods) {
        builders_.emplace(period, BarBuilder(period, config_.max_bars_per_period));
    }
}

std::int64_t ReplaySession::clamp_to_dataset(std::int64_t timestamp_ns) const
{
    return std::clamp(timestamp_ns, config_.dataset_range.start_ns, config_.dataset_range.end_ns);
}

ReplayAdvanceResult ReplaySession::current_result(std::size_t ticks_consumed) const
{
    return ReplayAdvanceResult{current_time_ns_, ticks_consumed, reached_end_};
}

} // namespace tradereview::replay
