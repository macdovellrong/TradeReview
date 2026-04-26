#include "tradereview/core/Assertions.h"
#include "tradereview/replay/ReplaySession.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

constexpr std::int64_t kSecondNs = 1000LL * 1000LL * 1000LL;
constexpr std::int64_t kMinuteNs = 60LL * kSecondNs;

class FakeReplayStore final : public tradereview::data::IDataStore {
public:
    explicit FakeReplayStore(std::vector<std::int64_t> timestamps)
        : timestamps_(std::move(timestamps))
    {
    }

    tradereview::data::DataSetInfo open_readonly(const std::string&) override
    {
        return {};
    }

    tradereview::data::CandleWindow query_candles(const tradereview::data::CandleWindowRequest&) override
    {
        return {};
    }

    tradereview::data::TickSlice query_ticks(tradereview::core::TimeRange, std::size_t) override
    {
        return {};
    }

    tradereview::data::ReplayChunk query_replay_ticks(
        std::int64_t from_ns,
        std::int64_t to_ns,
        std::size_t max_ticks) override
    {
        last_from_ns = from_ns;
        last_to_ns = to_ns;
        last_max_ticks = max_ticks;
        ++query_count;

        tradereview::data::ReplayChunk chunk;
        for (const auto timestamp : timestamps_) {
            if (timestamp <= from_ns || timestamp > to_ns) {
                continue;
            }
            if (chunk.ticks.timestamp_ns.size() >= max_ticks) {
                break;
            }
            chunk.ticks.timestamp_ns.push_back(timestamp);
            chunk.ticks.price.push_back(static_cast<double>(timestamp / kSecondNs));
            chunk.ticks.volume.push_back(1.0);
        }

        const auto first_after_to = std::find_if(
            timestamps_.begin(),
            timestamps_.end(),
            [to_ns](std::int64_t timestamp) {
                return timestamp > to_ns;
            });
        const auto capped = chunk.ticks.timestamp_ns.size() == max_ticks;
        chunk.reached_end = !capped && first_after_to == timestamps_.end();
        return chunk;
    }

    std::int64_t last_from_ns = 0;
    std::int64_t last_to_ns = 0;
    std::size_t last_max_ticks = 0;
    int query_count = 0;

private:
    std::vector<std::int64_t> timestamps_;
};

tradereview::replay::ReplaySession make_session(
    std::shared_ptr<FakeReplayStore> store,
    std::int64_t dataset_end_ns = 10LL * kMinuteNs)
{
    tradereview::replay::ReplaySession session(std::move(store));
    tradereview::replay::ReplayConfig config;
    config.dataset_range = {0, dataset_end_ns};
    config.periods = {"1min", "5min"};
    config.start_time_ns = 0;
    config.max_ticks_per_frame = 2;
    config.max_bars_per_period = 10;
    session.configure(config);
    session.set_enabled(true);
    return session;
}

void test_replay_session_advances_with_tick_cap()
{
    auto store = std::make_shared<FakeReplayStore>(std::vector<std::int64_t>{
        1LL * kSecondNs,
        2LL * kSecondNs,
        3LL * kSecondNs});
    auto session = make_session(store);

    const auto result = session.advance_to(5LL * kSecondNs);

    tradereview::core::assert_equal(result.ticks_consumed, std::size_t{2}, "tick cap consumed count");
    tradereview::core::assert_equal(result.current_time_ns, 2LL * kSecondNs, "current time stops at capped tick");
    tradereview::core::assert_true(!result.reached_end, "cap does not report dataset end");
    tradereview::core::assert_equal(store->last_from_ns, std::int64_t{0}, "query from cursor");
    tradereview::core::assert_equal(store->last_to_ns, 5LL * kSecondNs, "query to target");
    tradereview::core::assert_equal(store->last_max_ticks, std::size_t{2}, "query max ticks");

    const auto one_minute = session.window_for_period("1min", 4, 9, {0, kMinuteNs});
    tradereview::core::assert_true(one_minute.has_value(), "1min replay window exists");
    tradereview::core::assert_equal(one_minute->row_count(), std::size_t{1}, "1min active row");
    tradereview::core::assert_near(one_minute->close.front(), 2.0, 0.000001, "active close after capped advance");
}

void test_replay_session_stops_playback_at_dataset_end()
{
    auto store = std::make_shared<FakeReplayStore>(std::vector<std::int64_t>{
        1LL * kSecondNs,
        2LL * kSecondNs});
    auto session = make_session(store, 2LL * kSecondNs);
    session.set_playing(true);

    const auto result = session.advance_to(10LL * kMinuteNs);

    tradereview::core::assert_true(result.reached_end, "dataset end reached");
    tradereview::core::assert_true(!session.playing(), "playback is paused at dataset end");
    tradereview::core::assert_equal(result.current_time_ns, 2LL * kSecondNs, "current time is last tick");
}

void test_replay_session_seek_resets_builders()
{
    auto store = std::make_shared<FakeReplayStore>(std::vector<std::int64_t>{
        1LL * kSecondNs,
        2LL * kSecondNs,
        3LL * kMinuteNs});
    auto session = make_session(store);

    static_cast<void>(session.advance_to(5LL * kSecondNs));
    session.seek(3LL * kMinuteNs);

    const auto window = session.window_for_period("1min", 1, 1, {0, 4LL * kMinuteNs});
    tradereview::core::assert_true(!window.has_value(), "seek clears prior bars");
    tradereview::core::assert_equal(session.current_time_ns(), 3LL * kMinuteNs, "seek updates current time");
}

struct RegisterReplaySessionTests {
    RegisterReplaySessionTests()
    {
        tradereview::tests::register_test(
            "replay session advances with tick cap",
            test_replay_session_advances_with_tick_cap);
        tradereview::tests::register_test(
            "replay session stops playback at dataset end",
            test_replay_session_stops_playback_at_dataset_end);
        tradereview::tests::register_test(
            "replay session seek resets builders",
            test_replay_session_seek_resets_builders);
    }
};

const RegisterReplaySessionTests register_replay_session_tests;

} // namespace
