#include "tradereview/data/DataScheduler.h"
#include "tradereview/data/WindowCache.h"

#include "tradereview/core/Assertions.h"
#include "tradereview/data/IDataStore.h"

#include <QCoreApplication>
#include <QObject>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::data::WindowCacheKey cache_key(
    std::string dataset,
    std::string period,
    std::int64_t start_ns,
    std::int64_t end_ns,
    std::string indicator_version)
{
    return tradereview::data::WindowCacheKey{
        std::move(dataset),
        std::move(period),
        {start_ns, end_ns},
        std::move(indicator_version),
        true,
        {},
    };
}

tradereview::data::ScheduledWindowRequest scheduled_request(
    std::uint64_t chart_id,
    std::uint64_t generation,
    std::int64_t start_ns,
    std::int64_t end_ns)
{
    tradereview::data::ScheduledWindowRequest request;
    request.dataset_path = "dataset.duckdb";
    request.indicator_version = "ind-v1";
    request.candle_request.chart_id = chart_id;
    request.candle_request.generation = generation;
    request.candle_request.requested_period = "1min";
    request.candle_request.visible_range = {start_ns, end_ns};
    request.candle_request.include_indicators = true;
    return request;
}

tradereview::data::CandleWindow window_for_generation(std::uint64_t generation, std::int64_t start_ns)
{
    tradereview::data::CandleWindow window;
    window.generation = generation;
    window.timestamp_ns = {start_ns, start_ns + 60LL};
    window.open = {1.0, 2.0};
    window.high = {1.5, 2.5};
    window.low = {0.5, 1.5};
    window.close = {1.25, 2.25};
    window.volume = {10.0, 20.0};
    return window;
}

class FakeStore final : public tradereview::data::IDataStore {
public:
    explicit FakeStore(int delay_ms = 0)
        : delay_ms_(delay_ms)
    {
    }

    tradereview::data::DataSetInfo open_readonly(const std::string& path) override
    {
        ++open_count;
        if (active_queries.load() > 0) {
            ++overlapped_open_count;
        }
        tradereview::data::DataSetInfo info;
        info.dataset_path = path;
        info.indicator_version = "ind-v1";
        info.tick_range = {0, 1000};
        info.available_periods = {"1min"};
        return info;
    }

    tradereview::data::CandleWindow query_candles(const tradereview::data::CandleWindowRequest& request) override
    {
        ++query_count;
        ++active_queries;
        if (delay_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
        }
        auto window = window_for_generation(request.generation, request.visible_range.start_ns);
        window.chart_id = request.chart_id;
        window.requested_period = request.requested_period;
        window.actual_period = request.requested_period;
        window.visible_range = request.visible_range;
        window.loaded_range = request.visible_range;
        --active_queries;
        return window;
    }

    tradereview::data::TickSlice query_ticks(tradereview::core::TimeRange, std::size_t) override
    {
        return {};
    }

    tradereview::data::ReplayChunk query_replay_ticks(std::int64_t, std::int64_t, std::size_t) override
    {
        return {};
    }

    std::atomic<int> query_count = 0;
    std::atomic<int> active_queries = 0;
    std::atomic<int> open_count = 0;
    std::atomic<int> overlapped_open_count = 0;

private:
    int delay_ms_ = 0;
};

bool wait_until_empty(const tradereview::data::DataScheduler& scheduler)
{
    for (int attempts = 0; attempts < 200; ++attempts) {
        if (scheduler.in_flight_count() == 0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

bool wait_until_active_query(const FakeStore& store)
{
    for (int attempts = 0; attempts < 200; ++attempts) {
        if (store.active_queries.load() > 0) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

bool wait_until_count(const std::atomic<int>& value, int expected)
{
    for (int attempts = 0; attempts < 200; ++attempts) {
        if (value.load() == expected) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

QCoreApplication& test_application()
{
    static int argc = 1;
    static char app_name[] = "tradereview_native_tests";
    static char* argv[] = {app_name, nullptr};
    static QCoreApplication app(argc, argv);
    return app;
}

void test_window_cache_evicts_least_recently_used_window()
{
    tradereview::data::WindowCache cache(2);
    const auto key_a = cache_key("a.duckdb", "1min", 0, 10, "v1");
    const auto key_b = cache_key("b.duckdb", "1min", 0, 10, "v1");
    const auto key_c = cache_key("c.duckdb", "1min", 0, 10, "v1");

    cache.put(key_a, window_for_generation(1, 0));
    cache.put(key_b, window_for_generation(1, 10));
    tradereview::core::assert_true(cache.get(key_a).has_value(), "cache hit promotes key a");
    cache.put(key_c, window_for_generation(1, 20));

    tradereview::core::assert_true(cache.get(key_a).has_value(), "promoted key a remains cached");
    tradereview::core::assert_true(!cache.get(key_b).has_value(), "least recently used key b is evicted");
    tradereview::core::assert_true(cache.get(key_c).has_value(), "new key c is cached");
}

void test_data_scheduler_coalesces_duplicate_in_flight_requests()
{
    auto store = std::make_shared<FakeStore>(40);
    tradereview::data::DataScheduler scheduler(store, 4);
    const auto request = scheduled_request(1, 1, 0, 100);
    std::atomic<int> callbacks = 0;

    const auto first_status = scheduler.submit_window(request, nullptr, [&callbacks](tradereview::data::ScheduledWindowResult) {
        ++callbacks;
    });
    const auto second_status = scheduler.submit_window(request, nullptr, [&callbacks](tradereview::data::ScheduledWindowResult) {
        ++callbacks;
    });

    tradereview::core::assert_equal(first_status, tradereview::data::ScheduleSubmitStatus::Scheduled, "first request is scheduled");
    tradereview::core::assert_equal(second_status, tradereview::data::ScheduleSubmitStatus::Coalesced, "duplicate request is coalesced");
    tradereview::core::assert_true(wait_until_empty(scheduler), "scheduler drains duplicate request");
    tradereview::core::assert_true(wait_until_count(callbacks, 1), "first callback receives result");
    tradereview::core::assert_equal(store->query_count.load(), 1, "only one query runs");
    tradereview::core::assert_equal(callbacks.load(), 1, "only first callback receives result");
}

void test_data_scheduler_uses_window_cache()
{
    auto store = std::make_shared<FakeStore>();
    tradereview::data::DataScheduler scheduler(store, 4);
    const auto first_request = scheduled_request(1, 1, 0, 100);
    auto second_request = first_request;
    second_request.candle_request.generation = 2;
    std::atomic<int> cached_callbacks = 0;
    std::atomic<int> cache_hit_callbacks = 0;

    scheduler.submit_window(first_request, nullptr, [&cached_callbacks](tradereview::data::ScheduledWindowResult result) {
        if (!result.from_cache) {
            ++cached_callbacks;
        }
    });
    tradereview::core::assert_true(wait_until_empty(scheduler), "first cacheable request drains");
    tradereview::core::assert_true(wait_until_count(cached_callbacks, 1), "first uncached callback receives result");

    const auto status = scheduler.submit_window(second_request, nullptr, [&cache_hit_callbacks](tradereview::data::ScheduledWindowResult result) {
        if (result.from_cache) {
            ++cache_hit_callbacks;
        }
    });

    tradereview::core::assert_equal(status, tradereview::data::ScheduleSubmitStatus::CacheHit, "same range uses cache across generations");
    tradereview::core::assert_equal(store->query_count.load(), 1, "cache hit does not query store again");
    tradereview::core::assert_equal(cached_callbacks.load(), 1, "first callback was uncached");
    tradereview::core::assert_equal(cache_hit_callbacks.load(), 1, "second callback was cached");
}

void test_data_scheduler_posts_results_through_queued_callback()
{
    auto& app = test_application();
    QObject receiver;
    auto store = std::make_shared<FakeStore>();
    tradereview::data::DataScheduler scheduler(store, 4);
    const auto request = scheduled_request(1, 1, 0, 100);
    std::atomic<int> callbacks = 0;

    const auto status = scheduler.submit_window(request, &receiver, [&callbacks](tradereview::data::ScheduledWindowResult) {
        ++callbacks;
    });

    tradereview::core::assert_equal(status, tradereview::data::ScheduleSubmitStatus::Scheduled, "receiver request is scheduled");
    tradereview::core::assert_true(wait_until_empty(scheduler), "receiver request drains");
    tradereview::core::assert_equal(callbacks.load(), 0, "queued callback does not run before event processing");

    for (int attempts = 0; attempts < 50 && callbacks.load() == 0; ++attempts) {
        app.processEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    tradereview::core::assert_equal(callbacks.load(), 1, "queued callback runs on event processing");
}

void test_data_scheduler_serializes_open_with_in_flight_window_query()
{
    auto store = std::make_shared<FakeStore>(40);
    tradereview::data::DataScheduler scheduler(store, 4);
    const auto request = scheduled_request(1, 1, 0, 100);

    scheduler.submit_window(request, nullptr, [](tradereview::data::ScheduledWindowResult) {});
    tradereview::core::assert_true(wait_until_active_query(*store), "query is active before open");

    const auto info = scheduler.open_readonly("next.duckdb");

    tradereview::core::assert_true(wait_until_empty(scheduler), "scheduler drains serialized open request");
    tradereview::core::assert_equal(info.dataset_path, std::string{"next.duckdb"}, "serialized open returns metadata");
    tradereview::core::assert_equal(store->query_count.load(), 1, "window query still ran once");
    tradereview::core::assert_equal(store->open_count.load(), 1, "open ran once");
    tradereview::core::assert_equal(store->overlapped_open_count.load(), 0, "open does not overlap active query");
}

void test_data_scheduler_drops_callback_when_receiver_is_destroyed()
{
    auto& app = test_application();
    auto receiver = std::make_unique<QObject>();
    auto store = std::make_shared<FakeStore>(30);
    tradereview::data::DataScheduler scheduler(store, 4);
    const auto request = scheduled_request(1, 1, 0, 100);
    std::atomic<int> callbacks = 0;

    scheduler.submit_window(request, receiver.get(), [&callbacks](tradereview::data::ScheduledWindowResult) {
        ++callbacks;
    });
    receiver.reset();

    tradereview::core::assert_true(wait_until_empty(scheduler), "destroyed receiver request drains");
    for (int attempts = 0; attempts < 20; ++attempts) {
        app.processEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    tradereview::core::assert_equal(callbacks.load(), 0, "destroyed receiver callback is dropped");
}

void test_data_scheduler_drops_stale_queued_generation_results()
{
    auto& app = test_application();
    QObject receiver;
    auto store = std::make_shared<FakeStore>();
    tradereview::data::DataScheduler scheduler(store, 4);
    const auto request = scheduled_request(1, 1, 0, 100);
    std::atomic<int> callbacks = 0;

    scheduler.submit_window(request, &receiver, [&callbacks](tradereview::data::ScheduledWindowResult) {
        ++callbacks;
    });

    tradereview::core::assert_true(wait_until_empty(scheduler), "queued stale request drains");
    scheduler.set_current_generation(1, 2);
    for (int attempts = 0; attempts < 20; ++attempts) {
        app.processEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    tradereview::core::assert_equal(callbacks.load(), 0, "queued stale generation callback is dropped");
}

void test_data_scheduler_skips_queries_for_queued_stale_requests()
{
    auto store = std::make_shared<FakeStore>(40);
    tradereview::data::DataScheduler scheduler(store, 4);
    const auto active_request = scheduled_request(1, 1, 0, 100);
    const auto stale_queued_request = scheduled_request(1, 2, 100, 200);
    const auto current_queued_request = scheduled_request(1, 3, 200, 300);
    std::atomic<int> callbacks = 0;

    scheduler.submit_window(active_request, nullptr, [](tradereview::data::ScheduledWindowResult) {});
    tradereview::core::assert_true(wait_until_active_query(*store), "first query is active before queuing stale request");
    scheduler.submit_window(stale_queued_request, nullptr, [&callbacks](tradereview::data::ScheduledWindowResult) {
        ++callbacks;
    });
    scheduler.submit_window(current_queued_request, nullptr, [&callbacks](tradereview::data::ScheduledWindowResult) {
        ++callbacks;
    });

    tradereview::core::assert_true(wait_until_empty(scheduler), "scheduler drains queued stale request");
    tradereview::core::assert_true(wait_until_count(callbacks, 1), "only current queued request callback runs");
    tradereview::core::assert_equal(store->query_count.load(), 2, "queued stale request does not query store");
}

void test_data_scheduler_drops_stale_generation_results()
{
    auto store = std::make_shared<FakeStore>(40);
    tradereview::data::DataScheduler scheduler(store, 4);
    const auto request = scheduled_request(1, 1, 0, 100);
    std::atomic<int> callbacks = 0;

    scheduler.submit_window(request, nullptr, [&callbacks](tradereview::data::ScheduledWindowResult) {
        ++callbacks;
    });
    tradereview::core::assert_true(wait_until_active_query(*store), "stale request query starts before generation changes");
    scheduler.set_current_generation(1, 2);

    tradereview::core::assert_true(wait_until_empty(scheduler), "scheduler drains stale request");
    tradereview::core::assert_equal(callbacks.load(), 0, "stale generation callback is dropped");
    tradereview::core::assert_equal(store->query_count.load(), 1, "stale request still queried once");
}

struct RegisterWindowCacheTests {
    RegisterWindowCacheTests()
    {
        tradereview::tests::register_test(
            "window cache evicts least recently used window",
            test_window_cache_evicts_least_recently_used_window);
        tradereview::tests::register_test(
            "data scheduler coalesces duplicate in-flight requests",
            test_data_scheduler_coalesces_duplicate_in_flight_requests);
        tradereview::tests::register_test(
            "data scheduler uses window cache",
            test_data_scheduler_uses_window_cache);
        tradereview::tests::register_test(
            "data scheduler posts results through queued callback",
            test_data_scheduler_posts_results_through_queued_callback);
        tradereview::tests::register_test(
            "data scheduler serializes open with in-flight window query",
            test_data_scheduler_serializes_open_with_in_flight_window_query);
        tradereview::tests::register_test(
            "data scheduler drops callback when receiver is destroyed",
            test_data_scheduler_drops_callback_when_receiver_is_destroyed);
        tradereview::tests::register_test(
            "data scheduler drops stale queued generation results",
            test_data_scheduler_drops_stale_queued_generation_results);
        tradereview::tests::register_test(
            "data scheduler skips queries for queued stale requests",
            test_data_scheduler_skips_queries_for_queued_stale_requests);
        tradereview::tests::register_test(
            "data scheduler drops stale generation results",
            test_data_scheduler_drops_stale_generation_results);
    }
};

const RegisterWindowCacheTests register_window_cache_tests;

} // namespace
