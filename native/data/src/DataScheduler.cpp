#include "tradereview/data/DataScheduler.h"

#include "tradereview/data/WindowCache.h"

#include <QCoreApplication>
#include <QMetaObject>
#include <QObject>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace tradereview::data {
namespace {

using DeliveryGuard = std::function<bool()>;

bool always_deliver()
{
    return true;
}

struct InFlightKey {
    std::uint64_t chart_id = 0;
    std::uint64_t generation = 0;
    std::string period;
    core::TimeRange visible_range;

    [[nodiscard]] bool operator==(const InFlightKey& other) const
    {
        return chart_id == other.chart_id &&
            generation == other.generation &&
            period == other.period &&
            visible_range.start_ns == other.visible_range.start_ns &&
            visible_range.end_ns == other.visible_range.end_ns;
    }
};

struct CallbackTarget {
    bool direct = true;
    std::shared_ptr<std::atomic_bool> receiver_alive;
};

struct WorkItem {
    ScheduledWindowRequest request;
    InFlightKey in_flight_key;
    WindowCacheKey cache_key;
    CallbackTarget target;
    ScheduledWindowCallback callback;
};

CallbackTarget callback_target(QObject* receiver)
{
    if (receiver == nullptr) {
        return CallbackTarget{};
    }
    auto receiver_alive = std::make_shared<std::atomic_bool>(true);
    QObject::connect(
        receiver,
        &QObject::destroyed,
        receiver,
        [receiver_alive]() {
            receiver_alive->store(false);
        });
    return CallbackTarget{false, std::move(receiver_alive)};
}

InFlightKey in_flight_key(const ScheduledWindowRequest& request)
{
    return {
        request.candle_request.chart_id,
        request.candle_request.generation,
        request.candle_request.requested_period,
        request.candle_request.visible_range,
    };
}

WindowCacheKey cache_key(const ScheduledWindowRequest& request)
{
    return {
        request.dataset_path,
        request.candle_request.requested_period,
        request.candle_request.visible_range,
        request.indicator_version,
        request.candle_request.include_indicators,
        request.candle_request.requested_indicators,
    };
}

void post_result(
    const CallbackTarget& target,
    ScheduledWindowCallback callback,
    ScheduledWindowResult result,
    DeliveryGuard delivery_guard = always_deliver)
{
    if (!callback) {
        return;
    }
    if (target.direct) {
        if (!delivery_guard()) {
            return;
        }
        callback(std::move(result));
        return;
    }
    if (!target.receiver_alive || !target.receiver_alive->load()) {
        return;
    }

    auto* app = QCoreApplication::instance();
    if (app == nullptr) {
        return;
    }

    QMetaObject::invokeMethod(
        app,
        [receiver_alive = target.receiver_alive,
            callback = std::move(callback),
            result = std::move(result),
            delivery_guard = std::move(delivery_guard)]() mutable {
            if (!receiver_alive->load() || !delivery_guard()) {
                return;
            }
            callback(std::move(result));
        },
        Qt::QueuedConnection);
}

} // namespace

struct DataScheduler::State : std::enable_shared_from_this<DataScheduler::State> {
    explicit State(std::shared_ptr<IDataStore> input_store, std::size_t cache_capacity)
        : store(std::move(input_store))
        , cache(cache_capacity)
    {
    }

    std::shared_ptr<IDataStore> store;
    WindowCache cache;
    mutable std::mutex mutex;
    mutable std::mutex store_mutex;
    mutable std::mutex queue_mutex;
    std::condition_variable queue_changed;
    std::vector<InFlightKey> in_flight;
    std::map<std::uint64_t, std::uint64_t> current_generation_by_chart;
    std::deque<WorkItem> queue;
    std::thread worker;
    bool stopping = false;

    void start_worker(std::shared_ptr<State> self)
    {
        worker = std::thread([self = std::move(self)]() {
            self->run_worker_loop();
        });
    }

    void stop_worker()
    {
        {
            std::lock_guard lock(queue_mutex);
            stopping = true;
        }
        queue_changed.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void enqueue(WorkItem item)
    {
        {
            std::lock_guard lock(queue_mutex);
            queue.push_back(std::move(item));
        }
        queue_changed.notify_one();
    }

    void run_worker_loop()
    {
        for (;;) {
            WorkItem item;
            {
                std::unique_lock lock(queue_mutex);
                queue_changed.wait(lock, [this]() {
                    return stopping || !queue.empty();
                });
                if (stopping && queue.empty()) {
                    return;
                }
                item = std::move(queue.front());
                queue.pop_front();
            }

            run_item(std::move(item));
        }
    }

    void run_item(WorkItem item)
    {
        if (drop_in_flight_if_stale(item.in_flight_key, item.request.candle_request.chart_id, item.request.candle_request.generation)) {
            return;
        }

        CandleWindow window;
        bool query_ok = true;
        try {
            std::lock_guard store_lock(store_mutex);
            window = store->query_candles(item.request.candle_request);
        } catch (const std::exception&) {
            query_ok = false;
        }

        bool stale = true;
        {
            std::lock_guard lock(mutex);
            const auto in_flight_item = std::find(in_flight.begin(), in_flight.end(), item.in_flight_key);
            if (in_flight_item != in_flight.end()) {
                in_flight.erase(in_flight_item);
            }

            const auto current_generation = current_generation_by_chart.find(item.request.candle_request.chart_id);
            stale = current_generation == current_generation_by_chart.end() ||
                current_generation->second != item.request.candle_request.generation;

            if (query_ok && !stale) {
                cache.put(item.cache_key, window);
            }
        }

        if (!query_ok || stale) {
            return;
        }

        window.from_cache = false;
        ScheduledWindowResult result{item.request, std::move(window), false};
        const auto chart_id = item.request.candle_request.chart_id;
        const auto generation = item.request.candle_request.generation;
        post_result(
            item.target,
            std::move(item.callback),
            std::move(result),
            [self = shared_from_this(), chart_id, generation]() {
                return self->is_current_generation(chart_id, generation);
            });
    }

    [[nodiscard]] bool drop_in_flight_if_stale(
        const InFlightKey& in_flight_key,
        std::uint64_t chart_id,
        std::uint64_t generation)
    {
        std::lock_guard lock(mutex);
        const auto current_generation = current_generation_by_chart.find(chart_id);
        const auto stale = current_generation == current_generation_by_chart.end() ||
            current_generation->second != generation;
        if (!stale) {
            return false;
        }

        const auto in_flight_item = std::find(in_flight.begin(), in_flight.end(), in_flight_key);
        if (in_flight_item != in_flight.end()) {
            in_flight.erase(in_flight_item);
        }
        return true;
    }

    [[nodiscard]] bool is_current_generation(std::uint64_t chart_id, std::uint64_t generation) const
    {
        std::lock_guard lock(mutex);
        const auto current_generation = current_generation_by_chart.find(chart_id);
        return current_generation != current_generation_by_chart.end() &&
            current_generation->second == generation;
    }
};

std::ostream& operator<<(std::ostream& out, ScheduleSubmitStatus status)
{
    switch (status) {
    case ScheduleSubmitStatus::Scheduled:
        return out << "Scheduled";
    case ScheduleSubmitStatus::Coalesced:
        return out << "Coalesced";
    case ScheduleSubmitStatus::CacheHit:
        return out << "CacheHit";
    }
    return out << "Unknown";
}

DataScheduler::DataScheduler(std::shared_ptr<IDataStore> store, std::size_t cache_capacity)
    : state_(std::make_shared<State>(std::move(store), cache_capacity))
{
    if (!state_->store) {
        throw std::invalid_argument("DataScheduler requires an IDataStore");
    }
    state_->start_worker(state_);
}

DataScheduler::~DataScheduler()
{
    if (state_) {
        state_->stop_worker();
    }
}

DataScheduler::DataScheduler(DataScheduler&&) noexcept = default;

DataScheduler& DataScheduler::operator=(DataScheduler&& other) noexcept
{
    if (this != &other) {
        if (state_) {
            state_->stop_worker();
        }
        state_ = std::move(other.state_);
    }
    return *this;
}

DataSetInfo DataScheduler::open_readonly(const std::string& path)
{
    std::lock_guard store_lock(state_->store_mutex);
    return state_->store->open_readonly(path);
}

ScheduleSubmitStatus DataScheduler::submit_window(
    ScheduledWindowRequest request,
    QObject* receiver,
    ScheduledWindowCallback callback)
{
    const auto request_in_flight_key = in_flight_key(request);
    const auto request_cache_key = cache_key(request);
    const auto target = callback_target(receiver);
    std::optional<ScheduledWindowResult> cached_result;
    {
        std::lock_guard lock(state_->mutex);
        state_->current_generation_by_chart[request.candle_request.chart_id] = request.candle_request.generation;

        if (auto cached_window = state_->cache.get(request_cache_key); cached_window.has_value()) {
            cached_window->generation = request.candle_request.generation;
            cached_window->chart_id = request.candle_request.chart_id;
            cached_window->from_cache = true;
            cached_result = ScheduledWindowResult{request, std::move(*cached_window), true};
        }

        if (!cached_result.has_value()) {
            const auto duplicate = std::find(state_->in_flight.begin(), state_->in_flight.end(), request_in_flight_key);
            if (duplicate != state_->in_flight.end()) {
                return ScheduleSubmitStatus::Coalesced;
            }

            state_->in_flight.push_back(request_in_flight_key);
        }
    }

    if (cached_result.has_value()) {
        auto state_for_guard = state_;
        const auto chart_id = request.candle_request.chart_id;
        const auto generation = request.candle_request.generation;
        post_result(
            target,
            std::move(callback),
            std::move(*cached_result),
            [state_for_guard = std::move(state_for_guard), chart_id, generation]() {
                return state_for_guard->is_current_generation(chart_id, generation);
            });
        return ScheduleSubmitStatus::CacheHit;
    }

    state_->enqueue(WorkItem{
        std::move(request),
        request_in_flight_key,
        request_cache_key,
        target,
        std::move(callback),
    });

    return ScheduleSubmitStatus::Scheduled;
}

void DataScheduler::set_current_generation(std::uint64_t chart_id, std::uint64_t generation)
{
    std::lock_guard lock(state_->mutex);
    state_->current_generation_by_chart[chart_id] = generation;
}

std::size_t DataScheduler::in_flight_count() const
{
    std::lock_guard lock(state_->mutex);
    return state_->in_flight.size();
}

} // namespace tradereview::data
