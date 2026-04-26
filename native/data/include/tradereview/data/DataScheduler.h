#pragma once

#include "tradereview/data/CandleWindow.h"
#include "tradereview/data/IDataStore.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

class QObject;

namespace tradereview::data {

struct ScheduledWindowRequest {
    std::string dataset_path;
    std::string indicator_version;
    CandleWindowRequest candle_request;
};

struct ScheduledWindowResult {
    ScheduledWindowRequest request;
    CandleWindow window;
    bool from_cache = false;
};

enum class ScheduleSubmitStatus {
    Scheduled,
    Coalesced,
    CacheHit,
};

std::ostream& operator<<(std::ostream& out, ScheduleSubmitStatus status);

using ScheduledWindowCallback = std::function<void(ScheduledWindowResult)>;

class DataScheduler final {
public:
    explicit DataScheduler(std::shared_ptr<IDataStore> store, std::size_t cache_capacity = 8);
    ~DataScheduler();

    DataScheduler(const DataScheduler&) = delete;
    DataScheduler& operator=(const DataScheduler&) = delete;
    DataScheduler(DataScheduler&&) noexcept;
    DataScheduler& operator=(DataScheduler&&) noexcept;

    [[nodiscard]] DataSetInfo open_readonly(const std::string& path);
    ScheduleSubmitStatus submit_window(ScheduledWindowRequest request, QObject* receiver, ScheduledWindowCallback callback);
    void set_current_generation(std::uint64_t chart_id, std::uint64_t generation);
    [[nodiscard]] std::size_t in_flight_count() const;

private:
    struct State;
    std::shared_ptr<State> state_;
};

} // namespace tradereview::data
