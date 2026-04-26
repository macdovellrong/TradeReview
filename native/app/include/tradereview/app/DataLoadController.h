#pragma once

#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"
#include "tradereview/data/DataScheduler.h"
#include "tradereview/data/DataSetInfo.h"
#include "tradereview/replay/ReplaySession.h"

#include <QString>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

class QObject;

namespace tradereview::chart {
class ChartWorkspaceWidget;
}

namespace tradereview::data {
class IDataStore;
}

namespace tradereview::app {

struct LoadResult {
    data::DataSetInfo dataset_info;
    data::CandleWindow window;
};

struct ReplayUpdateResult {
    std::int64_t current_time_ns = 0;
    std::size_t ticks_consumed = 0;
    bool enabled = false;
    bool playing = false;
    bool reached_end = false;
    bool applied_window = false;
};

class DataLoadController final {
public:
    using LoadCallback = std::function<void(LoadResult)>;

    DataLoadController();
    explicit DataLoadController(std::unique_ptr<data::IDataStore> store);
    explicit DataLoadController(std::shared_ptr<data::IDataStore> store);
    ~DataLoadController();

    DataLoadController(const DataLoadController&) = delete;
    DataLoadController& operator=(const DataLoadController&) = delete;
    DataLoadController(DataLoadController&&) noexcept;
    DataLoadController& operator=(DataLoadController&&) noexcept;

    data::ScheduleSubmitStatus load_file_async(
        const QString& path,
        chart::ChartWorkspaceWidget& workspace,
        QObject* receiver,
        LoadCallback callback);
    data::ScheduleSubmitStatus request_window_async(
        core::TimeRange visible_range,
        chart::ChartWorkspaceWidget& workspace,
        QObject* receiver,
        LoadCallback callback);
    data::ScheduleSubmitStatus request_window_async(
        std::uint64_t chart_id,
        core::TimeRange visible_range,
        chart::ChartWorkspaceWidget& workspace,
        QObject* receiver,
        LoadCallback callback);
    void set_replay_enabled(bool enabled, chart::ChartWorkspaceWidget& workspace);
    [[nodiscard]] bool replay_enabled() const;
    [[nodiscard]] bool replay_playing() const;
    [[nodiscard]] bool toggle_replay_playing();
    void set_replay_speed(int speed);
    [[nodiscard]] int replay_speed() const;
    [[nodiscard]] ReplayUpdateResult advance_replay_by(std::int64_t delta_ns, chart::ChartWorkspaceWidget& workspace);
    [[nodiscard]] ReplayUpdateResult advance_replay_by_speed(chart::ChartWorkspaceWidget& workspace);

private:
    data::ScheduleSubmitStatus submit_window_async(
        data::CandleWindowRequest request,
        chart::ChartWorkspaceWidget& workspace,
        QObject* receiver,
        LoadCallback callback);
    void configure_replay(chart::ChartWorkspaceWidget& workspace, std::int64_t start_time_ns);
    [[nodiscard]] ReplayUpdateResult apply_replay_windows(
        const replay::ReplayAdvanceResult& frame,
        chart::ChartWorkspaceWidget& workspace);

    std::shared_ptr<data::IDataStore> store_;
    std::unique_ptr<data::DataScheduler> scheduler_;
    std::unique_ptr<replay::ReplaySession> replay_session_;
    data::DataSetInfo dataset_info_;
    std::string dataset_path_;
    std::string requested_period_ = "1min";
};

} // namespace tradereview::app
