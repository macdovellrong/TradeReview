#pragma once

#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"
#include "tradereview/data/DataScheduler.h"
#include "tradereview/data/DataSetInfo.h"

#include <QString>

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

private:
    data::ScheduleSubmitStatus submit_window_async(
        data::CandleWindowRequest request,
        chart::ChartWorkspaceWidget& workspace,
        QObject* receiver,
        LoadCallback callback);

    std::shared_ptr<data::IDataStore> store_;
    std::unique_ptr<data::DataScheduler> scheduler_;
    data::DataSetInfo dataset_info_;
    std::string dataset_path_;
    std::string requested_period_ = "1min";
};

} // namespace tradereview::app
