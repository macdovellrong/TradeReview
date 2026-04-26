#include "tradereview/app/DataLoadController.h"

#include "tradereview/chart/ChartViewWidget.h"
#include "tradereview/chart/ChartWorkspaceWidget.h"
#include "tradereview/data/DuckDbRepository.h"
#include "tradereview/data/IDataStore.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <QObject>
#include <stdexcept>
#include <string>
#include <utility>

namespace tradereview::app {
namespace {

constexpr std::int64_t kNanosecondsPerHour = 60LL * 60LL * 1000LL * 1000LL * 1000LL;
constexpr std::int64_t kInitialWindowWidthNs = 6LL * kNanosecondsPerHour;
constexpr int kDefaultPixelWidth = 1200;

std::string toPathString(const QString& path)
{
    const auto utf8 = path.toUtf8();
    return {utf8.constData(), static_cast<std::size_t>(utf8.size())};
}

core::TimeRange initialVisibleRange(core::TimeRange tick_range)
{
    if (tick_range.end_ns <= tick_range.start_ns) {
        throw std::runtime_error("dataset has an empty tick time range");
    }

    const auto span = tick_range.end_ns - tick_range.start_ns;
    if (span <= kInitialWindowWidthNs) {
        return tick_range;
    }

    const auto midpoint = tick_range.start_ns + span / 2;
    const auto half_width = kInitialWindowWidthNs / 2;
    auto start = midpoint - half_width;
    auto end = midpoint + half_width;

    if (start < tick_range.start_ns) {
        start = tick_range.start_ns;
        end = start + kInitialWindowWidthNs;
    }
    if (end > tick_range.end_ns) {
        end = tick_range.end_ns;
        start = end - kInitialWindowWidthNs;
    }

    return {start, end};
}

int chartPixelWidth(const chart::ChartWorkspaceWidget& workspace)
{
    return std::max(workspace.chart_view().width(), kDefaultPixelWidth);
}

std::string datasetPathForRequest(const std::string& opened_path, const data::DataSetInfo& dataset_info)
{
    if (!dataset_info.dataset_path.empty()) {
        return dataset_info.dataset_path;
    }
    return opened_path;
}

std::string requestedPeriod(const data::DataSetInfo& dataset_info)
{
    const auto one_minute = std::find(
        dataset_info.available_periods.begin(),
        dataset_info.available_periods.end(),
        "1min");
    if (one_minute != dataset_info.available_periods.end()) {
        return *one_minute;
    }
    if (!dataset_info.available_periods.empty()) {
        return dataset_info.available_periods.front();
    }
    return "1min";
}

data::CandleWindowRequest makeWindowRequest(
    chart::ChartWorkspaceWidget& workspace,
    core::TimeRange visible_range,
    const std::string& requested_period)
{
    data::CandleWindowRequest request;
    request.chart_id = 1;
    request.generation = workspace.chart_view().bump_generation();
    request.requested_period = requested_period;
    request.visible_range = visible_range;
    request.pixel_width = chartPixelWidth(workspace);
    return request;
}

} // namespace

DataLoadController::DataLoadController()
    : DataLoadController(std::unique_ptr<data::IDataStore>(std::make_unique<data::DuckDbRepository>()))
{
}

DataLoadController::DataLoadController(std::unique_ptr<data::IDataStore> store)
    : DataLoadController(std::shared_ptr<data::IDataStore>(std::move(store)))
{
}

DataLoadController::DataLoadController(std::shared_ptr<data::IDataStore> store)
    : store_(std::move(store))
    , scheduler_(std::make_unique<data::DataScheduler>(store_))
{
    if (!store_) {
        throw std::invalid_argument("DataLoadController requires an IDataStore");
    }
}

DataLoadController::~DataLoadController() = default;

DataLoadController::DataLoadController(DataLoadController&&) noexcept = default;

DataLoadController& DataLoadController::operator=(DataLoadController&&) noexcept = default;

data::ScheduleSubmitStatus DataLoadController::load_file_async(
    const QString& path,
    chart::ChartWorkspaceWidget& workspace,
    QObject* receiver,
    LoadCallback callback)
{
    if (receiver == nullptr) {
        throw std::invalid_argument("load_file_async requires a QObject receiver");
    }

    const auto opened_path = toPathString(path);
    dataset_info_ = scheduler_->open_readonly(opened_path);
    dataset_path_ = datasetPathForRequest(opened_path, dataset_info_);
    requested_period_ = requestedPeriod(dataset_info_);

    auto request = makeWindowRequest(workspace, initialVisibleRange(dataset_info_.tick_range), requested_period_);
    return submit_window_async(std::move(request), workspace, receiver, std::move(callback));
}

data::ScheduleSubmitStatus DataLoadController::request_window_async(
    core::TimeRange visible_range,
    chart::ChartWorkspaceWidget& workspace,
    QObject* receiver,
    LoadCallback callback)
{
    if (receiver == nullptr) {
        throw std::invalid_argument("request_window_async requires a QObject receiver");
    }
    if (dataset_path_.empty()) {
        throw std::runtime_error("no dataset loaded");
    }

    auto request = makeWindowRequest(workspace, visible_range, requested_period_);
    return submit_window_async(std::move(request), workspace, receiver, std::move(callback));
}

data::ScheduleSubmitStatus DataLoadController::submit_window_async(
    data::CandleWindowRequest request,
    chart::ChartWorkspaceWidget& workspace,
    QObject* receiver,
    LoadCallback callback)
{
    data::ScheduledWindowRequest scheduled_request;
    scheduled_request.dataset_path = dataset_path_;
    scheduled_request.indicator_version = dataset_info_.indicator_version;
    scheduled_request.candle_request = std::move(request);

    auto dataset_info = dataset_info_;
    auto* workspace_ptr = &workspace;
    return scheduler_->submit_window(
        std::move(scheduled_request),
        receiver,
        [dataset_info = std::move(dataset_info), workspace_ptr, callback = std::move(callback)](data::ScheduledWindowResult result) mutable {
            result.window.from_cache = result.from_cache;
            LoadResult load_result{dataset_info, std::move(result.window)};
            const auto accepted = workspace_ptr->apply_window(data::CandleWindow{load_result.window});
            if (accepted && callback) {
                callback(std::move(load_result));
            }
        });
}

} // namespace tradereview::app
