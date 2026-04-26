#include "tradereview/app/DataLoadController.h"

#include "tradereview/app/TimeNavigation.h"
#include "tradereview/chart/ChartViewWidget.h"
#include "tradereview/chart/ChartWorkspaceWidget.h"
#include "tradereview/core/Period.h"
#include "tradereview/data/DuckDbRepository.h"
#include "tradereview/data/IDataStore.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <QObject>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tradereview::app {
namespace {

constexpr std::int64_t kNanosecondsPerHour = 60LL * 60LL * 1000LL * 1000LL * 1000LL;
constexpr std::int64_t kNanosecondsPerSecond = 1000LL * 1000LL * 1000LL;
constexpr std::int64_t kInitialWindowWidthNs = 6LL * kNanosecondsPerHour;
constexpr int kDefaultPixelWidth = 1200;
constexpr std::size_t kReplayMaxTicksPerFrame = 20000;
constexpr std::size_t kReplayMaxBarsPerPeriod = 1200;

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

int chartPixelWidth(const chart::ChartWorkspaceWidget& workspace, std::uint64_t chart_id)
{
    return std::max(workspace.chart_pixel_width(chart_id), kDefaultPixelWidth);
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

std::vector<std::string> replayPeriods(const chart::ChartWorkspaceWidget& workspace, const std::string& fallback_period)
{
    std::vector<std::string> periods;
    for (const auto chart_id : workspace.enabled_chart_ids()) {
        auto period = workspace.requested_period(chart_id);
        if (period.empty()) {
            period = fallback_period;
        }
        if (std::find(periods.begin(), periods.end(), period) == periods.end()) {
            periods.push_back(std::move(period));
        }
    }
    if (periods.empty()) {
        periods.push_back(fallback_period);
    }
    return periods;
}

core::TimeRange replayVisibleRange(
    const replay::ReplaySession& session,
    std::string_view period,
    const data::DataSetInfo& dataset_info)
{
    auto period_ns = 60LL * kNanosecondsPerSecond;
    if (const auto seconds = core::try_period_seconds(period); seconds.has_value()) {
        period_ns = *seconds * kNanosecondsPerSecond;
    }

    const auto current = session.current_time_ns();
    const auto start = std::max(dataset_info.tick_range.start_ns, current - (300LL * period_ns));
    auto end = std::min(dataset_info.tick_range.end_ns, current + (20LL * period_ns));
    if (end <= start) {
        end = start + period_ns;
    }
    return {start, end};
}

std::int64_t replayStartTime(const chart::ChartWorkspaceWidget& workspace, const data::DataSetInfo& dataset_info)
{
    try {
        const auto& window = workspace.chart_view(workspace.active_chart_id()).scene_model().window();
        if (window.has_visible_range()) {
            return std::clamp(window.visible_range.start_ns, dataset_info.tick_range.start_ns, dataset_info.tick_range.end_ns);
        }
    } catch (const std::exception&) {
    }
    return dataset_info.tick_range.start_ns;
}

std::int64_t midpoint(core::TimeRange range)
{
    return range.start_ns + ((range.end_ns - range.start_ns) / 2);
}

std::int64_t activeVisibleWidthNs(const chart::ChartWorkspaceWidget& workspace)
{
    try {
        const auto& window = workspace.chart_view(workspace.active_chart_id()).scene_model().window();
        if (window.has_visible_range() && window.visible_range.end_ns > window.visible_range.start_ns) {
            return window.visible_range.end_ns - window.visible_range.start_ns;
        }
    } catch (const std::exception&) {
    }
    return kInitialWindowWidthNs;
}

data::CandleWindowRequest makeWindowRequest(
    chart::ChartWorkspaceWidget& workspace,
    std::uint64_t chart_id,
    core::TimeRange visible_range,
    const std::string& fallback_period)
{
    data::CandleWindowRequest request;
    request.chart_id = chart_id;
    request.generation = workspace.chart_view(chart_id).bump_generation();
    request.requested_period = workspace.requested_period(chart_id);
    if (request.requested_period.empty()) {
        request.requested_period = fallback_period;
    }
    request.visible_range = visible_range;
    request.pixel_width = chartPixelWidth(workspace, chart_id);
    request.requested_indicators = workspace.requested_indicators(chart_id);
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
    , replay_session_(std::make_unique<replay::ReplaySession>(store_))
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
    replay_session_->set_enabled(false);

    const auto visible_range = initialVisibleRange(dataset_info_.tick_range);
    auto first_status = data::ScheduleSubmitStatus::Scheduled;
    first_status = request_all_enabled_windows_async(visible_range, workspace, receiver, std::move(callback));
    return first_status;
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

    return request_window_async(workspace.active_chart_id(), visible_range, workspace, receiver, std::move(callback));
}

data::ScheduleSubmitStatus DataLoadController::request_window_async(
    std::uint64_t chart_id,
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

    auto request = makeWindowRequest(workspace, chart_id, visible_range, requested_period_);
    return submit_window_async(std::move(request), workspace, receiver, std::move(callback));
}

data::ScheduleSubmitStatus DataLoadController::reset_view_async(
    chart::ChartWorkspaceWidget& workspace,
    QObject* receiver,
    LoadCallback callback)
{
    if (receiver == nullptr) {
        throw std::invalid_argument("reset_view_async requires a QObject receiver");
    }
    if (dataset_path_.empty()) {
        throw std::runtime_error("no dataset loaded");
    }

    return request_all_enabled_windows_async(initialVisibleRange(dataset_info_.tick_range), workspace, receiver, std::move(callback));
}

data::ScheduleSubmitStatus DataLoadController::jump_to_time_async(
    std::int64_t target_time_ns,
    chart::ChartWorkspaceWidget& workspace,
    QObject* receiver,
    LoadCallback callback)
{
    if (receiver == nullptr) {
        throw std::invalid_argument("jump_to_time_async requires a QObject receiver");
    }
    if (dataset_path_.empty()) {
        throw std::runtime_error("no dataset loaded");
    }

    const auto target = clamp_jump_timestamp_ns(
        normalize_jump_timestamp_ns(target_time_ns),
        dataset_info_.tick_range);
    const auto visible_range = centered_visible_range(
        target,
        dataset_info_.tick_range,
        activeVisibleWidthNs(workspace));

    if (replay_session_->enabled()) {
        configure_replay(workspace, target);
        replay_session_->set_enabled(true);
    }

    return request_all_enabled_windows_async(visible_range, workspace, receiver, std::move(callback));
}

data::ScheduleSubmitStatus DataLoadController::step_time_async(
    std::int64_t delta_ns,
    chart::ChartWorkspaceWidget& workspace,
    QObject* receiver,
    LoadCallback callback)
{
    return jump_to_time_async(current_view_center_time_ns(workspace) + delta_ns, workspace, receiver, std::move(callback));
}

bool DataLoadController::dataset_loaded() const
{
    return !dataset_path_.empty();
}

const data::DataSetInfo& DataLoadController::dataset_info() const
{
    return dataset_info_;
}

const std::string& DataLoadController::dataset_path() const
{
    return dataset_path_;
}

std::int64_t DataLoadController::current_view_center_time_ns(const chart::ChartWorkspaceWidget& workspace) const
{
    if (dataset_path_.empty()) {
        throw std::runtime_error("no dataset loaded");
    }

    try {
        const auto& window = workspace.chart_view(workspace.active_chart_id()).scene_model().window();
        if (window.has_visible_range() && window.visible_range.end_ns > window.visible_range.start_ns) {
            return normalize_jump_timestamp_ns(
                clamp_jump_timestamp_ns(midpoint(window.visible_range), dataset_info_.tick_range));
        }
    } catch (const std::exception&) {
    }

    return normalize_jump_timestamp_ns(midpoint(dataset_info_.tick_range));
}

void DataLoadController::set_error_callback(ErrorCallback callback)
{
    error_callback_ = std::move(callback);
}

void DataLoadController::set_replay_enabled(bool enabled, chart::ChartWorkspaceWidget& workspace)
{
    if (dataset_path_.empty()) {
        throw std::runtime_error("no dataset loaded");
    }
    if (enabled) {
        configure_replay(workspace, replayStartTime(workspace, dataset_info_));
    }
    replay_session_->set_enabled(enabled);
}

bool DataLoadController::replay_enabled() const
{
    return replay_session_->enabled();
}

bool DataLoadController::replay_playing() const
{
    return replay_session_->playing();
}

bool DataLoadController::toggle_replay_playing()
{
    if (dataset_path_.empty()) {
        throw std::runtime_error("no dataset loaded");
    }
    return replay_session_->toggle_playing();
}

void DataLoadController::set_replay_speed(int speed)
{
    replay_session_->set_speed(speed);
}

int DataLoadController::replay_speed() const
{
    return replay_session_->speed();
}

ReplayUpdateResult DataLoadController::advance_replay_by(std::int64_t delta_ns, chart::ChartWorkspaceWidget& workspace)
{
    if (dataset_path_.empty()) {
        throw std::runtime_error("no dataset loaded");
    }
    if (!replay_session_->enabled()) {
        throw std::runtime_error("replay mode is not enabled");
    }

    const auto frame = replay_session_->advance_by(delta_ns);
    return apply_replay_windows(frame, workspace);
}

ReplayUpdateResult DataLoadController::advance_replay_by_speed(chart::ChartWorkspaceWidget& workspace)
{
    const auto delta_ns = static_cast<std::int64_t>(replay_session_->speed()) * kNanosecondsPerSecond;
    return advance_replay_by(delta_ns, workspace);
}

data::ScheduleSubmitStatus DataLoadController::submit_window_async(
    data::CandleWindowRequest request,
    chart::ChartWorkspaceWidget& workspace,
    QObject* receiver,
    LoadCallback callback)
{
    const auto chart_id = request.chart_id;
    workspace.setChartLoading(chart_id, true);

    data::ScheduledWindowRequest scheduled_request;
    scheduled_request.dataset_path = dataset_path_;
    scheduled_request.indicator_version = dataset_info_.indicator_version;
    scheduled_request.candle_request = std::move(request);

    auto dataset_info = dataset_info_;
    auto* workspace_ptr = &workspace;
    auto error_callback = error_callback_;
    return scheduler_->submit_window(
        std::move(scheduled_request),
        receiver,
        [dataset_info = std::move(dataset_info),
            workspace_ptr,
            callback = std::move(callback),
            error_callback = std::move(error_callback)](data::ScheduledWindowResult result) mutable {
            workspace_ptr->setChartLoading(result.request.candle_request.chart_id, false);
            if (result.error.has_value()) {
                if (error_callback) {
                    error_callback(std::move(*result.error));
                }
                return;
            }

            result.window.from_cache = result.from_cache;
            LoadResult load_result{dataset_info, std::move(result.window)};
            const auto accepted = workspace_ptr->apply_window(data::CandleWindow{load_result.window});
            if (accepted && callback) {
                callback(std::move(load_result));
            }
        });
}

data::ScheduleSubmitStatus DataLoadController::request_all_enabled_windows_async(
    core::TimeRange visible_range,
    chart::ChartWorkspaceWidget& workspace,
    QObject* receiver,
    LoadCallback callback)
{
    auto first_status = data::ScheduleSubmitStatus::Scheduled;
    auto first_request = true;
    const auto chart_ids = workspace.enabled_chart_ids();
    for (const auto chart_id : chart_ids) {
        auto request = makeWindowRequest(workspace, chart_id, visible_range, requested_period_);
        const auto status = submit_window_async(std::move(request), workspace, receiver, callback);
        if (first_request) {
            first_status = status;
            first_request = false;
        }
    }
    return first_status;
}

void DataLoadController::configure_replay(chart::ChartWorkspaceWidget& workspace, std::int64_t start_time_ns)
{
    replay::ReplayConfig config;
    config.dataset_range = dataset_info_.tick_range;
    config.periods = replayPeriods(workspace, requested_period_);
    config.start_time_ns = start_time_ns;
    config.max_ticks_per_frame = kReplayMaxTicksPerFrame;
    config.max_bars_per_period = kReplayMaxBarsPerPeriod;
    replay_session_->configure(std::move(config));
}

ReplayUpdateResult DataLoadController::apply_replay_windows(
    const replay::ReplayAdvanceResult& frame,
    chart::ChartWorkspaceWidget& workspace)
{
    ReplayUpdateResult result;
    result.current_time_ns = replay_session_->current_time_ns();
    result.ticks_consumed = frame.ticks_consumed;
    result.enabled = replay_session_->enabled();
    result.playing = replay_session_->playing();
    result.reached_end = frame.reached_end;

    for (const auto chart_id : workspace.enabled_chart_ids()) {
        auto period = workspace.requested_period(chart_id);
        if (period.empty()) {
            period = requested_period_;
        }

        auto replay_window = replay_session_->window_for_period(
            period,
            chart_id,
            0,
            replayVisibleRange(*replay_session_, period, dataset_info_));
        if (!replay_window.has_value()) {
            continue;
        }

        replay_window->generation = workspace.chart_view(chart_id).bump_generation();
        if (workspace.apply_window(std::move(*replay_window))) {
            result.applied_window = true;
        }
    }

    return result;
}

} // namespace tradereview::app
