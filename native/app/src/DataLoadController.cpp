#include "tradereview/app/DataLoadController.h"

#include "tradereview/chart/ChartViewWidget.h"
#include "tradereview/chart/ChartWorkspaceWidget.h"
#include "tradereview/data/DuckDbRepository.h"
#include "tradereview/data/IDataStore.h"

#include <algorithm>
#include <cstdint>
#include <memory>
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

} // namespace

DataLoadController::DataLoadController()
    : DataLoadController(std::make_unique<data::DuckDbRepository>())
{
}

DataLoadController::DataLoadController(std::unique_ptr<data::IDataStore> store)
    : store_(std::move(store))
{
    if (!store_) {
        throw std::invalid_argument("DataLoadController requires an IDataStore");
    }
}

DataLoadController::~DataLoadController() = default;

DataLoadController::DataLoadController(DataLoadController&&) noexcept = default;

DataLoadController& DataLoadController::operator=(DataLoadController&&) noexcept = default;

LoadResult DataLoadController::load_file(const QString& path, chart::ChartWorkspaceWidget& workspace)
{
    const auto dataset_info = store_->open_readonly(toPathString(path));
    data::CandleWindowRequest request;
    request.chart_id = 1;
    request.generation = workspace.chart_view().bump_generation();
    request.requested_period = "1min";
    request.visible_range = initialVisibleRange(dataset_info.tick_range);
    request.pixel_width = chartPixelWidth(workspace);

    auto window = store_->query_candles(request);
    workspace.apply_window(data::CandleWindow{window});

    return {dataset_info, std::move(window)};
}

} // namespace tradereview::app
