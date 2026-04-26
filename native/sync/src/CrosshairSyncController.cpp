#include "tradereview/sync/CrosshairSyncController.h"

#include <algorithm>
#include <utility>

namespace tradereview::sync {
namespace {

class SyncGuard final {
public:
    explicit SyncGuard(bool& syncing)
        : syncing_(syncing)
    {
        syncing_ = true;
    }

    ~SyncGuard()
    {
        syncing_ = false;
    }

private:
    bool& syncing_;
};

} // namespace

bool CrosshairSyncController::register_chart(
    std::uint64_t chart_id,
    DenseXResolver dense_x_resolver,
    CrosshairCallback crosshair_callback,
    CenterTimeCallback center_time_callback,
    YCenterCallback y_center_callback)
{
    if (chart_id == 0) {
        return false;
    }

    auto* existing = find_chart(chart_id);
    if (existing != nullptr) {
        existing->dense_x_resolver = std::move(dense_x_resolver);
        existing->crosshair_callback = std::move(crosshair_callback);
        existing->center_time_callback = std::move(center_time_callback);
        existing->y_center_callback = std::move(y_center_callback);
        return false;
    }

    charts_.push_back(ChartRegistration{
        chart_id,
        true,
        std::move(dense_x_resolver),
        std::move(crosshair_callback),
        std::move(center_time_callback),
        std::move(y_center_callback),
    });
    return true;
}

bool CrosshairSyncController::unregister_chart(std::uint64_t chart_id)
{
    const auto original_size = charts_.size();
    charts_.erase(
        std::remove_if(charts_.begin(), charts_.end(), [chart_id](const ChartRegistration& chart) {
            return chart.chart_id == chart_id;
        }),
        charts_.end());
    return charts_.size() != original_size;
}

bool CrosshairSyncController::set_chart_enabled(std::uint64_t chart_id, bool enabled)
{
    auto* chart = find_chart(chart_id);
    if (chart == nullptr || chart->enabled == enabled) {
        return false;
    }
    chart->enabled = enabled;
    return true;
}

bool CrosshairSyncController::chart_enabled(std::uint64_t chart_id) const
{
    const auto* chart = find_chart(chart_id);
    return chart != nullptr && chart->enabled;
}

bool CrosshairSyncController::is_syncing() const
{
    return syncing_;
}

std::vector<std::uint64_t> CrosshairSyncController::registered_chart_ids() const
{
    std::vector<std::uint64_t> ids;
    ids.reserve(charts_.size());
    for (const auto& chart : charts_) {
        ids.push_back(chart.chart_id);
    }
    return ids;
}

std::vector<std::uint64_t> CrosshairSyncController::enabled_chart_ids() const
{
    std::vector<std::uint64_t> ids;
    ids.reserve(charts_.size());
    for (const auto& chart : charts_) {
        if (chart.enabled) {
            ids.push_back(chart.chart_id);
        }
    }
    return ids;
}

bool CrosshairSyncController::sync_crosshair_from(std::uint64_t source_chart_id, std::int64_t timestamp_ns, double price)
{
    const auto* source = find_chart(source_chart_id);
    if (source == nullptr || !source->enabled || syncing_) {
        return false;
    }

    SyncGuard guard(syncing_);
    auto emitted = false;
    for (const auto& target : charts_) {
        if (!target.enabled || target.chart_id == source_chart_id || !target.dense_x_resolver || !target.crosshair_callback) {
            continue;
        }

        const auto dense_x = target.dense_x_resolver(timestamp_ns);
        if (!dense_x.has_value()) {
            continue;
        }

        target.crosshair_callback(CrosshairUpdate{
            source_chart_id,
            target.chart_id,
            timestamp_ns,
            price,
            *dense_x,
        });
        emitted = true;
    }
    return emitted;
}

bool CrosshairSyncController::sync_center_from(
    std::uint64_t source_chart_id,
    std::int64_t timestamp_ns,
    std::optional<double> price)
{
    const auto* source = find_chart(source_chart_id);
    if (source == nullptr || !source->enabled || syncing_) {
        return false;
    }

    SyncGuard guard(syncing_);
    auto emitted = false;
    for (const auto& target : charts_) {
        if (!target.enabled || !target.dense_x_resolver || !target.center_time_callback) {
            continue;
        }

        const auto dense_x = target.dense_x_resolver(timestamp_ns);
        if (!dense_x.has_value()) {
            continue;
        }

        target.center_time_callback(CenterTimeUpdate{
            source_chart_id,
            target.chart_id,
            timestamp_ns,
            *dense_x,
            price,
        });
        emitted = true;
    }
    return emitted;
}

bool CrosshairSyncController::sync_y_center_from(std::uint64_t source_chart_id, double price)
{
    const auto* source = find_chart(source_chart_id);
    if (source == nullptr || !source->enabled || syncing_) {
        return false;
    }

    SyncGuard guard(syncing_);
    auto emitted = false;
    for (const auto& target : charts_) {
        if (!target.enabled || !target.y_center_callback) {
            continue;
        }

        target.y_center_callback(YCenterUpdate{
            source_chart_id,
            target.chart_id,
            price,
        });
        emitted = true;
    }
    return emitted;
}

CrosshairSyncController::ChartRegistration* CrosshairSyncController::find_chart(std::uint64_t chart_id)
{
    const auto found = std::find_if(charts_.begin(), charts_.end(), [chart_id](const ChartRegistration& chart) {
        return chart.chart_id == chart_id;
    });
    if (found == charts_.end()) {
        return nullptr;
    }
    return &(*found);
}

const CrosshairSyncController::ChartRegistration* CrosshairSyncController::find_chart(std::uint64_t chart_id) const
{
    const auto found = std::find_if(charts_.begin(), charts_.end(), [chart_id](const ChartRegistration& chart) {
        return chart.chart_id == chart_id;
    });
    if (found == charts_.end()) {
        return nullptr;
    }
    return &(*found);
}

} // namespace tradereview::sync
