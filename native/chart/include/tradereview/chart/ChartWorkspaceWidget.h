#pragma once

#include <QWidget>
#include <QVBoxLayout>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "tradereview/chart/ChartWorkspaceState.h"
#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"
#include "tradereview/sync/CrosshairSyncController.h"

class QString;

namespace tradereview::chart {

class ChartPanelWidget;
class ChartToolbarWidget;
class ChartViewWidget;

class ChartWorkspaceWidget final : public QWidget {
public:
    using StatusCallback = std::function<void(const QString&)>;
    using ReloadRequestCallback = std::function<void(std::uint64_t, core::TimeRange)>;

    explicit ChartWorkspaceWidget(QWidget* parent = nullptr);

    void setStatusCallback(StatusCallback callback);
    void setReloadRequestCallback(ReloadRequestCallback callback);
    bool setChartCount(int count);
    bool setLayoutMode(ChartLayoutMode mode);
    bool setRequestedPeriod(std::uint64_t chart_id, std::string period);
    bool setChartLoading(std::uint64_t chart_id, bool loading);
    bool apply_window(data::CandleWindow window);
    bool triggerDrawingAction(const QString& action);
    [[nodiscard]] ChartViewWidget& chart_view();
    [[nodiscard]] const ChartViewWidget& chart_view() const;
    [[nodiscard]] ChartViewWidget& chart_view(std::uint64_t chart_id);
    [[nodiscard]] const ChartViewWidget& chart_view(std::uint64_t chart_id) const;
    [[nodiscard]] ChartPanelWidget* panel(std::uint64_t chart_id);
    [[nodiscard]] const ChartPanelWidget* panel(std::uint64_t chart_id) const;
    [[nodiscard]] std::vector<std::string> requested_indicators() const;
    [[nodiscard]] std::vector<std::string> requested_indicators(std::uint64_t chart_id) const;
    [[nodiscard]] std::string requested_period(std::uint64_t chart_id) const;
    [[nodiscard]] int chart_pixel_width(std::uint64_t chart_id) const;
    [[nodiscard]] std::uint64_t active_chart_id() const;
    [[nodiscard]] std::vector<std::uint64_t> enabled_chart_ids() const;
    [[nodiscard]] std::size_t chart_count() const;
    [[nodiscard]] ChartLayoutMode layout_mode() const;
    bool syncCrosshairFrom(std::uint64_t source_chart_id, std::int64_t timestamp_ns, double price);
    bool syncCenterFrom(
        std::uint64_t source_chart_id,
        std::int64_t timestamp_ns,
        std::optional<double> price = std::nullopt);
    bool syncYCenterFrom(std::uint64_t source_chart_id, double price);

private:
    void rebuild_layout();
    void reset_content_widget();
    void connect_panel(ChartPanelWidget& panel);
    void refresh_sync_enabled_charts();

    ChartWorkspaceState state_;
    QVBoxLayout* root_layout_ = nullptr;
    QWidget* content_ = nullptr;
    std::vector<ChartPanelWidget*> panels_;
    StatusCallback status_callback_;
    ReloadRequestCallback reload_request_callback_;
    sync::CrosshairSyncController crosshair_sync_controller_;
};

} // namespace tradereview::chart
