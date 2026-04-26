#pragma once

#include <QWidget>
#include <QVBoxLayout>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "tradereview/chart/ChartWorkspaceState.h"
#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"

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
    bool apply_window(data::CandleWindow window);
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

private:
    void rebuild_layout();
    void reset_content_widget();
    void connect_panel(ChartPanelWidget& panel);

    ChartWorkspaceState state_;
    QVBoxLayout* root_layout_ = nullptr;
    QWidget* content_ = nullptr;
    std::vector<ChartPanelWidget*> panels_;
    StatusCallback status_callback_;
    ReloadRequestCallback reload_request_callback_;
};

} // namespace tradereview::chart
