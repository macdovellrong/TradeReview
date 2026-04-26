#pragma once

#include <QWidget>

#include "tradereview/chart/ChartToolbarWidget.h"
#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tradereview::chart {

class ChartViewWidget;

class ChartPanelWidget final : public QWidget {
public:
    using ReloadRequestCallback = std::function<void(std::uint64_t, core::TimeRange)>;
    using PeriodChangedCallback = std::function<void(std::uint64_t, const std::string&)>;

    explicit ChartPanelWidget(std::uint64_t chart_id, QWidget* parent = nullptr);

    [[nodiscard]] std::uint64_t chart_id() const;
    [[nodiscard]] std::string requested_period() const;
    void set_requested_period(std::string period);
    void setStatusCallback(ChartToolbarWidget::StatusCallback callback);
    void setReloadRequestCallback(ReloadRequestCallback callback);
    void setPeriodChangedCallback(PeriodChangedCallback callback);
    bool set_loading(bool loading);
    bool apply_window(data::CandleWindow window);
    [[nodiscard]] ChartViewWidget& chart_view();
    [[nodiscard]] const ChartViewWidget& chart_view() const;
    [[nodiscard]] std::vector<std::string> requested_indicators() const;

private:
    void connect_toolbar();
    void set_indicator_enabled(const QString& indicator, bool enabled);
    void select_period(const QString& period);
    void handle_drawing_action(const QString& action);

    std::uint64_t chart_id_ = 0;
    std::string requested_period_ = "1min";
    ChartToolbarWidget* toolbar_ = nullptr;
    ChartViewWidget* chart_view_ = nullptr;
    ReloadRequestCallback reload_request_callback_;
    PeriodChangedCallback period_changed_callback_;
};

} // namespace tradereview::chart
