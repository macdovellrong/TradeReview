#pragma once

#include <QWidget>

#include <string>
#include <vector>

#include "tradereview/chart/ChartToolbarWidget.h"
#include "tradereview/data/CandleWindow.h"

namespace tradereview::chart {

class ChartViewWidget;

class ChartWorkspaceWidget final : public QWidget {
public:
    explicit ChartWorkspaceWidget(QWidget* parent = nullptr);

    void setStatusCallback(ChartToolbarWidget::StatusCallback callback);
    bool apply_window(data::CandleWindow window);
    [[nodiscard]] ChartViewWidget& chart_view();
    [[nodiscard]] const ChartViewWidget& chart_view() const;
    [[nodiscard]] std::vector<std::string> requested_indicators() const;

private:
    ChartToolbarWidget* toolbar_;
    ChartViewWidget* chart_view_;
};

} // namespace tradereview::chart
