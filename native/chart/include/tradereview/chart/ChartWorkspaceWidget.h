#pragma once

#include <QWidget>

#include "tradereview/chart/ChartToolbarWidget.h"

namespace tradereview::chart {

class ChartViewWidget;

class ChartWorkspaceWidget final : public QWidget {
public:
    explicit ChartWorkspaceWidget(QWidget* parent = nullptr);

    void setStatusCallback(ChartToolbarWidget::StatusCallback callback);
    [[nodiscard]] ChartViewWidget& chart_view();
    [[nodiscard]] const ChartViewWidget& chart_view() const;

private:
    ChartToolbarWidget* toolbar_;
    ChartViewWidget* chart_view_;
};

} // namespace tradereview::chart
