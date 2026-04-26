#include "tradereview/chart/ChartWorkspaceWidget.h"

#include "tradereview/chart/ChartViewWidget.h"

#include <QVBoxLayout>

#include <utility>

namespace tradereview::chart {

ChartWorkspaceWidget::ChartWorkspaceWidget(QWidget* parent)
    : QWidget(parent)
    , toolbar_(new ChartToolbarWidget(this))
    , chart_view_(new ChartViewWidget(this))
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(toolbar_);
    layout->addWidget(chart_view_, 1);
}

void ChartWorkspaceWidget::setStatusCallback(ChartToolbarWidget::StatusCallback callback)
{
    toolbar_->setStatusCallback(std::move(callback));
}

bool ChartWorkspaceWidget::apply_window(data::CandleWindow window)
{
    return chart_view_->apply_window(std::move(window));
}

ChartViewWidget& ChartWorkspaceWidget::chart_view()
{
    return *chart_view_;
}

const ChartViewWidget& ChartWorkspaceWidget::chart_view() const
{
    return *chart_view_;
}

} // namespace tradereview::chart
