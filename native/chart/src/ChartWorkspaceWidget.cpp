#include "tradereview/chart/ChartWorkspaceWidget.h"

#include "tradereview/chart/ChartViewWidget.h"

#include <QString>
#include <QVBoxLayout>

#include <string>
#include <utility>
#include <vector>

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

    toolbar_->setIndicatorToggleCallback([this](const QString& indicator, bool enabled) {
        if (indicator == "BB") {
            chart_view_->set_bollinger_bands_enabled(enabled);
            return;
        }
        if (indicator == "MACD/RSI") {
            chart_view_->set_indicator_panels_enabled(enabled);
            return;
        }

        const auto utf8 = indicator.toUtf8();
        chart_view_->set_indicator_enabled(
            std::string{utf8.constData(), static_cast<std::size_t>(utf8.size())},
            enabled);
    });
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

std::vector<std::string> ChartWorkspaceWidget::requested_indicators() const
{
    return chart_view_->requested_indicators();
}

} // namespace tradereview::chart
