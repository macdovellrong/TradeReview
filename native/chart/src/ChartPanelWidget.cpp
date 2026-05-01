#include "tradereview/chart/ChartPanelWidget.h"

#include "tradereview/chart/ChartPeriod.h"
#include "tradereview/chart/ChartViewWidget.h"
#include "tradereview/drawing/DrawingSpec.h"

#include <QString>
#include <QVBoxLayout>

#include <cstddef>
#include <utility>

namespace tradereview::chart {
namespace {

std::string to_string(const QString& text)
{
    const auto utf8 = text.toUtf8();
    return {utf8.constData(), static_cast<std::size_t>(utf8.size())};
}

std::string canonical_period(const QString& period)
{
    return canonical_chart_period(to_string(period));
}

QString toolbar_period(const std::string& period)
{
    return QString::fromStdString(toolbar_chart_period(period));
}

} // namespace

ChartPanelWidget::ChartPanelWidget(std::uint64_t chart_id, QWidget* parent)
    : QWidget(parent)
    , chart_id_(chart_id)
    , toolbar_(new ChartToolbarWidget(this))
    , chart_view_(new ChartViewWidget(this))
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    layout->addWidget(toolbar_);
    layout->addWidget(chart_view_, 1);
    connect_toolbar();
}

std::uint64_t ChartPanelWidget::chart_id() const
{
    return chart_id_;
}

std::string ChartPanelWidget::requested_period() const
{
    return requested_period_;
}

void ChartPanelWidget::set_requested_period(std::string period)
{
    if (period.empty()) {
        return;
    }
    requested_period_ = canonical_chart_period(period);
    toolbar_->setSelectedPeriod(toolbar_period(requested_period_));
}

void ChartPanelWidget::setStatusCallback(ChartToolbarWidget::StatusCallback callback)
{
    toolbar_->setStatusCallback(std::move(callback));
}

void ChartPanelWidget::setReloadRequestCallback(ReloadRequestCallback callback)
{
    reload_request_callback_ = std::move(callback);
    chart_view_->set_reload_request_callback([this](core::TimeRange range) {
        if (reload_request_callback_) {
            reload_request_callback_(chart_id_, range);
        }
    });
}

void ChartPanelWidget::setPeriodChangedCallback(PeriodChangedCallback callback)
{
    period_changed_callback_ = std::move(callback);
}

void ChartPanelWidget::setPopoutCallback(PopoutCallback callback)
{
    popout_callback_ = std::move(callback);
}

bool ChartPanelWidget::set_loading(bool loading)
{
    return chart_view_->set_loading(loading);
}

bool ChartPanelWidget::apply_window(data::CandleWindow window)
{
    if (window.chart_id != chart_id_) {
        return false;
    }
    return chart_view_->apply_window(std::move(window));
}

void ChartPanelWidget::trigger_drawing_action(const QString& action)
{
    handle_drawing_action(action);
}

ChartViewWidget& ChartPanelWidget::chart_view()
{
    return *chart_view_;
}

const ChartViewWidget& ChartPanelWidget::chart_view() const
{
    return *chart_view_;
}

std::vector<std::string> ChartPanelWidget::requested_indicators() const
{
    return chart_view_->requested_indicators();
}

void ChartPanelWidget::connect_toolbar()
{
    toolbar_->setIndicatorToggleCallback([this](const QString& indicator, bool enabled) {
        set_indicator_enabled(indicator, enabled);
    });
    toolbar_->setPeriodSelectedCallback([this](const QString& period) {
        select_period(period);
    });
    toolbar_->setPriceAxisFitCallback([this]() {
        chart_view_->fit_price_axis_to_visible_range();
    });
    toolbar_->setPopoutCallback([this]() {
        if (popout_callback_) {
            popout_callback_(chart_id_);
        }
    });
    toolbar_->setSelectedPeriod(toolbar_period(requested_period_));
}

void ChartPanelWidget::set_indicator_enabled(const QString& indicator, bool enabled)
{
    if (indicator == "BB") {
        chart_view_->set_bollinger_bands_enabled(enabled);
        return;
    }
    if (indicator == "MACD/RSI") {
        chart_view_->set_indicator_panels_enabled(enabled);
        return;
    }

    chart_view_->set_indicator_enabled(to_string(indicator), enabled);
}

void ChartPanelWidget::select_period(const QString& period)
{
    const auto selected_period = canonical_period(period);
    if (selected_period.empty() || selected_period == requested_period_) {
        return;
    }

    requested_period_ = selected_period;
    if (period_changed_callback_) {
        period_changed_callback_(chart_id_, requested_period_);
    }
    chart_view_->request_current_visible_window();
}

void ChartPanelWidget::handle_drawing_action(const QString& action)
{
    using drawing::DrawingType;

    if (action == "Sel") {
        chart_view_->clear_active_drawing_tool();
        return;
    }
    if (action == "H") {
        chart_view_->set_active_drawing_tool(DrawingType::HorizontalLine);
        return;
    }
    if (action == "V") {
        chart_view_->set_active_drawing_tool(DrawingType::VerticalLine);
        return;
    }
    if (action == "Line") {
        chart_view_->set_active_drawing_tool(DrawingType::Line);
        return;
    }
    if (action == "Fib") {
        chart_view_->set_active_drawing_tool(DrawingType::FibRetracement);
        return;
    }
    if (action == "Fib Ext") {
        chart_view_->set_active_drawing_tool(DrawingType::FibExtension);
        return;
    }
    if (action == "Clear") {
        chart_view_->clear_drawings();
    }
}

} // namespace tradereview::chart
