#include "tradereview/chart/FloatingChartWindow.h"

#include "tradereview/chart/ChartPanelWidget.h"

#include <QCloseEvent>
#include <QString>

#include <utility>

namespace tradereview::chart {

FloatingChartWindow::FloatingChartWindow(ChartPanelWidget* panel, QWidget* parent)
    : QMainWindow(parent)
    , panel_(panel)
{
    setWindowFlag(Qt::Window, true);
    setAttribute(Qt::WA_DeleteOnClose, false);
    if (panel_ != nullptr) {
        setCentralWidget(panel_);
    }
    resize(1200, 800);
    refresh_title();
}

std::uint64_t FloatingChartWindow::chart_id() const
{
    return panel_ == nullptr ? 0 : panel_->chart_id();
}

ChartPanelWidget* FloatingChartWindow::take_panel()
{
    auto* widget = takeCentralWidget();
    panel_ = nullptr;
    return dynamic_cast<ChartPanelWidget*>(widget);
}

void FloatingChartWindow::setCloseCallback(CloseCallback callback)
{
    close_callback_ = std::move(callback);
}

void FloatingChartWindow::refresh_title()
{
    if (panel_ == nullptr) {
        setWindowTitle("Chart");
        return;
    }
    setWindowTitle(QString("Chart %1 - %2")
        .arg(static_cast<qulonglong>(panel_->chart_id()))
        .arg(QString::fromStdString(panel_->requested_period())));
}

void FloatingChartWindow::closeEvent(QCloseEvent* event)
{
    const auto id = chart_id();
    if (close_callback_ && id != 0) {
        close_callback_(id);
    }
    event->accept();
}

} // namespace tradereview::chart
