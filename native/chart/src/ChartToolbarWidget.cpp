#include "tradereview/chart/ChartToolbarWidget.h"

#include <QButtonGroup>
#include <QFrame>
#include <QHBoxLayout>
#include <QPushButton>
#include <QScrollArea>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <utility>

namespace tradereview::chart {
namespace {

constexpr int kButtonHeight = 30;

QPushButton* createToolbarButton(QWidget* parent, const QString& text, int width)
{
    auto* button = new QPushButton(text, parent);
    button->setFixedSize(width, kButtonHeight);
    return button;
}

bool isDefaultSelectedEma(const QString& label)
{
    return label == "EMA20" || label == "EMA30" || label == "EMA40" || label == "EMA50" || label == "EMA60";
}

} // namespace

ChartToolbarWidget::ChartToolbarWidget(QWidget* parent)
    : QWidget(parent)
{
    setObjectName("ChartToolbarWidget");

    auto* toolbarLayout = new QHBoxLayout(this);
    toolbarLayout->setContentsMargins(4, 3, 4, 3);
    toolbarLayout->setSpacing(4);

    auto* periodWidget = new QWidget(this);
    periodWidget->setObjectName("ToolbarGroup");
    periodWidget->setAttribute(Qt::WA_StyledBackground, true);
    auto* periodGroupLayout = new QHBoxLayout(periodWidget);
    periodGroupLayout->setContentsMargins(0, 0, 8, 0);
    periodGroupLayout->setSpacing(4);

    auto* periodScrollArea = new QScrollArea(this);
    periodScrollArea->setObjectName("PeriodScrollArea");
    periodScrollArea->setAttribute(Qt::WA_StyledBackground, true);
    periodScrollArea->viewport()->setObjectName("PeriodScrollViewport");
    periodScrollArea->viewport()->setAttribute(Qt::WA_StyledBackground, true);
    periodScrollArea->setFixedHeight(kButtonHeight + 6);
    periodScrollArea->setFrameShape(QFrame::NoFrame);
    periodScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    periodScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    periodScrollArea->setWidgetResizable(true);

    auto* periodContent = new QWidget(periodScrollArea);
    periodContent->setObjectName("PeriodScrollContent");
    periodContent->setAttribute(Qt::WA_StyledBackground, true);
    auto* periodLayout = new QHBoxLayout(periodContent);
    periodLayout->setContentsMargins(0, 0, 0, 0);
    periodLayout->setSpacing(2);

    auto* periodGroup = new QButtonGroup(this);
    periodGroup->setExclusive(true);

    const QStringList periods{
        "30s", "1m", "2m", "3m", "5m", "10m", "15m", "20m", "30m", "45m", "90m",
        "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1D", "1W", "1M"};
    for (const auto& period : periods) {
        auto* button = createToolbarButton(periodContent, period, 40);
        button->setCheckable(true);
        button->setChecked(period == "1m");
        periodGroup->addButton(button);
        period_buttons_.push_back(button);
        connect(button, &QPushButton::clicked, this, [this, period](bool) {
            notify_period_selected(period);
        });
        periodLayout->addWidget(button);
    }
    periodLayout->addStretch();
    periodScrollArea->setWidget(periodContent);
    periodGroupLayout->addWidget(periodScrollArea);
    toolbarLayout->addWidget(periodWidget, 1);

    auto* indicator_widget = new QWidget(this);
    indicator_widget->setObjectName("ToolbarGroup");
    indicator_widget->setAttribute(Qt::WA_StyledBackground, true);
    auto* indicator_layout = new QHBoxLayout(indicator_widget);
    indicator_layout->setContentsMargins(8, 0, 8, 0);
    indicator_layout->setSpacing(4);
    toolbarLayout->addWidget(indicator_widget);

    auto* indicatorGroup = new QButtonGroup(this);
    indicatorGroup->setExclusive(false);
    const QStringList emaIndicators{"EMA20", "EMA30", "EMA40", "EMA50", "EMA60", "EMA100", "EMA240"};
    for (const auto& label : emaIndicators) {
        auto* button = createToolbarButton(indicator_widget, label, 68);
        button->setCheckable(true);
        button->setChecked(isDefaultSelectedEma(label));
        indicatorGroup->addButton(button);
        connect(button, &QPushButton::toggled, this, [this, label](bool checked) {
            notify_indicator_toggle(label, checked);
        });
        indicator_layout->addWidget(button);
    }

    for (const auto& label : QStringList{"BB", "MACD/RSI"}) {
        auto* button = createToolbarButton(indicator_widget, label, label == "BB" ? 44 : 86);
        button->setCheckable(true);
        button->setChecked(true);
        indicatorGroup->addButton(button);
        connect(button, &QPushButton::toggled, this, [this, label](bool checked) {
            notify_indicator_toggle(label, checked);
        });
        indicator_layout->addWidget(button);
    }

}

void ChartToolbarWidget::setStatusCallback(StatusCallback callback)
{
    status_callback_ = std::move(callback);
}

void ChartToolbarWidget::setIndicatorToggleCallback(IndicatorToggleCallback callback)
{
    indicator_toggle_callback_ = std::move(callback);
}

void ChartToolbarWidget::setPeriodSelectedCallback(PeriodSelectedCallback callback)
{
    period_selected_callback_ = std::move(callback);
}

void ChartToolbarWidget::setSelectedPeriod(const QString& period)
{
    for (auto* button : period_buttons_) {
        if (button != nullptr) {
            button->setChecked(button->text() == period);
        }
    }
}

void ChartToolbarWidget::notify_indicator_toggle(const QString& indicator, bool enabled) const
{
    if (indicator_toggle_callback_) {
        indicator_toggle_callback_(indicator, enabled);
    }
}

void ChartToolbarWidget::notify_period_selected(const QString& period) const
{
    if (period_selected_callback_) {
        period_selected_callback_(period);
    }
}

} // namespace tradereview::chart
