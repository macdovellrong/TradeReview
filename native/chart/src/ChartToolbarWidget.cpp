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

QString placeholderMessage(const QString& action)
{
    return action + " is not wired yet";
}

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
    setStyleSheet(R"(
        #ChartToolbarWidget {
            background-color: #181818;
        }
        #ChartToolbarWidget QPushButton {
            background-color: #222222;
            border: 1px solid #444444;
            border-radius: 2px;
            color: #aaaaaa;
            padding: 2px 6px;
        }
        #ChartToolbarWidget QPushButton:hover {
            background-color: #333333;
            color: #ffffff;
        }
        #ChartToolbarWidget QPushButton:checked {
            background-color: #007acc;
            border-color: #007acc;
            color: #ffffff;
        }
        #ChartToolbarWidget QScrollArea {
            background-color: transparent;
            border: none;
        }
    )");

    auto* toolbarLayout = new QHBoxLayout(this);
    toolbarLayout->setContentsMargins(4, 3, 4, 3);
    toolbarLayout->setSpacing(4);

    auto* periodScrollArea = new QScrollArea(this);
    periodScrollArea->setFixedHeight(kButtonHeight + 6);
    periodScrollArea->setFrameShape(QFrame::NoFrame);
    periodScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    periodScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    periodScrollArea->setWidgetResizable(true);

    auto* periodContent = new QWidget(periodScrollArea);
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
        periodGroup->addButton(button);
        connect(button, &QPushButton::clicked, this, [this, period](bool) {
            notify(QString("Period ") + period);
        });
        periodLayout->addWidget(button);
    }
    periodLayout->addStretch();
    periodScrollArea->setWidget(periodContent);
    toolbarLayout->addWidget(periodScrollArea, 1);

    auto* indicatorGroup = new QButtonGroup(this);
    indicatorGroup->setExclusive(false);
    const QStringList emaIndicators{"EMA20", "EMA30", "EMA40", "EMA50", "EMA60", "EMA100", "EMA240"};
    for (const auto& label : emaIndicators) {
        auto* button = createToolbarButton(this, label, 68);
        button->setCheckable(true);
        button->setChecked(isDefaultSelectedEma(label));
        indicatorGroup->addButton(button);
        connect(button, &QPushButton::toggled, this, [this, label](bool) {
            notify(label);
        });
        toolbarLayout->addWidget(button);
    }

    for (const auto& label : QStringList{"BB", "MACD/RSI"}) {
        auto* button = createToolbarButton(this, label, label == "BB" ? 44 : 86);
        button->setCheckable(true);
        button->setChecked(true);
        indicatorGroup->addButton(button);
        connect(button, &QPushButton::toggled, this, [this, label](bool) {
            notify(label);
        });
        toolbarLayout->addWidget(button);
    }

    const QStringList drawingActions{"Sel", "H", "V", "Line", "Fib", "Fib Ext", "Fib Config", "Clear", "Pop"};
    for (const auto& action : drawingActions) {
        int width = 44;
        if (action == "Fib Ext" || action == "Fib Config") {
            width = 76;
        }
        auto* button = createToolbarButton(this, action, width);
        connect(button, &QPushButton::clicked, this, [this, action](bool) {
            notify(action);
        });
        toolbarLayout->addWidget(button);
    }
}

void ChartToolbarWidget::setStatusCallback(StatusCallback callback)
{
    status_callback_ = std::move(callback);
}

void ChartToolbarWidget::notify(const QString& action) const
{
    if (status_callback_) {
        status_callback_(placeholderMessage(action));
    }
}

} // namespace tradereview::chart
