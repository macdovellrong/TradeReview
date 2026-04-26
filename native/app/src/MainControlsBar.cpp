#include "tradereview/app/MainControlsBar.h"

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDateTimeEdit>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QString>
#include <QStringList>

#include <utility>

namespace tradereview::app {
namespace {

constexpr int kControlHeight = 28;

QString placeholderMessage(const QString& action)
{
    return action + " is not wired yet";
}

QPushButton* addButton(QHBoxLayout* layout, QWidget* parent, const QString& text, int width = 0)
{
    auto* button = new QPushButton(text, parent);
    button->setMinimumHeight(kControlHeight);
    if (width > 0) {
        button->setFixedWidth(width);
    }
    layout->addWidget(button);
    return button;
}

} // namespace

MainControlsBar::MainControlsBar(QWidget* parent)
    : QWidget(parent)
{
    setObjectName("MainControlsBar");
    setStyleSheet(R"(
        #MainControlsBar {
            background-color: #181818;
        }
        #MainControlsBar QLabel,
        #MainControlsBar QCheckBox {
            color: #cfcfcf;
        }
        #MainControlsBar QPushButton,
        #MainControlsBar QComboBox,
        #MainControlsBar QDateTimeEdit {
            background-color: #242424;
            border: 1px solid #444444;
            border-radius: 2px;
            color: #d8d8d8;
            padding: 3px 8px;
        }
        #MainControlsBar QPushButton:hover,
        #MainControlsBar QComboBox:hover,
        #MainControlsBar QDateTimeEdit:hover {
            background-color: #303030;
            color: #ffffff;
        }
        #MainControlsBar QPushButton:checked {
            background-color: #007acc;
            border-color: #007acc;
            color: #ffffff;
        }
    )");

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    const QStringList primaryActions{"Load Data", "Reset View", "Save View"};
    for (const auto& action : primaryActions) {
        auto* button = addButton(layout, this, action);
        if (action == "Load Data") {
            connect(button, &QPushButton::clicked, this, [this](bool) {
                loadData();
            });
        } else {
            connect(button, &QPushButton::clicked, this, [this, action](bool) {
                notify(action);
            });
        }
    }

    layout->addWidget(new QLabel("Layout:", this));
    auto* layoutCombo = new QComboBox(this);
    layoutCombo->addItems({"Tabs", "Dual Vertical", "Grid 2x2", "Vertical"});
    layoutCombo->setMinimumHeight(kControlHeight);
    connect(layoutCombo, &QComboBox::currentTextChanged, this, [this](const QString& text) {
        notify(QString("Layout ") + text);
    });
    layout->addWidget(layoutCombo);

    auto* popLayoutButton = addButton(layout, this, "Pop Layout");
    connect(popLayoutButton, &QPushButton::clicked, this, [this](bool) {
        notify("Pop Layout");
    });

    layout->addWidget(new QLabel("Charts:", this));
    auto* chartsCombo = new QComboBox(this);
    chartsCombo->addItems({"1", "2", "3", "4"});
    chartsCombo->setCurrentText("4");
    chartsCombo->setMinimumHeight(kControlHeight);
    connect(chartsCombo, &QComboBox::currentTextChanged, this, [this](const QString& text) {
        notify(QString("Charts ") + text);
    });
    layout->addWidget(chartsCombo);

    auto* replayMode = new QCheckBox("Replay Mode", this);
    connect(replayMode, &QCheckBox::toggled, this, [this](bool) {
        notify("Replay Mode");
    });
    layout->addWidget(replayMode);

    const QStringList replayActions{"Play", "Back", "Forward"};
    for (const auto& action : replayActions) {
        auto* button = addButton(layout, this, action);
        connect(button, &QPushButton::clicked, this, [this, action](bool) {
            notify(action);
        });
    }

    auto* stepCombo = new QComboBox(this);
    stepCombo->addItems({"30s", "1m", "5m", "15m", "30m", "1h", "2h", "4h", "1D"});
    stepCombo->setCurrentText("1h");
    stepCombo->setMinimumHeight(kControlHeight);
    connect(stepCombo, &QComboBox::currentTextChanged, this, [this](const QString& text) {
        notify(QString("Step ") + text);
    });
    layout->addWidget(stepCombo);

    layout->addWidget(new QLabel("Speed:", this));
    auto* speedGroup = new QButtonGroup(this);
    speedGroup->setExclusive(true);
    for (const auto speed : {1, 10, 60, 120, 300, 600}) {
        const auto label = QString::number(speed) + "x";
        auto* button = addButton(layout, this, label, 44);
        button->setCheckable(true);
        button->setChecked(speed == 60);
        speedGroup->addButton(button);
        connect(button, &QPushButton::clicked, this, [this, label](bool) {
            notify(QString("Speed ") + label);
        });
    }

    auto* dateTimeEdit = new QDateTimeEdit(this);
    dateTimeEdit->setDisplayFormat("yyyy-MM-dd HH:mm");
    dateTimeEdit->setCalendarPopup(true);
    dateTimeEdit->setKeyboardTracking(false);
    dateTimeEdit->setDateTime(QDateTime::currentDateTime());
    dateTimeEdit->setMinimumHeight(kControlHeight);
    connect(dateTimeEdit, &QDateTimeEdit::editingFinished, this, [this]() {
        notify("Date Time");
    });
    layout->addWidget(dateTimeEdit);

    layout->addStretch();
}

void MainControlsBar::setStatusCallback(StatusCallback callback)
{
    status_callback_ = std::move(callback);
}

void MainControlsBar::setLoadDataCallback(LoadDataCallback callback)
{
    load_data_callback_ = std::move(callback);
}

void MainControlsBar::loadData() const
{
    if (load_data_callback_) {
        load_data_callback_();
    }
}

void MainControlsBar::notify(const QString& action) const
{
    if (status_callback_) {
        status_callback_(placeholderMessage(action));
    }
}

} // namespace tradereview::app
