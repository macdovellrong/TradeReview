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

#include <cstdint>
#include <utility>

namespace tradereview::app {
namespace {

constexpr int kControlHeight = 28;
constexpr std::int64_t kNanosecondsPerSecond = 1000LL * 1000LL * 1000LL;

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

std::int64_t stepNanoseconds(const QString& text)
{
    if (text == "30s") {
        return 30LL * kNanosecondsPerSecond;
    }
    if (text == "1m") {
        return 60LL * kNanosecondsPerSecond;
    }
    if (text == "5m") {
        return 5LL * 60LL * kNanosecondsPerSecond;
    }
    if (text == "15m") {
        return 15LL * 60LL * kNanosecondsPerSecond;
    }
    if (text == "30m") {
        return 30LL * 60LL * kNanosecondsPerSecond;
    }
    if (text == "2h") {
        return 2LL * 60LL * 60LL * kNanosecondsPerSecond;
    }
    if (text == "4h") {
        return 4LL * 60LL * 60LL * kNanosecondsPerSecond;
    }
    if (text == "1D") {
        return 24LL * 60LL * 60LL * kNanosecondsPerSecond;
    }
    return 60LL * 60LL * kNanosecondsPerSecond;
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
        selectLayoutMode(text);
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
        selectChartCount(text);
    });
    layout->addWidget(chartsCombo);

    auto* replayMode = new QCheckBox("Replay Mode", this);
    connect(replayMode, &QCheckBox::toggled, this, [this](bool checked) {
        setReplayMode(checked);
    });
    layout->addWidget(replayMode);

    const QStringList replayActions{"Play", "Back", "Forward"};
    for (const auto& action : replayActions) {
        auto* button = addButton(layout, this, action);
        if (action == "Play") {
            replay_play_button_ = button;
        } else if (action == "Back") {
            replay_back_button_ = button;
        } else if (action == "Forward") {
            replay_forward_button_ = button;
        }
        connect(button, &QPushButton::clicked, this, [this, action](bool) {
            if (action == "Play") {
                toggleReplayPlay();
            } else if (action == "Back") {
                stepReplay(-1);
            } else {
                stepReplay(1);
            }
        });
    }

    replay_step_combo_ = new QComboBox(this);
    replay_step_combo_->addItems({"30s", "1m", "5m", "15m", "30m", "1h", "2h", "4h", "1D"});
    replay_step_combo_->setCurrentText("1h");
    replay_step_combo_->setMinimumHeight(kControlHeight);
    connect(replay_step_combo_, &QComboBox::currentTextChanged, this, [this](const QString& text) {
        notify(QString("Step ") + text);
    });
    layout->addWidget(replay_step_combo_);

    layout->addWidget(new QLabel("Speed:", this));
    auto* speedGroup = new QButtonGroup(this);
    speedGroup->setExclusive(true);
    for (const auto speed : {1, 10, 60, 120, 300, 600}) {
        const auto label = QString::number(speed) + "x";
        auto* button = addButton(layout, this, label, 44);
        button->setCheckable(true);
        button->setChecked(speed == 60);
        speedGroup->addButton(button);
        connect(button, &QPushButton::clicked, this, [this, label, speed](bool) {
            if (replay_speed_callback_) {
                replay_speed_callback_(speed);
                return;
            }
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
    setReplayControlsEnabled(false);
}

void MainControlsBar::setStatusCallback(StatusCallback callback)
{
    status_callback_ = std::move(callback);
}

void MainControlsBar::setLoadDataCallback(LoadDataCallback callback)
{
    load_data_callback_ = std::move(callback);
}

void MainControlsBar::setLayoutModeCallback(LayoutModeCallback callback)
{
    layout_mode_callback_ = std::move(callback);
}

void MainControlsBar::setChartCountCallback(ChartCountCallback callback)
{
    chart_count_callback_ = std::move(callback);
}

void MainControlsBar::setReplayModeCallback(ReplayModeCallback callback)
{
    replay_mode_callback_ = std::move(callback);
}

void MainControlsBar::setReplayPlayCallback(ReplayPlayCallback callback)
{
    replay_play_callback_ = std::move(callback);
}

void MainControlsBar::setReplayStepCallback(ReplayStepCallback callback)
{
    replay_step_callback_ = std::move(callback);
}

void MainControlsBar::setReplaySpeedCallback(ReplaySpeedCallback callback)
{
    replay_speed_callback_ = std::move(callback);
}

void MainControlsBar::setReplayControlsEnabled(bool enabled)
{
    if (replay_play_button_ != nullptr) {
        replay_play_button_->setEnabled(enabled);
    }
    if (replay_back_button_ != nullptr) {
        replay_back_button_->setEnabled(enabled);
    }
    if (replay_forward_button_ != nullptr) {
        replay_forward_button_->setEnabled(enabled);
    }
}

void MainControlsBar::setReplayPlaying(bool playing)
{
    if (replay_play_button_ != nullptr) {
        replay_play_button_->setText(playing ? "Pause" : "Play");
    }
}

void MainControlsBar::loadData() const
{
    if (load_data_callback_) {
        load_data_callback_();
    }
}

void MainControlsBar::selectLayoutMode(const QString& mode) const
{
    if (layout_mode_callback_) {
        layout_mode_callback_(mode);
    }
}

void MainControlsBar::selectChartCount(const QString& count) const
{
    if (chart_count_callback_) {
        chart_count_callback_(count.toInt());
    }
}

void MainControlsBar::setReplayMode(bool enabled) const
{
    if (replay_mode_callback_) {
        replay_mode_callback_(enabled);
        return;
    }
    notify("Replay Mode");
}

void MainControlsBar::toggleReplayPlay() const
{
    if (replay_play_callback_) {
        replay_play_callback_();
        return;
    }
    notify("Play");
}

void MainControlsBar::stepReplay(int direction) const
{
    if (replay_step_callback_) {
        const auto step = stepNanoseconds(replay_step_combo_ != nullptr ? replay_step_combo_->currentText() : QString{"1h"});
        replay_step_callback_(direction < 0 ? -step : step);
        return;
    }
    notify(direction < 0 ? "Back" : "Forward");
}

void MainControlsBar::notify(const QString& action) const
{
    if (status_callback_) {
        status_callback_(placeholderMessage(action));
    }
}

} // namespace tradereview::app
