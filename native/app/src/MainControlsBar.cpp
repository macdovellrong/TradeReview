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
#include <QSignalBlocker>
#include <QStyle>
#include <QTimeZone>

#include <cstdint>
#include <utility>

namespace tradereview::app {
namespace {

constexpr int kControlHeight = 28;
constexpr std::int64_t kNanosecondsPerSecond = 1000LL * 1000LL * 1000LL;
constexpr std::int64_t kNanosecondsPerMillisecond = 1000LL * 1000LL;

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

QHBoxLayout* addGroup(QHBoxLayout* root, QWidget* parent)
{
    auto* group_widget = new QWidget(parent);
    group_widget->setObjectName("ToolbarGroup");
    auto* group_layout = new QHBoxLayout(group_widget);
    group_layout->setContentsMargins(0, 0, 10, 0);
    group_layout->setSpacing(5);
    root->addWidget(group_widget);
    return group_layout;
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

QDateTime dateTimeFromNs(std::int64_t timestamp_ns)
{
    return QDateTime::fromMSecsSinceEpoch(timestamp_ns / kNanosecondsPerMillisecond, QTimeZone::UTC);
}

std::int64_t dateTimeToNs(const QDateTime& date_time)
{
    return date_time.toUTC().toMSecsSinceEpoch() * kNanosecondsPerMillisecond;
}

} // namespace

MainControlsBar::MainControlsBar(QWidget* parent)
    : QWidget(parent)
{
    setObjectName("MainControlsBar");

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    auto* data_group = addGroup(layout, this);
    const QStringList primaryActions{"Load Data", "Reset View", "Save View"};
    for (const auto& action : primaryActions) {
        auto* button = addButton(data_group, this, action);
        if (action == "Load Data") {
            button->setProperty("primary", true);
            button->style()->unpolish(button);
            button->style()->polish(button);
            connect(button, &QPushButton::clicked, this, [this](bool) {
                loadData();
            });
        } else if (action == "Reset View") {
            connect(button, &QPushButton::clicked, this, [this](bool) {
                resetView();
            });
        } else if (action == "Save View") {
            connect(button, &QPushButton::clicked, this, [this](bool) {
                saveView();
            });
        } else {
            connect(button, &QPushButton::clicked, this, [this, action](bool) {
                notify(action);
            });
        }
    }

    auto* layout_group = addGroup(layout, this);
    layout_group->addWidget(new QLabel("Layout:", this));
    layout_combo_ = new QComboBox(this);
    layout_combo_->addItems({"Tabs", "Dual Vertical", "Grid 2x2", "Vertical"});
    layout_combo_->setMinimumHeight(kControlHeight);
    connect(layout_combo_, &QComboBox::currentTextChanged, this, [this](const QString& text) {
        selectLayoutMode(text);
    });
    layout_group->addWidget(layout_combo_);

    auto* popLayoutButton = addButton(layout_group, this, "Pop Layout");
    connect(popLayoutButton, &QPushButton::clicked, this, [this](bool) {
        notify("Pop Layout");
    });

    layout_group->addWidget(new QLabel("Charts:", this));
    charts_combo_ = new QComboBox(this);
    charts_combo_->addItems({"1", "2", "3", "4"});
    charts_combo_->setCurrentText("4");
    charts_combo_->setMinimumHeight(kControlHeight);
    connect(charts_combo_, &QComboBox::currentTextChanged, this, [this](const QString& text) {
        selectChartCount(text);
    });
    layout_group->addWidget(charts_combo_);

    auto* replay_group = addGroup(layout, this);
    auto* replayMode = new QCheckBox("Replay Mode", this);
    connect(replayMode, &QCheckBox::toggled, this, [this](bool checked) {
        setReplayMode(checked);
    });
    replay_group->addWidget(replayMode);

    const QStringList replayActions{"Play", "Back", "Forward"};
    for (const auto& action : replayActions) {
        auto* button = addButton(replay_group, this, action);
        if (action == "Play") {
            replay_play_button_ = button;
            button->setProperty("primary", true);
            button->style()->unpolish(button);
            button->style()->polish(button);
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
    replay_group->addWidget(replay_step_combo_);

    auto* speed_layout = addGroup(layout, this);
    speed_layout->addWidget(new QLabel("Speed:", this));
    auto* speedGroup = new QButtonGroup(this);
    speedGroup->setExclusive(true);
    for (const auto speed : {1, 10, 60, 120, 300, 600}) {
        const auto label = QString::number(speed) + "x";
        auto* button = addButton(speed_layout, this, label, 44);
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

    auto* time_group = addGroup(layout, this);
    date_time_edit_ = new QDateTimeEdit(this);
    date_time_edit_->setDisplayFormat("yyyy-MM-dd HH:mm");
    date_time_edit_->setCalendarPopup(true);
    date_time_edit_->setKeyboardTracking(false);
    date_time_edit_->setTimeZone(QTimeZone::UTC);
    date_time_edit_->setDateTime(QDateTime::currentDateTimeUtc());
    date_time_edit_->setMinimumHeight(kControlHeight);
    connect(date_time_edit_, &QDateTimeEdit::editingFinished, this, [this]() {
        jumpToDateTime();
    });
    time_group->addWidget(date_time_edit_);

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

void MainControlsBar::setResetViewCallback(ResetViewCallback callback)
{
    reset_view_callback_ = std::move(callback);
}

void MainControlsBar::setSaveViewCallback(SaveViewCallback callback)
{
    save_view_callback_ = std::move(callback);
}

void MainControlsBar::setLayoutModeCallback(LayoutModeCallback callback)
{
    layout_mode_callback_ = std::move(callback);
}

void MainControlsBar::setChartCountCallback(ChartCountCallback callback)
{
    chart_count_callback_ = std::move(callback);
}

void MainControlsBar::setDateTimeJumpCallback(DateTimeJumpCallback callback)
{
    date_time_jump_callback_ = std::move(callback);
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
        replay_play_button_->setProperty("primary", true);
        replay_play_button_->setText(playing ? "Pause" : "Play");
        replay_play_button_->style()->unpolish(replay_play_button_);
        replay_play_button_->style()->polish(replay_play_button_);
    }
}

void MainControlsBar::setDateTimeRange(std::int64_t start_ns, std::int64_t end_ns)
{
    if (date_time_edit_ == nullptr) {
        return;
    }
    if (end_ns < start_ns) {
        end_ns = start_ns;
    }
    QSignalBlocker blocker(date_time_edit_);
    date_time_edit_->setMinimumDateTime(dateTimeFromNs(start_ns));
    date_time_edit_->setMaximumDateTime(dateTimeFromNs(end_ns));
}

void MainControlsBar::setDateTimeValue(std::int64_t timestamp_ns)
{
    if (date_time_edit_ == nullptr) {
        return;
    }
    QSignalBlocker blocker(date_time_edit_);
    date_time_edit_->setDateTime(dateTimeFromNs(timestamp_ns));
}

void MainControlsBar::setLayoutModeText(const QString& text)
{
    if (layout_combo_ == nullptr) {
        return;
    }
    QSignalBlocker blocker(layout_combo_);
    layout_combo_->setCurrentText(text);
}

void MainControlsBar::setChartCountValue(int count)
{
    if (charts_combo_ == nullptr) {
        return;
    }
    QSignalBlocker blocker(charts_combo_);
    charts_combo_->setCurrentText(QString::number(count));
}

void MainControlsBar::loadData() const
{
    if (load_data_callback_) {
        load_data_callback_();
    }
}

void MainControlsBar::resetView() const
{
    if (reset_view_callback_) {
        reset_view_callback_();
        return;
    }
    notify("Reset View");
}

void MainControlsBar::saveView() const
{
    if (save_view_callback_) {
        save_view_callback_();
        return;
    }
    notify("Save View");
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

void MainControlsBar::jumpToDateTime() const
{
    if (date_time_jump_callback_ && date_time_edit_ != nullptr) {
        date_time_jump_callback_(dateTimeToNs(date_time_edit_->dateTime()));
        return;
    }
    notify("Date Time");
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
