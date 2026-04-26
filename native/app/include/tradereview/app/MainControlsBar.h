#pragma once

#include <QWidget>

#include <cstdint>
#include <functional>

class QString;
class QComboBox;
class QPushButton;

namespace tradereview::app {

class MainControlsBar final : public QWidget {
public:
    using StatusCallback = std::function<void(const QString&)>;
    using LoadDataCallback = std::function<void()>;
    using LayoutModeCallback = std::function<void(const QString&)>;
    using ChartCountCallback = std::function<void(int)>;
    using ReplayModeCallback = std::function<void(bool)>;
    using ReplayPlayCallback = std::function<void()>;
    using ReplayStepCallback = std::function<void(std::int64_t)>;
    using ReplaySpeedCallback = std::function<void(int)>;

    explicit MainControlsBar(QWidget* parent = nullptr);

    void setStatusCallback(StatusCallback callback);
    void setLoadDataCallback(LoadDataCallback callback);
    void setLayoutModeCallback(LayoutModeCallback callback);
    void setChartCountCallback(ChartCountCallback callback);
    void setReplayModeCallback(ReplayModeCallback callback);
    void setReplayPlayCallback(ReplayPlayCallback callback);
    void setReplayStepCallback(ReplayStepCallback callback);
    void setReplaySpeedCallback(ReplaySpeedCallback callback);
    void setReplayControlsEnabled(bool enabled);
    void setReplayPlaying(bool playing);

private:
    void loadData() const;
    void selectLayoutMode(const QString& mode) const;
    void selectChartCount(const QString& count) const;
    void setReplayMode(bool enabled) const;
    void toggleReplayPlay() const;
    void stepReplay(int direction) const;
    void notify(const QString& action) const;

    StatusCallback status_callback_;
    LoadDataCallback load_data_callback_;
    LayoutModeCallback layout_mode_callback_;
    ChartCountCallback chart_count_callback_;
    ReplayModeCallback replay_mode_callback_;
    ReplayPlayCallback replay_play_callback_;
    ReplayStepCallback replay_step_callback_;
    ReplaySpeedCallback replay_speed_callback_;
    QPushButton* replay_play_button_ = nullptr;
    QPushButton* replay_back_button_ = nullptr;
    QPushButton* replay_forward_button_ = nullptr;
    QComboBox* replay_step_combo_ = nullptr;
};

} // namespace tradereview::app
