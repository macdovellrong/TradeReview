#pragma once

#include <QWidget>

#include <cstdint>
#include <functional>

class QString;
class QComboBox;
class QDateTimeEdit;
class QPushButton;

namespace tradereview::app {

class MainControlsBar final : public QWidget {
public:
    using StatusCallback = std::function<void(const QString&)>;
    using LoadDataCallback = std::function<void()>;
    using ResetViewCallback = std::function<void()>;
    using SaveViewCallback = std::function<void()>;
    using LayoutModeCallback = std::function<void(const QString&)>;
    using ChartCountCallback = std::function<void(int)>;
    using DateTimeJumpCallback = std::function<void(std::int64_t)>;
    using ReplayModeCallback = std::function<void(bool)>;
    using ReplayPlayCallback = std::function<void()>;
    using ReplayStepCallback = std::function<void(std::int64_t)>;
    using ReplaySpeedCallback = std::function<void(int)>;

    explicit MainControlsBar(QWidget* parent = nullptr);

    void setStatusCallback(StatusCallback callback);
    void setLoadDataCallback(LoadDataCallback callback);
    void setResetViewCallback(ResetViewCallback callback);
    void setSaveViewCallback(SaveViewCallback callback);
    void setLayoutModeCallback(LayoutModeCallback callback);
    void setChartCountCallback(ChartCountCallback callback);
    void setDateTimeJumpCallback(DateTimeJumpCallback callback);
    void setReplayModeCallback(ReplayModeCallback callback);
    void setReplayPlayCallback(ReplayPlayCallback callback);
    void setReplayStepCallback(ReplayStepCallback callback);
    void setReplaySpeedCallback(ReplaySpeedCallback callback);
    void setReplayControlsEnabled(bool enabled);
    void setReplayPlaying(bool playing);
    void setDateTimeRange(std::int64_t start_ns, std::int64_t end_ns);
    void setDateTimeValue(std::int64_t timestamp_ns);
    void setLayoutModeText(const QString& text);
    void setChartCountValue(int count);

private:
    void loadData() const;
    void resetView() const;
    void saveView() const;
    void selectLayoutMode(const QString& mode) const;
    void selectChartCount(const QString& count) const;
    void jumpToDateTime() const;
    void setReplayMode(bool enabled) const;
    void toggleReplayPlay() const;
    void stepReplay(int direction) const;
    void notify(const QString& action) const;

    StatusCallback status_callback_;
    LoadDataCallback load_data_callback_;
    ResetViewCallback reset_view_callback_;
    SaveViewCallback save_view_callback_;
    LayoutModeCallback layout_mode_callback_;
    ChartCountCallback chart_count_callback_;
    DateTimeJumpCallback date_time_jump_callback_;
    ReplayModeCallback replay_mode_callback_;
    ReplayPlayCallback replay_play_callback_;
    ReplayStepCallback replay_step_callback_;
    ReplaySpeedCallback replay_speed_callback_;
    QPushButton* replay_play_button_ = nullptr;
    QPushButton* replay_back_button_ = nullptr;
    QPushButton* replay_forward_button_ = nullptr;
    QComboBox* layout_combo_ = nullptr;
    QComboBox* charts_combo_ = nullptr;
    QComboBox* replay_step_combo_ = nullptr;
    QDateTimeEdit* date_time_edit_ = nullptr;
};

} // namespace tradereview::app
