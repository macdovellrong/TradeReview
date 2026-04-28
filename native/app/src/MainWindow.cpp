#include "tradereview/app/MainWindow.h"

#include "tradereview/app/DataLoadController.h"
#include "tradereview/app/ErrorPresenter.h"
#include "tradereview/app/MainControlsBar.h"
#include "tradereview/app/SessionState.h"
#include "tradereview/app/SideInfoPanelWidget.h"
#include "tradereview/app/StatusStripWidget.h"
#include "tradereview/app/TimeNavigation.h"
#include "tradereview/chart/ChartViewWidget.h"
#include "tradereview/chart/ChartWorkspaceWidget.h"

#include <QDateTime>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QString>
#include <QStatusBar>
#include <QTimeZone>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>

namespace tradereview::app {
namespace {

chart::ChartLayoutMode layoutModeFromText(const QString& text)
{
    const auto utf8 = text.toUtf8();
    const auto parsed = layout_mode_from_string({utf8.constData(), static_cast<std::size_t>(utf8.size())});
    if (parsed.has_value()) {
        return *parsed;
    }
    return chart::ChartLayoutMode::Tabs;
}

QString layoutModeText(chart::ChartLayoutMode mode)
{
    const auto text = layout_mode_to_string(mode);
    return QString::fromUtf8(text.data(), static_cast<qsizetype>(text.size()));
}

QString qstringFromStdString(const std::string& text)
{
    return QString::fromUtf8(text.data(), static_cast<qsizetype>(text.size()));
}

QString formatTimestamp(std::int64_t timestamp_ns)
{
    return QDateTime::fromMSecsSinceEpoch(timestamp_ns / 1000000LL, QTimeZone::UTC).toString(Qt::ISODate);
}

std::int64_t midpoint(core::TimeRange range)
{
    return range.start_ns + ((range.end_ns - range.start_ns) / 2);
}

void syncWorkspaceToTarget(
    chart::ChartWorkspaceWidget& workspace,
    const data::CandleWindow& window,
    std::int64_t target_ns)
{
    if (const auto target = resolve_chart_target_row(window, target_ns); target.has_value()) {
        workspace.syncCenterFrom(window.chart_id, target_ns, target->close);
    }
}

QString loadedMessage(const QString& path, const LoadResult& result)
{
    return QString("Loaded %1 rows from %2 (%3 to %4)")
        .arg(static_cast<qulonglong>(result.window.row_count()))
        .arg(QFileInfo(path).fileName())
        .arg(formatTimestamp(result.window.visible_range.start_ns))
        .arg(formatTimestamp(result.window.visible_range.end_ns));
}

QString windowMessage(const LoadResult& result)
{
    return QString("Loaded %1 rows (%2 to %3)")
        .arg(static_cast<qulonglong>(result.window.row_count()))
        .arg(formatTimestamp(result.window.visible_range.start_ns))
        .arg(formatTimestamp(result.window.visible_range.end_ns));
}

QString pendingMessage(data::ScheduleSubmitStatus status)
{
    switch (status) {
    case data::ScheduleSubmitStatus::Scheduled:
        return "Loading window...";
    case data::ScheduleSubmitStatus::Coalesced:
        return "Window load already pending";
    case data::ScheduleSubmitStatus::CacheHit:
        return "Loading cached window...";
    }
    return "Loading window...";
}

QString replayMessage(const ReplayUpdateResult& result)
{
    if (result.reached_end) {
        return QString("Replay reached end at %1").arg(formatTimestamp(result.current_time_ns));
    }
    return QString("Replay %1, %2 tick(s)")
        .arg(formatTimestamp(result.current_time_ns))
        .arg(static_cast<qulonglong>(result.ticks_consumed));
}

QString replaySummary(bool enabled, bool playing, int speed)
{
    if (!enabled) {
        return "Disabled";
    }
    return QString("Enabled / %1 / %2x").arg(playing ? "Playing" : "Paused").arg(speed);
}

QString replaySummary(const ReplayUpdateResult& result, int speed)
{
    if (result.reached_end) {
        return QString("Enabled / Reached end / %1x").arg(speed);
    }
    return replaySummary(result.enabled, result.playing, speed);
}

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , settings_("TradeReview", "TradeReview")
    , data_load_controller_(std::make_unique<DataLoadController>())
{
    setWindowTitle("TradeReview Native");
    resize(1400, 950);

    menuBar()->addMenu("&File");

    auto* central = new QWidget(this);
    auto* root = new QVBoxLayout(central);
    root->setContentsMargins(0, 0, 0, 0);
    root->setSpacing(0);

    main_controls_ = new MainControlsBar(central);
    chart_workspace_ = new chart::ChartWorkspaceWidget(central);
    side_info_panel_ = new SideInfoPanelWidget(central);
    status_strip_ = new StatusStripWidget(central);

    auto* workspace = new QWidget(central);
    auto* workspace_layout = new QHBoxLayout(workspace);
    workspace_layout->setContentsMargins(0, 0, 0, 0);
    workspace_layout->setSpacing(0);
    workspace_layout->addWidget(chart_workspace_, 1);
    workspace_layout->addWidget(side_info_panel_);

    root->addWidget(main_controls_);
    root->addWidget(workspace, 1);
    root->addWidget(status_strip_);
    setCentralWidget(central);

    auto showStatus = [this](const QString& message) {
        showStatusMessage(message);
    };
    main_controls_->setStatusCallback(showStatus);
    chart_workspace_->setStatusCallback(showStatus);
    data_load_controller_->set_error_callback([this](data::DataError error) {
        present_error(this, statusBar(), "Window load", error, false);
    });
    chart_workspace_->setReloadRequestCallback([this](std::uint64_t chart_id, core::TimeRange range) {
        try {
            const auto status = data_load_controller_->request_window_async(
                chart_id,
                range,
                *chart_workspace_,
                this,
                [this](LoadResult result) {
                    side_info_panel_->setVisibleRange(
                        formatTimestamp(result.window.visible_range.start_ns) + " to "
                        + formatTimestamp(result.window.visible_range.end_ns));
                    setReadyStatus(windowMessage(result));
                });
            setLoadingStatus(pendingMessage(status));
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Window reload", error, false);
        }
    });
    main_controls_->setLayoutModeCallback([this](const QString& mode) {
        chart_workspace_->setLayoutMode(layoutModeFromText(mode));
        side_info_panel_->setLayoutSummary(QString("%1 charts / %2")
                                               .arg(static_cast<int>(chart_workspace_->chart_count()))
                                               .arg(layoutModeText(chart_workspace_->layout_mode())));
        showStatusMessage(QString("Layout ") + mode);
    });
    main_controls_->setChartCountCallback([this](int count) {
        chart_workspace_->setChartCount(count);
        side_info_panel_->setLayoutSummary(QString("%1 charts / %2")
                                               .arg(static_cast<int>(chart_workspace_->chart_count()))
                                               .arg(layoutModeText(chart_workspace_->layout_mode())));
        showStatusMessage(QString("Charts %1").arg(count));
    });
    main_controls_->setResetViewCallback([this]() {
        try {
            const auto status = data_load_controller_->reset_view_async(
                *chart_workspace_,
                this,
                [this](LoadResult result) {
                    main_controls_->setDateTimeValue(midpoint(result.window.visible_range));
                    side_info_panel_->setVisibleRange(
                        formatTimestamp(result.window.visible_range.start_ns) + " to "
                        + formatTimestamp(result.window.visible_range.end_ns));
                    setReadyStatus(windowMessage(result));
                });
            const auto center = data_load_controller_->current_view_center_time_ns(*chart_workspace_);
            main_controls_->setDateTimeValue(center);
            setLoadingStatus(pendingMessage(status));
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Reset View", error);
        }
    });
    main_controls_->setSaveViewCallback([this]() {
        try {
            if (!data_load_controller_->dataset_loaded()) {
                showStatusMessage("Load a dataset before saving the view");
                return;
            }

            SessionState state;
            state.dataset_path = data_load_controller_->dataset_path();
            state.center_time_ns = data_load_controller_->current_view_center_time_ns(*chart_workspace_);
            state.chart_count = static_cast<int>(chart_workspace_->chart_count());
            state.layout_mode = chart_workspace_->layout_mode();
            for (std::uint64_t chart_id = 1; chart_id <= 4; ++chart_id) {
                state.periods.push_back(chart_workspace_->requested_period(chart_id));
            }
            save_session_state(settings_, state);
            showStatusMessage("View saved");
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Save View", error);
        }
    });
    main_controls_->setDateTimeJumpCallback([this](std::int64_t timestamp_ns) {
        try {
            const auto target = clamp_jump_timestamp_ns(
                normalize_jump_timestamp_ns(timestamp_ns),
                data_load_controller_->dataset_info().tick_range);
            const auto status = data_load_controller_->jump_to_time_async(
                target,
                *chart_workspace_,
                this,
                [this, target](LoadResult result) {
                    syncWorkspaceToTarget(*chart_workspace_, result.window, target);
                    main_controls_->setDateTimeValue(target);
                    side_info_panel_->setVisibleRange(
                        formatTimestamp(result.window.visible_range.start_ns) + " to "
                        + formatTimestamp(result.window.visible_range.end_ns));
                    setReadyStatus(windowMessage(result));
                });
            main_controls_->setDateTimeValue(target);
            setLoadingStatus(pendingMessage(status));
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Date jump", error);
        }
    });
    main_controls_->setReplayModeCallback([this](bool enabled) {
        try {
            data_load_controller_->set_replay_enabled(enabled, *chart_workspace_);
            main_controls_->setReplayControlsEnabled(enabled);
            main_controls_->setReplayPlaying(false);
            side_info_panel_->setReplaySummary(replaySummary(enabled, false, data_load_controller_->replay_speed()));
            showStatusMessage(enabled ? "Replay Mode enabled" : "Replay Mode disabled");
        } catch (const std::exception& error) {
            main_controls_->setReplayControlsEnabled(false);
            main_controls_->setReplayPlaying(false);
            side_info_panel_->setReplaySummary("Disabled");
            present_error(this, statusBar(), "Replay Mode", error, false);
        }
    });
    main_controls_->setReplayPlayCallback([this]() {
        try {
            const auto playing = data_load_controller_->toggle_replay_playing();
            main_controls_->setReplayPlaying(playing);
            side_info_panel_->setReplaySummary(
                replaySummary(data_load_controller_->replay_enabled(), playing, data_load_controller_->replay_speed()));
            showStatusMessage(playing ? "Replay playing" : "Replay paused");
        } catch (const std::exception& error) {
            main_controls_->setReplayPlaying(false);
            side_info_panel_->setReplaySummary(
                replaySummary(data_load_controller_->replay_enabled(), false, data_load_controller_->replay_speed()));
            present_error(this, statusBar(), "Replay play", error, false);
        }
    });
    main_controls_->setReplayStepCallback([this](std::int64_t delta_ns) {
        try {
            if (data_load_controller_->replay_enabled()) {
                const auto result = data_load_controller_->advance_replay_by(delta_ns, *chart_workspace_);
                main_controls_->setReplayPlaying(result.playing);
                main_controls_->setDateTimeValue(result.current_time_ns);
                side_info_panel_->setReplaySummary(replaySummary(result, data_load_controller_->replay_speed()));
                showStatusMessage(replayMessage(result));
                return;
            }

            side_info_panel_->setReplaySummary("Manual step");
            const auto status = data_load_controller_->step_time_async(
                delta_ns,
                *chart_workspace_,
                this,
                [this](LoadResult result) {
                    const auto target = midpoint(result.window.visible_range);
                    syncWorkspaceToTarget(*chart_workspace_, result.window, target);
                    main_controls_->setDateTimeValue(target);
                    side_info_panel_->setVisibleRange(
                        formatTimestamp(result.window.visible_range.start_ns) + " to "
                        + formatTimestamp(result.window.visible_range.end_ns));
                    side_info_panel_->setReplaySummary("Disabled");
                    setReadyStatus(windowMessage(result));
                });
            main_controls_->setDateTimeValue(data_load_controller_->current_view_center_time_ns(*chart_workspace_));
            setLoadingStatus(pendingMessage(status));
        } catch (const std::exception& error) {
            main_controls_->setReplayPlaying(false);
            side_info_panel_->setReplaySummary(
                replaySummary(data_load_controller_->replay_enabled(), false, data_load_controller_->replay_speed()));
            present_error(this, statusBar(), "Step", error, false);
        }
    });
    main_controls_->setReplaySpeedCallback([this](int speed) {
        data_load_controller_->set_replay_speed(speed);
        side_info_panel_->setReplaySummary(
            replaySummary(data_load_controller_->replay_enabled(), data_load_controller_->replay_playing(), speed));
        showStatusMessage(QString("Replay Speed %1x").arg(speed));
    });
    main_controls_->setLoadDataCallback([this]() {
        const auto path = QFileDialog::getOpenFileName(this, "Load DuckDB Data", QString(), "DuckDB (*.duckdb)");
        if (path.isEmpty()) {
            return;
        }

        try {
            const auto status = data_load_controller_->load_file_async(
                path,
                *chart_workspace_,
                this,
                [this, path](LoadResult result) {
                    main_controls_->setDateTimeRange(
                        result.dataset_info.tick_range.start_ns,
                        result.dataset_info.tick_range.end_ns);
                    main_controls_->setDateTimeValue(midpoint(result.window.visible_range));
                    side_info_panel_->setDatasetName(QFileInfo(path).fileName());
                    side_info_panel_->setDataRange(
                        formatTimestamp(result.dataset_info.tick_range.start_ns) + " to "
                        + formatTimestamp(result.dataset_info.tick_range.end_ns));
                    side_info_panel_->setVisibleRange(
                        formatTimestamp(result.window.visible_range.start_ns) + " to "
                        + formatTimestamp(result.window.visible_range.end_ns));
                    status_strip_->setDataRangeText(
                        QString("Range: ") + formatTimestamp(result.dataset_info.tick_range.start_ns) + " to "
                        + formatTimestamp(result.dataset_info.tick_range.end_ns));
                    setReadyStatus(loadedMessage(path, result));
                });
            const auto& info = data_load_controller_->dataset_info();
            main_controls_->setDateTimeRange(info.tick_range.start_ns, info.tick_range.end_ns);
            main_controls_->setDateTimeValue(midpoint(info.tick_range));
            setLoadingStatus(pendingMessage(status));
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Load Data", error);
        }
    });

    auto* replayTimer = new QTimer(this);
    connect(replayTimer, &QTimer::timeout, this, [this]() {
        if (!data_load_controller_->replay_playing()) {
            return;
        }
        try {
            const auto result = data_load_controller_->advance_replay_by_speed(*chart_workspace_);
            main_controls_->setReplayPlaying(result.playing);
            main_controls_->setDateTimeValue(result.current_time_ns);
            side_info_panel_->setReplaySummary(replaySummary(result, data_load_controller_->replay_speed()));
            if (result.reached_end) {
                showStatusMessage(replayMessage(result));
            }
        } catch (const std::exception& error) {
            main_controls_->setReplayPlaying(false);
            side_info_panel_->setReplaySummary(
                replaySummary(data_load_controller_->replay_enabled(), false, data_load_controller_->replay_speed()));
            present_error(this, statusBar(), "Replay timer", error, false);
        }
    });
    replayTimer->start(100);

    QTimer::singleShot(0, this, [this]() {
        const auto state = load_session_state(settings_);
        if (!state.has_value()) {
            return;
        }
        if (!QFileInfo::exists(qstringFromStdString(state->dataset_path))) {
            return;
        }

        chart_workspace_->setChartCount(state->chart_count);
        chart_workspace_->setLayoutMode(state->layout_mode);
        main_controls_->setChartCountValue(state->chart_count);
        main_controls_->setLayoutModeText(layoutModeText(state->layout_mode));
        side_info_panel_->setLayoutSummary(QString("%1 charts / %2")
                                               .arg(static_cast<int>(chart_workspace_->chart_count()))
                                               .arg(layoutModeText(chart_workspace_->layout_mode())));
        for (std::size_t index = 0; index < state->periods.size() && index < 4; ++index) {
            chart_workspace_->setRequestedPeriod(static_cast<std::uint64_t>(index + 1), state->periods[index]);
        }

        const auto path = qstringFromStdString(state->dataset_path);
        try {
            const auto status = data_load_controller_->load_file_async(
                path,
                *chart_workspace_,
                this,
                [this, path](LoadResult result) {
                    main_controls_->setDateTimeRange(
                        result.dataset_info.tick_range.start_ns,
                        result.dataset_info.tick_range.end_ns);
                    main_controls_->setDateTimeValue(midpoint(result.window.visible_range));
                    side_info_panel_->setDatasetName(QFileInfo(path).fileName());
                    side_info_panel_->setDataRange(
                        formatTimestamp(result.dataset_info.tick_range.start_ns) + " to "
                        + formatTimestamp(result.dataset_info.tick_range.end_ns));
                    side_info_panel_->setVisibleRange(
                        formatTimestamp(result.window.visible_range.start_ns) + " to "
                        + formatTimestamp(result.window.visible_range.end_ns));
                    status_strip_->setDataRangeText(
                        QString("Range: ") + formatTimestamp(result.dataset_info.tick_range.start_ns) + " to "
                        + formatTimestamp(result.dataset_info.tick_range.end_ns));
                    setReadyStatus(loadedMessage(path, result));
                });
            const auto& info = data_load_controller_->dataset_info();
            const auto target = clamp_jump_timestamp_ns(state->center_time_ns, info.tick_range);
            main_controls_->setDateTimeRange(info.tick_range.start_ns, info.tick_range.end_ns);
            main_controls_->setDateTimeValue(target);
            data_load_controller_->jump_to_time_async(
                target,
                *chart_workspace_,
                this,
                [this, target](LoadResult result) {
                    syncWorkspaceToTarget(*chart_workspace_, result.window, target);
                    main_controls_->setDateTimeValue(target);
                    side_info_panel_->setVisibleRange(
                        formatTimestamp(result.window.visible_range.start_ns) + " to "
                        + formatTimestamp(result.window.visible_range.end_ns));
                    setReadyStatus(windowMessage(result));
                });
            setLoadingStatus(pendingMessage(status));
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Restore View", error, false);
        }
    });

    showStatusMessage("Native OpenGL chart ready");
}

void MainWindow::showStatusMessage(const QString& message)
{
    statusBar()->showMessage(message);
    if (status_strip_ != nullptr) {
        status_strip_->setMessageText(message);
    }
    if (side_info_panel_ != nullptr) {
        side_info_panel_->setLastMessage(message);
    }
}

void MainWindow::setLoadingStatus(const QString& message)
{
    if (status_strip_ != nullptr) {
        status_strip_->setStateText("Loading");
    }
    showStatusMessage(message);
}

void MainWindow::setReadyStatus(const QString& message)
{
    if (status_strip_ != nullptr) {
        status_strip_->setStateText("Ready");
    }
    showStatusMessage(message);
}

} // namespace tradereview::app
