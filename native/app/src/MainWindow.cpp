#include "tradereview/app/MainWindow.h"

#include "tradereview/app/DataLoadController.h"
#include "tradereview/app/ErrorPresenter.h"
#include "tradereview/app/MainControlsBar.h"
#include "tradereview/app/SessionState.h"
#include "tradereview/app/TimeNavigation.h"
#include "tradereview/chart/ChartViewWidget.h"
#include "tradereview/chart/ChartWorkspaceWidget.h"

#include <QDateTime>
#include <QFileDialog>
#include <QFileInfo>
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
    auto* centralLayout = new QVBoxLayout(central);
    centralLayout->setContentsMargins(0, 0, 0, 0);
    centralLayout->setSpacing(0);

    auto* mainControls = new MainControlsBar(central);
    auto* chartWorkspace = new chart::ChartWorkspaceWidget(central);
    auto showStatus = [this](const QString& message) {
        statusBar()->showMessage(message);
    };
    mainControls->setStatusCallback(showStatus);
    chartWorkspace->setStatusCallback(showStatus);
    data_load_controller_->set_error_callback([this](data::DataError error) {
        present_error(this, statusBar(), "Window load", error, false);
    });
    chartWorkspace->setReloadRequestCallback([this, chartWorkspace](std::uint64_t chart_id, core::TimeRange range) {
        try {
            const auto status = data_load_controller_->request_window_async(
                chart_id,
                range,
                *chartWorkspace,
                this,
                [this](LoadResult result) {
                    statusBar()->showMessage(windowMessage(result));
                });
            statusBar()->showMessage(pendingMessage(status));
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Window reload", error, false);
        }
    });
    mainControls->setLayoutModeCallback([this, chartWorkspace](const QString& mode) {
        chartWorkspace->setLayoutMode(layoutModeFromText(mode));
        statusBar()->showMessage(QString("Layout ") + mode);
    });
    mainControls->setChartCountCallback([this, chartWorkspace](int count) {
        chartWorkspace->setChartCount(count);
        statusBar()->showMessage(QString("Charts %1").arg(count));
    });
    mainControls->setResetViewCallback([this, chartWorkspace, mainControls]() {
        try {
            const auto status = data_load_controller_->reset_view_async(
                *chartWorkspace,
                this,
                [this, mainControls](LoadResult result) {
                    mainControls->setDateTimeValue(midpoint(result.window.visible_range));
                    statusBar()->showMessage(windowMessage(result));
                });
            const auto center = data_load_controller_->current_view_center_time_ns(*chartWorkspace);
            mainControls->setDateTimeValue(center);
            statusBar()->showMessage(pendingMessage(status));
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Reset View", error);
        }
    });
    mainControls->setSaveViewCallback([this, chartWorkspace]() {
        try {
            if (!data_load_controller_->dataset_loaded()) {
                statusBar()->showMessage("Load a dataset before saving the view");
                return;
            }

            SessionState state;
            state.dataset_path = data_load_controller_->dataset_path();
            state.center_time_ns = data_load_controller_->current_view_center_time_ns(*chartWorkspace);
            state.chart_count = static_cast<int>(chartWorkspace->chart_count());
            state.layout_mode = chartWorkspace->layout_mode();
            for (std::uint64_t chart_id = 1; chart_id <= 4; ++chart_id) {
                state.periods.push_back(chartWorkspace->requested_period(chart_id));
            }
            save_session_state(settings_, state);
            statusBar()->showMessage("View saved");
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Save View", error);
        }
    });
    mainControls->setDateTimeJumpCallback([this, chartWorkspace, mainControls](std::int64_t timestamp_ns) {
        try {
            const auto target = clamp_jump_timestamp_ns(
                normalize_jump_timestamp_ns(timestamp_ns),
                data_load_controller_->dataset_info().tick_range);
            const auto status = data_load_controller_->jump_to_time_async(
                target,
                *chartWorkspace,
                this,
                [this, chartWorkspace, mainControls, target](LoadResult result) {
                    syncWorkspaceToTarget(*chartWorkspace, result.window, target);
                    mainControls->setDateTimeValue(target);
                    statusBar()->showMessage(windowMessage(result));
                });
            mainControls->setDateTimeValue(target);
            statusBar()->showMessage(pendingMessage(status));
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Date jump", error);
        }
    });
    mainControls->setReplayModeCallback([this, chartWorkspace, mainControls](bool enabled) {
        try {
            data_load_controller_->set_replay_enabled(enabled, *chartWorkspace);
            mainControls->setReplayControlsEnabled(enabled);
            mainControls->setReplayPlaying(false);
            statusBar()->showMessage(enabled ? "Replay Mode enabled" : "Replay Mode disabled");
        } catch (const std::exception& error) {
            mainControls->setReplayControlsEnabled(false);
            mainControls->setReplayPlaying(false);
            present_error(this, statusBar(), "Replay Mode", error, false);
        }
    });
    mainControls->setReplayPlayCallback([this, mainControls]() {
        try {
            const auto playing = data_load_controller_->toggle_replay_playing();
            mainControls->setReplayPlaying(playing);
            statusBar()->showMessage(playing ? "Replay playing" : "Replay paused");
        } catch (const std::exception& error) {
            mainControls->setReplayPlaying(false);
            present_error(this, statusBar(), "Replay play", error, false);
        }
    });
    mainControls->setReplayStepCallback([this, chartWorkspace, mainControls](std::int64_t delta_ns) {
        try {
            if (data_load_controller_->replay_enabled()) {
                const auto result = data_load_controller_->advance_replay_by(delta_ns, *chartWorkspace);
                mainControls->setReplayPlaying(result.playing);
                mainControls->setDateTimeValue(result.current_time_ns);
                statusBar()->showMessage(replayMessage(result));
                return;
            }

            const auto status = data_load_controller_->step_time_async(
                delta_ns,
                *chartWorkspace,
                this,
                [this, chartWorkspace, mainControls](LoadResult result) {
                    const auto target = midpoint(result.window.visible_range);
                    syncWorkspaceToTarget(*chartWorkspace, result.window, target);
                    mainControls->setDateTimeValue(target);
                    statusBar()->showMessage(windowMessage(result));
                });
            mainControls->setDateTimeValue(data_load_controller_->current_view_center_time_ns(*chartWorkspace));
            statusBar()->showMessage(pendingMessage(status));
        } catch (const std::exception& error) {
            mainControls->setReplayPlaying(false);
            present_error(this, statusBar(), "Step", error, false);
        }
    });
    mainControls->setReplaySpeedCallback([this](int speed) {
        data_load_controller_->set_replay_speed(speed);
        statusBar()->showMessage(QString("Replay Speed %1x").arg(speed));
    });
    mainControls->setLoadDataCallback([this, chartWorkspace, mainControls]() {
        const auto path = QFileDialog::getOpenFileName(this, "Load DuckDB Data", QString(), "DuckDB (*.duckdb)");
        if (path.isEmpty()) {
            return;
        }

        try {
            const auto status = data_load_controller_->load_file_async(
                path,
                *chartWorkspace,
                this,
                [this, path, mainControls](LoadResult result) {
                    mainControls->setDateTimeRange(result.dataset_info.tick_range.start_ns, result.dataset_info.tick_range.end_ns);
                    mainControls->setDateTimeValue(midpoint(result.window.visible_range));
                    statusBar()->showMessage(loadedMessage(path, result));
                });
            const auto& info = data_load_controller_->dataset_info();
            mainControls->setDateTimeRange(info.tick_range.start_ns, info.tick_range.end_ns);
            mainControls->setDateTimeValue(midpoint(info.tick_range));
            statusBar()->showMessage(pendingMessage(status));
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Load Data", error);
        }
    });

    centralLayout->addWidget(mainControls);
    centralLayout->addWidget(chartWorkspace, 1);
    setCentralWidget(central);

    auto* replayTimer = new QTimer(this);
    connect(replayTimer, &QTimer::timeout, this, [this, chartWorkspace, mainControls]() {
        if (!data_load_controller_->replay_playing()) {
            return;
        }
        try {
            const auto result = data_load_controller_->advance_replay_by_speed(*chartWorkspace);
            mainControls->setReplayPlaying(result.playing);
            mainControls->setDateTimeValue(result.current_time_ns);
            if (result.reached_end) {
                statusBar()->showMessage(replayMessage(result));
            }
        } catch (const std::exception& error) {
            mainControls->setReplayPlaying(false);
            present_error(this, statusBar(), "Replay timer", error, false);
        }
    });
    replayTimer->start(100);

    QTimer::singleShot(0, this, [this, chartWorkspace, mainControls]() {
        const auto state = load_session_state(settings_);
        if (!state.has_value()) {
            return;
        }
        if (!QFileInfo::exists(qstringFromStdString(state->dataset_path))) {
            return;
        }

        chartWorkspace->setChartCount(state->chart_count);
        chartWorkspace->setLayoutMode(state->layout_mode);
        mainControls->setChartCountValue(state->chart_count);
        mainControls->setLayoutModeText(layoutModeText(state->layout_mode));
        for (std::size_t index = 0; index < state->periods.size() && index < 4; ++index) {
            chartWorkspace->setRequestedPeriod(static_cast<std::uint64_t>(index + 1), state->periods[index]);
        }

        const auto path = qstringFromStdString(state->dataset_path);
        try {
            const auto status = data_load_controller_->load_file_async(
                path,
                *chartWorkspace,
                this,
                [this, path, mainControls](LoadResult result) {
                    mainControls->setDateTimeRange(result.dataset_info.tick_range.start_ns, result.dataset_info.tick_range.end_ns);
                    mainControls->setDateTimeValue(midpoint(result.window.visible_range));
                    statusBar()->showMessage(loadedMessage(path, result));
                });
            const auto& info = data_load_controller_->dataset_info();
            const auto target = clamp_jump_timestamp_ns(state->center_time_ns, info.tick_range);
            mainControls->setDateTimeRange(info.tick_range.start_ns, info.tick_range.end_ns);
            mainControls->setDateTimeValue(target);
            data_load_controller_->jump_to_time_async(
                target,
                *chartWorkspace,
                this,
                [this, chartWorkspace, mainControls, target](LoadResult result) {
                    syncWorkspaceToTarget(*chartWorkspace, result.window, target);
                    mainControls->setDateTimeValue(target);
                    statusBar()->showMessage(windowMessage(result));
                });
            statusBar()->showMessage(pendingMessage(status));
        } catch (const std::exception& error) {
            present_error(this, statusBar(), "Restore View", error, false);
        }
    });

    statusBar()->showMessage("Native OpenGL chart ready");
}

} // namespace tradereview::app
