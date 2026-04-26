#include "tradereview/app/MainWindow.h"

#include "tradereview/app/DataLoadController.h"
#include "tradereview/app/MainControlsBar.h"
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

#include <cstdint>
#include <exception>
#include <memory>

namespace tradereview::app {
namespace {

chart::ChartLayoutMode layoutModeFromText(const QString& text)
{
    if (text == "Vertical") {
        return chart::ChartLayoutMode::Vertical;
    }
    if (text == "Dual Vertical") {
        return chart::ChartLayoutMode::DualVertical;
    }
    if (text == "Grid 2x2") {
        return chart::ChartLayoutMode::Grid2x2;
    }
    return chart::ChartLayoutMode::Tabs;
}

QString formatTimestamp(std::int64_t timestamp_ns)
{
    return QDateTime::fromMSecsSinceEpoch(timestamp_ns / 1000000LL, QTimeZone::UTC).toString(Qt::ISODate);
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
    auto showPlaceholderStatus = [this](const QString& message) {
        statusBar()->showMessage(message);
    };
    mainControls->setStatusCallback(showPlaceholderStatus);
    chartWorkspace->setStatusCallback(showPlaceholderStatus);
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
            statusBar()->showMessage(QString("Window reload failed: ") + error.what());
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
    mainControls->setReplayModeCallback([this, chartWorkspace, mainControls](bool enabled) {
        try {
            data_load_controller_->set_replay_enabled(enabled, *chartWorkspace);
            mainControls->setReplayControlsEnabled(enabled);
            mainControls->setReplayPlaying(false);
            statusBar()->showMessage(enabled ? "Replay Mode enabled" : "Replay Mode disabled");
        } catch (const std::exception& error) {
            mainControls->setReplayControlsEnabled(false);
            mainControls->setReplayPlaying(false);
            statusBar()->showMessage(QString("Replay Mode failed: ") + error.what());
        }
    });
    mainControls->setReplayPlayCallback([this, mainControls]() {
        try {
            const auto playing = data_load_controller_->toggle_replay_playing();
            mainControls->setReplayPlaying(playing);
            statusBar()->showMessage(playing ? "Replay playing" : "Replay paused");
        } catch (const std::exception& error) {
            mainControls->setReplayPlaying(false);
            statusBar()->showMessage(QString("Replay play failed: ") + error.what());
        }
    });
    mainControls->setReplayStepCallback([this, chartWorkspace, mainControls](std::int64_t delta_ns) {
        try {
            const auto result = data_load_controller_->advance_replay_by(delta_ns, *chartWorkspace);
            mainControls->setReplayPlaying(result.playing);
            statusBar()->showMessage(replayMessage(result));
        } catch (const std::exception& error) {
            mainControls->setReplayPlaying(false);
            statusBar()->showMessage(QString("Replay step failed: ") + error.what());
        }
    });
    mainControls->setReplaySpeedCallback([this](int speed) {
        data_load_controller_->set_replay_speed(speed);
        statusBar()->showMessage(QString("Replay Speed %1x").arg(speed));
    });
    mainControls->setLoadDataCallback([this, chartWorkspace]() {
        const auto path = QFileDialog::getOpenFileName(this, "Load DuckDB Data", QString(), "DuckDB (*.duckdb)");
        if (path.isEmpty()) {
            return;
        }

        try {
            const auto status = data_load_controller_->load_file_async(
                path,
                *chartWorkspace,
                this,
                [this, path](LoadResult result) {
                    statusBar()->showMessage(loadedMessage(path, result));
                });
            statusBar()->showMessage(pendingMessage(status));
        } catch (const std::exception& error) {
            statusBar()->showMessage(QString("Load Data failed: ") + error.what());
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
            if (result.reached_end) {
                statusBar()->showMessage(replayMessage(result));
            }
        } catch (const std::exception& error) {
            mainControls->setReplayPlaying(false);
            statusBar()->showMessage(QString("Replay timer failed: ") + error.what());
        }
    });
    replayTimer->start(100);

    statusBar()->showMessage("Native OpenGL chart ready");
}

} // namespace tradereview::app
