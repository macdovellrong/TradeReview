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
#include <QVBoxLayout>
#include <QWidget>

#include <cstdint>
#include <exception>
#include <memory>

namespace tradereview::app {
namespace {

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
    chartWorkspace->chart_view().set_reload_request_callback([this, chartWorkspace](core::TimeRange range) {
        try {
            const auto status = data_load_controller_->request_window_async(
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

    statusBar()->showMessage("Native OpenGL chart ready");
}

} // namespace tradereview::app
