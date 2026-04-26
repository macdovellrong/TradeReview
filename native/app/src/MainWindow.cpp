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

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
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
    chartWorkspace->chart_view().set_reload_request_callback([showPlaceholderStatus](core::TimeRange range) {
        showPlaceholderStatus(QString("Window reload requested (%1 to %2)")
            .arg(formatTimestamp(range.start_ns))
            .arg(formatTimestamp(range.end_ns)));
    });
    mainControls->setLoadDataCallback([this, chartWorkspace]() {
        const auto path = QFileDialog::getOpenFileName(this, "Load DuckDB Data", QString(), "DuckDB (*.duckdb)");
        if (path.isEmpty()) {
            return;
        }

        try {
            DataLoadController controller;
            const auto result = controller.load_file(path, *chartWorkspace);
            statusBar()->showMessage(loadedMessage(path, result));
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
