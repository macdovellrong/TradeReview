#include "tradereview/app/MainWindow.h"

#include "tradereview/app/MainControlsBar.h"
#include "tradereview/chart/ChartWorkspaceWidget.h"

#include <QMenuBar>
#include <QString>
#include <QStatusBar>
#include <QVBoxLayout>
#include <QWidget>

namespace tradereview::app {

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

    centralLayout->addWidget(mainControls);
    centralLayout->addWidget(chartWorkspace, 1);
    setCentralWidget(central);

    statusBar()->showMessage("Native OpenGL chart ready");
}

} // namespace tradereview::app
