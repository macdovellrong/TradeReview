#include "tradereview/app/MainWindow.h"

#include "tradereview/chart/ChartViewWidget.h"

#include <QMenuBar>
#include <QStatusBar>

namespace tradereview::app {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("TradeReview Native");
    resize(1400, 950);

    menuBar()->addMenu("&File");
    setCentralWidget(new chart::ChartViewWidget(this));
    statusBar()->showMessage("Native OpenGL chart ready");
}

} // namespace tradereview::app
