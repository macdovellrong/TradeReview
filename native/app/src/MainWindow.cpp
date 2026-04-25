#include "tradereview/app/MainWindow.h"

#include <QMenuBar>
#include <QStatusBar>

namespace tradereview::app {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    setWindowTitle("TradeReview Native");
    resize(1400, 950);

    menuBar()->addMenu("&File");
    statusBar()->showMessage("Native workspace ready");
}

} // namespace tradereview::app
