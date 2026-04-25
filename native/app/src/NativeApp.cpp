#include "tradereview/app/NativeApp.h"

#include "tradereview/app/MainWindow.h"

#include <QApplication>

namespace tradereview::app {

int run_native_app(int argc, char** argv)
{
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    return QApplication::exec();
}

} // namespace tradereview::app
