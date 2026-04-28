#include "tradereview/app/NativeApp.h"

#include "tradereview/app/AppTheme.h"
#include "tradereview/app/MainWindow.h"

#include <QApplication>
#include <QSurfaceFormat>

namespace tradereview::app {

int run_native_app(int argc, char** argv)
{
    QSurfaceFormat format;
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    QSurfaceFormat::setDefaultFormat(format);

    QApplication app(argc, argv);
    theme::apply(app);
    MainWindow window;
    window.show();
    return QApplication::exec();
}

} // namespace tradereview::app
