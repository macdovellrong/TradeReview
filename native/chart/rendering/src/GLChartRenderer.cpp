#include "tradereview/chart/rendering/GLChartRenderer.h"

#include <QOpenGLContext>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLVersionFunctionsFactory>

namespace tradereview::chart::rendering {

namespace {

QOpenGLFunctions_3_3_Core* current_functions()
{
    QOpenGLContext* context = QOpenGLContext::currentContext();
    if (context == nullptr) {
        return nullptr;
    }
    auto* functions = QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_3_3_Core>(context);
    if (functions == nullptr) {
        return nullptr;
    }
    if (!functions->initializeOpenGLFunctions()) {
        return nullptr;
    }
    return functions;
}

} // namespace

void GLChartRenderer::initialize()
{
    QOpenGLFunctions_3_3_Core* functions = current_functions();
    if (functions == nullptr) {
        return;
    }

    functions->glClearColor(0.05F, 0.06F, 0.07F, 1.0F);
    functions->glDisable(GL_DEPTH_TEST);
    functions->glEnable(GL_BLEND);
    functions->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    candle_layer_.initialize(*functions);
    initialized_ = true;
}

void GLChartRenderer::release()
{
    QOpenGLFunctions_3_3_Core* functions = current_functions();
    if (functions == nullptr || !initialized_) {
        return;
    }

    candle_layer_.release(*functions);
    initialized_ = false;
}

void GLChartRenderer::resize(int width, int height)
{
    QOpenGLFunctions_3_3_Core* functions = current_functions();
    if (functions == nullptr) {
        return;
    }

    functions->glViewport(0, 0, width, height);
}

void GLChartRenderer::render(const ChartSceneModel& scene_model)
{
    QOpenGLFunctions_3_3_Core* functions = current_functions();
    if (functions == nullptr) {
        return;
    }
    if (!initialized_) {
        initialize();
        if (!initialized_) {
            return;
        }
    }

    functions->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    candle_layer_.upload(*functions, scene_model.window(), scene_model.visible_dense_range(), scene_model.revision());
    candle_layer_.render(*functions);
}

} // namespace tradereview::chart::rendering
