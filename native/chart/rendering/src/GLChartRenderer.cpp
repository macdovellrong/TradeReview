#include "tradereview/chart/rendering/GLChartRenderer.h"

#include <QOpenGLContext>
#include <QOpenGLFunctions>

namespace tradereview::chart::rendering {

namespace {

QOpenGLFunctions* current_functions()
{
    QOpenGLContext* context = QOpenGLContext::currentContext();
    if (context == nullptr) {
        return nullptr;
    }
    return context->functions();
}

} // namespace

void GLChartRenderer::initialize()
{
    QOpenGLFunctions* functions = current_functions();
    if (functions == nullptr) {
        return;
    }

    functions->glClearColor(0.05F, 0.06F, 0.07F, 1.0F);
}

void GLChartRenderer::resize(int width, int height)
{
    QOpenGLFunctions* functions = current_functions();
    if (functions == nullptr) {
        return;
    }

    functions->glViewport(0, 0, width, height);
}

void GLChartRenderer::render(const ChartSceneModel& scene_model)
{
    (void)scene_model;

    QOpenGLFunctions* functions = current_functions();
    if (functions == nullptr) {
        return;
    }

    functions->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

} // namespace tradereview::chart::rendering
