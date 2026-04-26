#include "tradereview/chart/ChartViewWidget.h"

#include <QOpenGLContext>

#include <utility>

namespace tradereview::chart {

ChartViewWidget::ChartViewWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
}

ChartViewWidget::~ChartViewWidget()
{
    release_renderer();
}

void ChartViewWidget::release_renderer()
{
    if (context() == nullptr) {
        return;
    }

    makeCurrent();
    renderer_.release();
    doneCurrent();
}

std::uint64_t ChartViewWidget::bump_generation()
{
    return scene_model_.bump_generation();
}

void ChartViewWidget::apply_window(data::CandleWindow window)
{
    if (scene_model_.apply_window(std::move(window))) {
        update();
    }
}

const ChartSceneModel& ChartViewWidget::scene_model() const
{
    return scene_model_;
}

void ChartViewWidget::initializeGL()
{
    QObject::connect(
        context(),
        &QOpenGLContext::aboutToBeDestroyed,
        this,
        &ChartViewWidget::release_renderer,
        Qt::DirectConnection);
    renderer_.initialize();
}

void ChartViewWidget::resizeGL(int width, int height)
{
    renderer_.resize(width, height);
}

void ChartViewWidget::paintGL()
{
    renderer_.render(scene_model_);
}

} // namespace tradereview::chart
