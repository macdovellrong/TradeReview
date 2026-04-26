#include "tradereview/chart/ChartViewWidget.h"

#include <utility>

namespace tradereview::chart {

ChartViewWidget::ChartViewWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
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
