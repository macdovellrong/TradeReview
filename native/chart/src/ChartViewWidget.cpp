#include "tradereview/chart/ChartViewWidget.h"

#include <QMouseEvent>
#include <QOpenGLContext>
#include <QWheelEvent>

#include <utility>

namespace tradereview::chart {

ChartViewWidget::ChartViewWidget(QWidget* parent)
    : QOpenGLWidget(parent)
{
    setMouseTracking(true);
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
        if (scene_model_.window().has_visible_range()) {
            interaction_.reset_for_visible_time_range(scene_model_.index_mapper(), scene_model_.window().visible_range);
        } else {
            interaction_.reset_for_row_count(scene_model_.row_count());
        }
        scene_model_.set_visible_dense_range(interaction_.visible_dense_range());
        has_last_reload_request_ = false;
        update();
    }
}

void ChartViewWidget::set_reload_request_callback(ReloadRequestCallback callback)
{
    reload_request_callback_ = std::move(callback);
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

void ChartViewWidget::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton) {
        is_panning_ = true;
        last_mouse_position_ = event->position();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
        return;
    }

    QOpenGLWidget::mousePressEvent(event);
}

void ChartViewWidget::mouseMoveEvent(QMouseEvent* event)
{
    if (is_panning_) {
        const auto current_position = event->position();
        interaction_.pan_by_pixels(current_position.x() - last_mouse_position_.x(), width());
        last_mouse_position_ = current_position;
        apply_interaction_update();
        event->accept();
        return;
    }

    QOpenGLWidget::mouseMoveEvent(event);
}

void ChartViewWidget::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && is_panning_) {
        is_panning_ = false;
        unsetCursor();
        event->accept();
        return;
    }

    QOpenGLWidget::mouseReleaseEvent(event);
}

void ChartViewWidget::wheelEvent(QWheelEvent* event)
{
    const auto scale = interaction_.zoom_scale_from_wheel_delta(event->angleDelta().y(), event->pixelDelta().y());
    if (scale == 1.0) {
        QOpenGLWidget::wheelEvent(event);
        return;
    }

    interaction_.zoom_at_pixel(event->position().x(), width(), scale);
    apply_interaction_update();
    event->accept();
}

void ChartViewWidget::apply_interaction_update()
{
    const auto range_changed = scene_model_.set_visible_dense_range(interaction_.visible_dense_range());
    const auto decision = interaction_.reload_decision(scene_model_.index_mapper(), scene_model_.window().loaded_range);
    if (decision.requested) {
        const auto repeated_request = has_last_reload_request_
            && last_reload_request_.start_ns == decision.visible_range.start_ns
            && last_reload_request_.end_ns == decision.visible_range.end_ns;
        if (!repeated_request && reload_request_callback_) {
            reload_request_callback_(decision.visible_range);
        }
        has_last_reload_request_ = true;
        last_reload_request_ = decision.visible_range;
    } else {
        has_last_reload_request_ = false;
    }

    if (range_changed) {
        update();
    }
}

} // namespace tradereview::chart
