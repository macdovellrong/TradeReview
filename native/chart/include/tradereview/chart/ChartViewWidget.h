#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <QPointF>
#include <QOpenGLWidget>

#include "tradereview/chart/ChartSceneModel.h"
#include "tradereview/chart/ChartInteractionController.h"
#include "tradereview/chart/rendering/GLChartRenderer.h"
#include "tradereview/core/TimeRange.h"
#include "tradereview/data/CandleWindow.h"

class QMouseEvent;
class QWheelEvent;

namespace tradereview::chart {

class ChartViewWidget final : public QOpenGLWidget {
public:
    using ReloadRequestCallback = std::function<void(core::TimeRange)>;

    explicit ChartViewWidget(QWidget* parent = nullptr);
    ~ChartViewWidget() override;

    std::uint64_t bump_generation();
    bool apply_window(data::CandleWindow window);
    bool set_indicator_enabled(const std::string& indicator_name, bool enabled);
    bool set_bollinger_bands_enabled(bool enabled);
    bool set_indicator_panels_enabled(bool enabled);
    [[nodiscard]] std::vector<std::string> requested_indicators() const;
    void set_reload_request_callback(ReloadRequestCallback callback);
    void request_current_visible_window();
    [[nodiscard]] const ChartSceneModel& scene_model() const;

private:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void release_renderer();
    void apply_interaction_update();

    ChartSceneModel scene_model_;
    ChartInteractionController interaction_;
    rendering::GLChartRenderer renderer_;
    ReloadRequestCallback reload_request_callback_;
    QPointF last_mouse_position_;
    bool is_panning_ = false;
    bool has_last_reload_request_ = false;
    core::TimeRange last_reload_request_;
};

} // namespace tradereview::chart
