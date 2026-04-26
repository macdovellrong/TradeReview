#pragma once

#include <cstdint>
#include <functional>
#include <optional>
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

struct ChartCrosshairState {
    std::int64_t timestamp_ns = 0;
    double price = 0.0;
    double dense_x = 0.0;
};

class ChartViewWidget final : public QOpenGLWidget {
public:
    using ReloadRequestCallback = std::function<void(core::TimeRange)>;
    using CrosshairMovedCallback = std::function<void(std::int64_t, double)>;

    explicit ChartViewWidget(QWidget* parent = nullptr);
    ~ChartViewWidget() override;

    std::uint64_t bump_generation();
    bool apply_window(data::CandleWindow window);
    bool set_indicator_enabled(const std::string& indicator_name, bool enabled);
    bool set_bollinger_bands_enabled(bool enabled);
    bool set_indicator_panels_enabled(bool enabled);
    [[nodiscard]] std::vector<std::string> requested_indicators() const;
    void set_reload_request_callback(ReloadRequestCallback callback);
    void set_crosshair_moved_callback(CrosshairMovedCallback callback);
    void request_current_visible_window();
    [[nodiscard]] std::optional<double> dense_x_for_timestamp(std::int64_t timestamp_ns) const;
    void sync_crosshair(std::int64_t timestamp_ns, double price, double dense_x);
    bool sync_center_on_timestamp(std::int64_t timestamp_ns, std::optional<double> price = std::nullopt);
    bool sync_y_center(double price);
    [[nodiscard]] std::optional<ChartCrosshairState> crosshair_state() const;
    [[nodiscard]] std::optional<double> synced_y_center_price() const;
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
    [[nodiscard]] std::optional<ChartCrosshairState> crosshair_from_position(QPointF position) const;
    [[nodiscard]] double dense_x_at_pixel(double pixel_x) const;
    [[nodiscard]] std::optional<double> price_at_pixel_y(double pixel_y) const;

    ChartSceneModel scene_model_;
    ChartInteractionController interaction_;
    rendering::GLChartRenderer renderer_;
    ReloadRequestCallback reload_request_callback_;
    CrosshairMovedCallback crosshair_moved_callback_;
    std::optional<ChartCrosshairState> crosshair_state_;
    std::optional<double> synced_y_center_price_;
    QPointF last_mouse_position_;
    bool is_panning_ = false;
    bool has_last_reload_request_ = false;
    core::TimeRange last_reload_request_;
};

} // namespace tradereview::chart
