#pragma once

#include <cstdint>

#include <QOpenGLWidget>

#include "tradereview/chart/ChartSceneModel.h"
#include "tradereview/chart/rendering/GLChartRenderer.h"
#include "tradereview/data/CandleWindow.h"

namespace tradereview::chart {

class ChartViewWidget final : public QOpenGLWidget {
public:
    explicit ChartViewWidget(QWidget* parent = nullptr);
    ~ChartViewWidget() override;

    std::uint64_t bump_generation();
    void apply_window(data::CandleWindow window);
    [[nodiscard]] const ChartSceneModel& scene_model() const;

private:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;
    void release_renderer();

    ChartSceneModel scene_model_;
    rendering::GLChartRenderer renderer_;
};

} // namespace tradereview::chart
