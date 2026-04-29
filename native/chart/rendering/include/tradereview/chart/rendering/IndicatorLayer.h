#pragma once

#include "tradereview/chart/ChartInteractionController.h"
#include "tradereview/chart/PaneLayout.h"
#include "tradereview/chart/rendering/GLResources.h"
#include "tradereview/data/CandleWindow.h"

#include <QOpenGLFunctions_3_3_Core>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tradereview::chart::rendering {

struct IndicatorVertex {
    float x = 0.0F;
    float y = 0.0F;
    float red = 0.0F;
    float green = 0.0F;
    float blue = 0.0F;
    float alpha = 1.0F;
};

struct IndicatorGeometry {
    std::vector<IndicatorVertex> vertices;
};

[[nodiscard]] IndicatorGeometry build_price_indicator_geometry(
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::vector<std::string>& series_names);

[[nodiscard]] IndicatorGeometry build_panel_indicator_geometry(
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::vector<std::string>& series_names);

[[nodiscard]] IndicatorGeometry build_rsi_indicator_geometry(
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::vector<std::string>& series_names);

class IndicatorLayer final {
public:
    void initialize(QOpenGLFunctions_3_3_Core& gl);
    void release(QOpenGLFunctions_3_3_Core& gl);
    void upload(QOpenGLFunctions_3_3_Core& gl, const IndicatorGeometry& geometry, std::uint64_t revision);
    void render(QOpenGLFunctions_3_3_Core& gl) const;

    [[nodiscard]] std::size_t vertex_count() const;

private:
    GLProgram program_;
    GLVertexArray vertex_array_;
    GLBuffer buffer_;
    std::uint64_t uploaded_revision_ = 0;
    std::size_t vertex_count_ = 0;
};

} // namespace tradereview::chart::rendering
