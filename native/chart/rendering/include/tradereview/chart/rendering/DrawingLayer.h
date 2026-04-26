#pragma once

#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/chart/ChartInteractionController.h"
#include "tradereview/chart/PaneLayout.h"
#include "tradereview/chart/rendering/GLResources.h"
#include "tradereview/data/CandleWindow.h"
#include "tradereview/drawing/DrawingSpec.h"

#include <QOpenGLFunctions_3_3_Core>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace tradereview::chart::rendering {

struct DrawingVertex {
    float x = 0.0F;
    float y = 0.0F;
    float red = 1.0F;
    float green = 0.84F;
    float blue = 0.29F;
    float alpha = 1.0F;
};

struct DrawingGeometry {
    std::vector<DrawingVertex> vertices;
};

[[nodiscard]] DrawingGeometry build_drawing_geometry(
    const data::CandleWindow& window,
    const ChartIndexMapper& mapper,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::vector<drawing::DrawingSpec>& drawings,
    std::optional<drawing::DrawingSpec> preview);

class DrawingLayer final {
public:
    void initialize(QOpenGLFunctions_3_3_Core& gl);
    void release(QOpenGLFunctions_3_3_Core& gl);
    void upload(QOpenGLFunctions_3_3_Core& gl, const DrawingGeometry& geometry, std::uint64_t revision);
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
