#pragma once

#include "tradereview/chart/ChartInteractionController.h"
#include "tradereview/chart/PaneLayout.h"
#include "tradereview/chart/rendering/GLResources.h"
#include "tradereview/chart/rendering/IndicatorLayer.h"
#include "tradereview/data/CandleWindow.h"

#include <QOpenGLFunctions_3_3_Core>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tradereview::chart::rendering {

struct HistogramGeometry {
    std::vector<IndicatorVertex> positive_vertices;
    std::vector<IndicatorVertex> negative_vertices;
};

[[nodiscard]] HistogramGeometry build_histogram_geometry(
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::string& series_name);

class HistogramLayer final {
public:
    void initialize(QOpenGLFunctions_3_3_Core& gl);
    void release(QOpenGLFunctions_3_3_Core& gl);
    void upload(QOpenGLFunctions_3_3_Core& gl, const HistogramGeometry& geometry, std::uint64_t revision);
    void render(QOpenGLFunctions_3_3_Core& gl) const;

    [[nodiscard]] std::size_t positive_vertex_count() const;
    [[nodiscard]] std::size_t negative_vertex_count() const;

private:
    void upload_vertices(
        QOpenGLFunctions_3_3_Core& gl,
        GLVertexArray& vertex_array,
        GLBuffer& buffer,
        const std::vector<IndicatorVertex>& vertices);

    GLProgram program_;
    GLVertexArray positive_vertex_array_;
    GLVertexArray negative_vertex_array_;
    GLBuffer positive_buffer_;
    GLBuffer negative_buffer_;
    std::uint64_t uploaded_revision_ = 0;
    std::size_t positive_vertex_count_ = 0;
    std::size_t negative_vertex_count_ = 0;
};

} // namespace tradereview::chart::rendering
