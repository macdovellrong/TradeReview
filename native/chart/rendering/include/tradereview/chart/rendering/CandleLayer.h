#pragma once

#include "tradereview/chart/ChartInteractionController.h"
#include "tradereview/chart/PaneLayout.h"
#include "tradereview/chart/PriceRange.h"
#include "tradereview/chart/rendering/GLResources.h"
#include "tradereview/data/CandleWindow.h"

#include <QOpenGLFunctions_3_3_Core>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace tradereview::chart::rendering {

struct CandleVertex {
    float x = 0.0F;
    float y = 0.0F;
    float red = 0.0F;
    float green = 0.0F;
    float blue = 0.0F;
    float alpha = 1.0F;
};

struct CandleGeometry {
    std::uint64_t generation = 0;
    std::vector<CandleVertex> grid_vertices;
    std::vector<CandleVertex> body_vertices;
    std::vector<CandleVertex> wick_vertices;

    [[nodiscard]] bool empty() const;
};

[[nodiscard]] CandleGeometry build_candle_geometry(const data::CandleWindow& window);
[[nodiscard]] CandleGeometry build_candle_geometry(const data::CandleWindow& window, DenseRange visible_dense_range);
[[nodiscard]] CandleGeometry build_candle_geometry(
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    PaneRect pane,
    std::optional<PriceRange> price_range_override = std::nullopt);

class CandleLayer final {
public:
    void initialize(QOpenGLFunctions_3_3_Core& gl);
    void release(QOpenGLFunctions_3_3_Core& gl);
    void upload(
        QOpenGLFunctions_3_3_Core& gl,
        const data::CandleWindow& window,
        DenseRange visible_dense_range,
        PaneRect pane,
        std::uint64_t window_revision,
        std::optional<PriceRange> price_range_override = std::nullopt);
    void render(QOpenGLFunctions_3_3_Core& gl) const;

    [[nodiscard]] std::uint64_t uploaded_generation() const;
    [[nodiscard]] std::size_t body_vertex_count() const;
    [[nodiscard]] std::size_t wick_vertex_count() const;

private:
    void upload_vertices(
        QOpenGLFunctions_3_3_Core& gl,
        GLVertexArray& vertex_array,
        GLBuffer& buffer,
        const std::vector<CandleVertex>& vertices);

    GLProgram program_;
    GLVertexArray grid_vertex_array_;
    GLVertexArray body_vertex_array_;
    GLVertexArray wick_vertex_array_;
    GLBuffer grid_buffer_;
    GLBuffer body_buffer_;
    GLBuffer wick_buffer_;
    std::uint64_t uploaded_generation_ = 0;
    std::uint64_t uploaded_revision_ = 0;
    std::size_t grid_vertex_count_ = 0;
    std::size_t body_vertex_count_ = 0;
    std::size_t wick_vertex_count_ = 0;
};

} // namespace tradereview::chart::rendering
