#include "tradereview/chart/rendering/HistogramLayer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tradereview::chart::rendering {
namespace {

constexpr char kVertexShader[] = R"(#version 330 core
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec4 in_color;
out vec4 vertex_color;

void main()
{
    gl_Position = vec4(in_position, 0.0, 1.0);
    vertex_color = in_color;
}
)";

constexpr char kFragmentShader[] = R"(#version 330 core
in vec4 vertex_color;
out vec4 out_color;

void main()
{
    out_color = vertex_color;
}
)";

constexpr IndicatorVertex kPositiveColor{0.0F, 0.0F, 0.18F, 0.74F, 0.42F, 0.7F};
constexpr IndicatorVertex kNegativeColor{0.0F, 0.0F, 0.86F, 0.28F, 0.24F, 0.7F};

[[nodiscard]] DenseRange normalized_visible_range(DenseRange range)
{
    if (!std::isfinite(range.start_x) || !std::isfinite(range.end_x)) {
        return {0.0, 1.0};
    }
    if (range.end_x < range.start_x) {
        std::swap(range.start_x, range.end_x);
    }
    if (range.end_x <= range.start_x) {
        range.end_x = range.start_x + 1.0;
    }
    return range;
}

[[nodiscard]] bool visible_row_bounds(DenseRange range, std::size_t rows, std::size_t& first_row, std::size_t& last_row)
{
    if (rows == 0) {
        return false;
    }
    const auto first_visible_row = std::ceil(range.start_x);
    const auto last_visible_row = std::floor(range.end_x);
    if (last_visible_row < 0.0 || first_visible_row > static_cast<double>(rows - 1)) {
        return false;
    }

    first_row = static_cast<std::size_t>(std::max(0.0, first_visible_row));
    last_row = static_cast<std::size_t>(std::min(static_cast<double>(rows - 1), last_visible_row));
    return last_row >= first_row;
}

[[nodiscard]] float mapped_x(std::size_t row, DenseRange range, PaneRect pane)
{
    const auto span = std::max(range.span(), 1.0);
    const auto normalized = (static_cast<double>(row) - range.start_x) / span;
    return pane.left + static_cast<float>(normalized) * pane.width();
}

[[nodiscard]] float mapped_y(double value, double max_abs, PaneRect pane)
{
    if (max_abs <= 0.0) {
        return pane.bottom + pane.height() * 0.5F;
    }
    const auto normalized = (value / max_abs + 1.0) * 0.5;
    return pane.bottom + static_cast<float>(normalized) * pane.height();
}

void append_bar(
    std::vector<IndicatorVertex>& vertices,
    float left,
    float right,
    float zero_y,
    float value_y,
    IndicatorVertex color)
{
    const auto bottom = std::min(zero_y, value_y);
    const auto top = std::max(zero_y, value_y);
    color.x = left;
    color.y = bottom;
    vertices.push_back(color);
    color.x = right;
    color.y = bottom;
    vertices.push_back(color);
    color.x = right;
    color.y = top;
    vertices.push_back(color);

    color.x = left;
    color.y = bottom;
    vertices.push_back(color);
    color.x = right;
    color.y = top;
    vertices.push_back(color);
    color.x = left;
    color.y = top;
    vertices.push_back(color);
}

} // namespace

HistogramGeometry build_histogram_geometry(
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::string& series_name)
{
    HistogramGeometry geometry;
    if (window.empty() || !window.has_consistent_columns()) {
        return geometry;
    }

    const auto found = window.indicators.find(series_name);
    if (found == window.indicators.end() || found->second.size() != window.row_count()) {
        return geometry;
    }

    const auto range = normalized_visible_range(visible_dense_range);
    std::size_t first_row = 0;
    std::size_t last_row = 0;
    if (!visible_row_bounds(range, window.row_count(), first_row, last_row)) {
        return geometry;
    }

    auto max_abs = 0.0;
    for (std::size_t row = first_row; row <= last_row; ++row) {
        const auto value = found->second[row];
        if (std::isfinite(value)) {
            max_abs = std::max(max_abs, std::abs(value));
        }
    }
    if (max_abs <= 0.0) {
        return geometry;
    }

    const auto span = static_cast<float>(std::max(range.span(), 1.0));
    const auto half_width = std::min(0.025F, 0.36F * (pane.width() / span));
    const auto zero_y = mapped_y(0.0, max_abs, pane);
    for (std::size_t row = first_row; row <= last_row; ++row) {
        const auto value = found->second[row];
        if (!std::isfinite(value) || value == 0.0) {
            continue;
        }
        const auto center_x = mapped_x(row, range, pane);
        const auto left = std::clamp(center_x - half_width, pane.left, pane.right);
        const auto right = std::clamp(center_x + half_width, pane.left, pane.right);
        auto& target = value >= 0.0 ? geometry.positive_vertices : geometry.negative_vertices;
        append_bar(target, left, right, zero_y, mapped_y(value, max_abs, pane), value >= 0.0 ? kPositiveColor : kNegativeColor);
    }
    return geometry;
}

void HistogramLayer::initialize(QOpenGLFunctions_3_3_Core& gl)
{
    program_.create(gl, kVertexShader, kFragmentShader);
    positive_vertex_array_.create(gl);
    negative_vertex_array_.create(gl);
    positive_buffer_.create(gl);
    negative_buffer_.create(gl);
}

void HistogramLayer::release(QOpenGLFunctions_3_3_Core& gl)
{
    positive_buffer_.destroy(gl);
    negative_buffer_.destroy(gl);
    positive_vertex_array_.destroy(gl);
    negative_vertex_array_.destroy(gl);
    program_.destroy(gl);
    uploaded_revision_ = 0;
    positive_vertex_count_ = 0;
    negative_vertex_count_ = 0;
}

void HistogramLayer::upload(QOpenGLFunctions_3_3_Core& gl, const HistogramGeometry& geometry, std::uint64_t revision)
{
    if (uploaded_revision_ == revision) {
        return;
    }

    initialize(gl);
    upload_vertices(gl, positive_vertex_array_, positive_buffer_, geometry.positive_vertices);
    upload_vertices(gl, negative_vertex_array_, negative_buffer_, geometry.negative_vertices);
    uploaded_revision_ = revision;
    positive_vertex_count_ = geometry.positive_vertices.size();
    negative_vertex_count_ = geometry.negative_vertices.size();
}

void HistogramLayer::render(QOpenGLFunctions_3_3_Core& gl) const
{
    if (!program_.valid()) {
        return;
    }

    gl.glUseProgram(program_.id());
    if (positive_vertex_count_ > 0) {
        gl.glBindVertexArray(positive_vertex_array_.id());
        gl.glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(positive_vertex_count_));
    }
    if (negative_vertex_count_ > 0) {
        gl.glBindVertexArray(negative_vertex_array_.id());
        gl.glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(negative_vertex_count_));
    }
    gl.glBindVertexArray(0);
    gl.glUseProgram(0);
}

std::size_t HistogramLayer::positive_vertex_count() const
{
    return positive_vertex_count_;
}

std::size_t HistogramLayer::negative_vertex_count() const
{
    return negative_vertex_count_;
}

void HistogramLayer::upload_vertices(
    QOpenGLFunctions_3_3_Core& gl,
    GLVertexArray& vertex_array,
    GLBuffer& buffer,
    const std::vector<IndicatorVertex>& vertices)
{
    gl.glBindVertexArray(vertex_array.id());
    gl.glBindBuffer(GL_ARRAY_BUFFER, buffer.id());
    gl.glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(IndicatorVertex)),
        vertices.empty() ? nullptr : vertices.data(),
        GL_STATIC_DRAW);

    gl.glEnableVertexAttribArray(0);
    gl.glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(IndicatorVertex)),
        reinterpret_cast<void*>(offsetof(IndicatorVertex, x)));
    gl.glEnableVertexAttribArray(1);
    gl.glVertexAttribPointer(
        1,
        4,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(IndicatorVertex)),
        reinterpret_cast<void*>(offsetof(IndicatorVertex, red)));
    gl.glBindBuffer(GL_ARRAY_BUFFER, 0);
    gl.glBindVertexArray(0);
}

} // namespace tradereview::chart::rendering
