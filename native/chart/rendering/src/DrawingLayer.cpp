#include "tradereview/chart/rendering/DrawingLayer.h"

#include "tradereview/drawing/FibMath.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

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

[[nodiscard]] bool finite(double value)
{
    return std::isfinite(value);
}

[[nodiscard]] DenseRange normalized_visible_range(DenseRange range)
{
    if (!finite(range.start_x) || !finite(range.end_x)) {
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

[[nodiscard]] bool visible_price_range(
    const data::CandleWindow& window,
    DenseRange range,
    double& min_price,
    double& max_price)
{
    if (window.empty() || !window.has_consistent_ohlcv()) {
        return false;
    }

    std::size_t first_row = 0;
    std::size_t last_row = 0;
    if (!visible_row_bounds(range, window.row_count(), first_row, last_row)) {
        return false;
    }

    min_price = std::numeric_limits<double>::max();
    max_price = std::numeric_limits<double>::lowest();
    for (std::size_t row = first_row; row <= last_row; ++row) {
        if (!finite(window.low[row]) || !finite(window.high[row])) {
            continue;
        }
        min_price = std::min(min_price, window.low[row]);
        max_price = std::max(max_price, window.high[row]);
    }
    if (min_price == std::numeric_limits<double>::max() || max_price == std::numeric_limits<double>::lowest()) {
        return false;
    }
    if (max_price <= min_price) {
        max_price = min_price + 1.0;
    }
    return true;
}

[[nodiscard]] float mapped_x(double dense_x, DenseRange range, PaneRect pane)
{
    const auto span = std::max(range.span(), 1.0);
    const auto normalized = (dense_x - range.start_x) / span;
    return pane.left + static_cast<float>(normalized) * pane.width();
}

[[nodiscard]] float mapped_y(double price, double min_price, double max_price, PaneRect pane)
{
    if (max_price <= min_price) {
        return pane.bottom + pane.height() * 0.5F;
    }
    const auto normalized = (price - min_price) / (max_price - min_price);
    return pane.bottom + static_cast<float>(normalized) * pane.height();
}

[[nodiscard]] DrawingVertex committed_color()
{
    return DrawingVertex{0.0F, 0.0F, 1.0F, 0.84F, 0.29F, 0.95F};
}

[[nodiscard]] DrawingVertex preview_color()
{
    return DrawingVertex{0.0F, 0.0F, 0.0F, 0.9F, 1.0F, 0.5F};
}

void append_segment(DrawingGeometry& geometry, float x1, float y1, float x2, float y2, DrawingVertex color)
{
    color.x = x1;
    color.y = y1;
    geometry.vertices.push_back(color);
    color.x = x2;
    color.y = y2;
    geometry.vertices.push_back(color);
}

[[nodiscard]] std::optional<double> dense_x_for_point(
    const drawing::DrawingPoint& point,
    const ChartIndexMapper& mapper)
{
    if (mapper.empty()) {
        return std::nullopt;
    }
    return mapper.dense_x_from_timestamp(point.timestamp_ns);
}

void append_horizontal(
    DrawingGeometry& geometry,
    double price,
    float left,
    float right,
    double min_price,
    double max_price,
    PaneRect pane,
    DrawingVertex color)
{
    const auto y = mapped_y(price, min_price, max_price, pane);
    append_segment(geometry, left, y, right, y, color);
}

void append_spec(
    DrawingGeometry& geometry,
    const drawing::DrawingSpec& spec,
    const ChartIndexMapper& mapper,
    DenseRange range,
    PaneRect pane,
    double min_price,
    double max_price,
    bool preview)
{
    const auto color = preview ? preview_color() : committed_color();
    if (spec.type == drawing::DrawingType::HorizontalLine && !spec.points.empty()) {
        append_horizontal(geometry, spec.points[0].price, pane.left, pane.right, min_price, max_price, pane, color);
        return;
    }

    if (spec.type == drawing::DrawingType::VerticalLine && !spec.points.empty()) {
        const auto dense_x = dense_x_for_point(spec.points[0], mapper);
        if (!dense_x.has_value()) {
            return;
        }
        const auto x = mapped_x(*dense_x, range, pane);
        append_segment(geometry, x, pane.bottom, x, pane.top, color);
        return;
    }

    if (spec.type == drawing::DrawingType::Line && spec.points.size() >= 2) {
        const auto x1 = dense_x_for_point(spec.points[0], mapper);
        const auto x2 = dense_x_for_point(spec.points[1], mapper);
        if (!x1.has_value() || !x2.has_value()) {
            return;
        }
        append_segment(
            geometry,
            mapped_x(*x1, range, pane),
            mapped_y(spec.points[0].price, min_price, max_price, pane),
            mapped_x(*x2, range, pane),
            mapped_y(spec.points[1].price, min_price, max_price, pane),
            color);
        return;
    }

    if (spec.type == drawing::DrawingType::FibRetracement && spec.points.size() >= 2) {
        const auto x1 = dense_x_for_point(spec.points[0], mapper);
        const auto x2 = dense_x_for_point(spec.points[1], mapper);
        if (!x1.has_value() || !x2.has_value()) {
            return;
        }
        const auto left = std::min(mapped_x(*x1, range, pane), mapped_x(*x2, range, pane));
        const auto right = std::max(mapped_x(*x1, range, pane), mapped_x(*x2, range, pane));
        append_horizontal(geometry, spec.points[0].price, left, right, min_price, max_price, pane, color);
        append_horizontal(geometry, spec.points[1].price, left, right, min_price, max_price, pane, color);

        if (spec.fib_snapshot.has_value()) {
            for (const auto& level : drawing::build_retracement_levels(
                     spec.points[0].price,
                     spec.points[1].price,
                     spec.fib_snapshot->levels)) {
                append_horizontal(geometry, level.price, left, right, min_price, max_price, pane, color);
            }
        }
        return;
    }

    if (spec.type == drawing::DrawingType::FibExtension && spec.points.size() >= 3) {
        const auto x_a = dense_x_for_point(spec.points[0], mapper);
        const auto x_b = dense_x_for_point(spec.points[1], mapper);
        const auto x_c = dense_x_for_point(spec.points[2], mapper);
        if (!x_a.has_value() || !x_b.has_value() || !x_c.has_value()) {
            return;
        }

        append_segment(
            geometry,
            mapped_x(*x_a, range, pane),
            mapped_y(spec.points[0].price, min_price, max_price, pane),
            mapped_x(*x_b, range, pane),
            mapped_y(spec.points[1].price, min_price, max_price, pane),
            color);
        append_segment(
            geometry,
            mapped_x(*x_b, range, pane),
            mapped_y(spec.points[1].price, min_price, max_price, pane),
            mapped_x(*x_c, range, pane),
            mapped_y(spec.points[2].price, min_price, max_price, pane),
            color);

        if (spec.fib_snapshot.has_value()) {
            const auto projection_span = std::max(std::abs(*x_b - *x_a), 1.0);
            const auto left = mapped_x(*x_c, range, pane);
            const auto right = mapped_x(*x_c + projection_span, range, pane);
            for (const auto& level : drawing::build_extension_levels(
                     spec.points[0].price,
                     spec.points[1].price,
                     spec.points[2].price,
                     spec.fib_snapshot->levels)) {
                append_horizontal(geometry, level.price, left, right, min_price, max_price, pane, color);
            }
        }
    }
}

} // namespace

DrawingGeometry build_drawing_geometry(
    const data::CandleWindow& window,
    const ChartIndexMapper& mapper,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::vector<drawing::DrawingSpec>& drawings,
    std::optional<drawing::DrawingSpec> preview)
{
    DrawingGeometry geometry;
    const auto range = normalized_visible_range(visible_dense_range);
    double min_price = 0.0;
    double max_price = 0.0;
    if (!visible_price_range(window, range, min_price, max_price)) {
        return geometry;
    }

    for (const auto& drawing : drawings) {
        append_spec(geometry, drawing, mapper, range, pane, min_price, max_price, false);
    }
    if (preview.has_value()) {
        append_spec(geometry, *preview, mapper, range, pane, min_price, max_price, true);
    }
    return geometry;
}

void DrawingLayer::initialize(QOpenGLFunctions_3_3_Core& gl)
{
    program_.create(gl, kVertexShader, kFragmentShader);
    vertex_array_.create(gl);
    buffer_.create(gl);
}

void DrawingLayer::release(QOpenGLFunctions_3_3_Core& gl)
{
    buffer_.destroy(gl);
    vertex_array_.destroy(gl);
    program_.destroy(gl);
    uploaded_revision_ = 0;
    vertex_count_ = 0;
}

void DrawingLayer::upload(QOpenGLFunctions_3_3_Core& gl, const DrawingGeometry& geometry, std::uint64_t revision)
{
    if (uploaded_revision_ == revision) {
        return;
    }

    initialize(gl);
    gl.glBindVertexArray(vertex_array_.id());
    gl.glBindBuffer(GL_ARRAY_BUFFER, buffer_.id());
    gl.glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(geometry.vertices.size() * sizeof(DrawingVertex)),
        geometry.vertices.empty() ? nullptr : geometry.vertices.data(),
        GL_STATIC_DRAW);

    gl.glEnableVertexAttribArray(0);
    gl.glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(DrawingVertex)),
        reinterpret_cast<void*>(offsetof(DrawingVertex, x)));
    gl.glEnableVertexAttribArray(1);
    gl.glVertexAttribPointer(
        1,
        4,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(DrawingVertex)),
        reinterpret_cast<void*>(offsetof(DrawingVertex, red)));
    gl.glBindBuffer(GL_ARRAY_BUFFER, 0);
    gl.glBindVertexArray(0);
    uploaded_revision_ = revision;
    vertex_count_ = geometry.vertices.size();
}

void DrawingLayer::render(QOpenGLFunctions_3_3_Core& gl) const
{
    if (!program_.valid() || vertex_count_ == 0) {
        return;
    }

    gl.glUseProgram(program_.id());
    gl.glBindVertexArray(vertex_array_.id());
    gl.glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertex_count_));
    gl.glBindVertexArray(0);
    gl.glUseProgram(0);
}

std::size_t DrawingLayer::vertex_count() const
{
    return vertex_count_;
}

} // namespace tradereview::chart::rendering
