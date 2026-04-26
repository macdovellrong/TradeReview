#include "tradereview/chart/rendering/CandleLayer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>

namespace tradereview::chart::rendering {
namespace {

constexpr CandleVertex kUpColor{0.0F, 0.0F, 0.18F, 0.75F, 0.38F, 1.0F};
constexpr CandleVertex kDownColor{0.0F, 0.0F, 0.86F, 0.28F, 0.24F, 1.0F};
constexpr CandleVertex kGridColor{0.0F, 0.0F, 0.24F, 0.27F, 0.30F, 0.55F};
constexpr float kMinimumBodyHeight = 0.006F;

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

[[nodiscard]] bool finite_candle_row(const data::CandleWindow& window, std::size_t row)
{
    return std::isfinite(window.open[row]) &&
        std::isfinite(window.high[row]) &&
        std::isfinite(window.low[row]) &&
        std::isfinite(window.close[row]);
}

[[nodiscard]] float normalized_y(double value, double min_value, double max_value)
{
    if (max_value <= min_value) {
        return 0.0F;
    }
    const auto normalized = (value - min_value) / (max_value - min_value);
    return static_cast<float>((normalized * 2.0) - 1.0);
}

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
    const auto first_visible_row = std::ceil(range.start_x);
    const auto last_visible_row = std::floor(range.end_x);
    if (last_visible_row < 0.0 || first_visible_row > static_cast<double>(rows - 1)) {
        return false;
    }

    first_row = static_cast<std::size_t>(std::max(0.0, first_visible_row));
    last_row = static_cast<std::size_t>(std::min(static_cast<double>(rows - 1), last_visible_row));
    if (last_row < first_row) {
        return false;
    }
    return true;
}

[[nodiscard]] float candle_center_x(std::size_t row, DenseRange visible_dense_range)
{
    const auto span = std::max(visible_dense_range.span(), 1.0);
    const auto normalized = (static_cast<double>(row) - visible_dense_range.start_x) / span;
    return static_cast<float>((normalized * 2.0) - 1.0);
}

[[nodiscard]] float candle_half_width(DenseRange visible_dense_range)
{
    const auto span = static_cast<float>(std::max(visible_dense_range.span(), 1.0));
    return std::min(0.32F, 0.35F * (2.0F / span));
}

void append_body(
    std::vector<CandleVertex>& vertices,
    float left,
    float right,
    float bottom,
    float top,
    CandleVertex color)
{
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

void append_wick(std::vector<CandleVertex>& vertices, float x, float low, float high, CandleVertex color)
{
    color.x = x;
    color.y = low;
    vertices.push_back(color);
    color.y = high;
    vertices.push_back(color);
}

void append_line(std::vector<CandleVertex>& vertices, float start_x, float start_y, float end_x, float end_y)
{
    auto color = kGridColor;
    color.x = start_x;
    color.y = start_y;
    vertices.push_back(color);
    color.x = end_x;
    color.y = end_y;
    vertices.push_back(color);
}

void append_grid(std::vector<CandleVertex>& vertices)
{
    constexpr float kGridLines[] = {-0.6F, -0.2F, 0.2F, 0.6F};
    vertices.reserve(std::size(kGridLines) * 4);
    for (const float value : kGridLines) {
        append_line(vertices, -1.0F, value, 1.0F, value);
        append_line(vertices, value, -1.0F, value, 1.0F);
    }
}

} // namespace

bool CandleGeometry::empty() const
{
    return grid_vertices.empty() && body_vertices.empty() && wick_vertices.empty();
}

CandleGeometry build_candle_geometry(const data::CandleWindow& window)
{
    if (window.empty()) {
        return CandleGeometry{window.generation};
    }
    return build_candle_geometry(window, DenseRange{0.0, static_cast<double>(window.row_count() - 1)});
}

CandleGeometry build_candle_geometry(const data::CandleWindow& window, DenseRange visible_dense_range)
{
    CandleGeometry geometry;
    geometry.generation = window.generation;
    if (window.empty() || !window.has_consistent_ohlcv()) {
        return geometry;
    }

    const auto range = normalized_visible_range(visible_dense_range);
    const auto rows = window.row_count();
    std::size_t first_row = 0;
    std::size_t last_row = 0;
    if (!visible_row_bounds(range, rows, first_row, last_row)) {
        append_grid(geometry.grid_vertices);
        return geometry;
    }

    auto min_price = std::numeric_limits<double>::max();
    auto max_price = std::numeric_limits<double>::lowest();
    for (std::size_t row = first_row; row <= last_row; ++row) {
        if (!finite_candle_row(window, row)) {
            continue;
        }
        min_price = std::min(min_price, window.low[row]);
        max_price = std::max(max_price, window.high[row]);
    }

    if (min_price == std::numeric_limits<double>::max() || max_price == std::numeric_limits<double>::lowest()) {
        return geometry;
    }
    if (max_price <= min_price) {
        max_price = min_price + 1.0;
    }

    append_grid(geometry.grid_vertices);
    const auto half_width = candle_half_width(range);
    geometry.body_vertices.reserve((last_row - first_row + 1) * 6);
    geometry.wick_vertices.reserve((last_row - first_row + 1) * 2);
    for (std::size_t row = first_row; row <= last_row; ++row) {
        if (!finite_candle_row(window, row)) {
            continue;
        }

        const auto center_x = candle_center_x(row, range);
        const auto left = std::clamp(center_x - half_width, -1.0F, 1.0F);
        const auto right = std::clamp(center_x + half_width, -1.0F, 1.0F);
        const auto open_y = normalized_y(window.open[row], min_price, max_price);
        const auto close_y = normalized_y(window.close[row], min_price, max_price);
        const auto high_y = normalized_y(window.high[row], min_price, max_price);
        const auto low_y = normalized_y(window.low[row], min_price, max_price);
        auto top = std::max(open_y, close_y);
        auto bottom = std::min(open_y, close_y);
        if (top - bottom < kMinimumBodyHeight) {
            const auto middle = (top + bottom) * 0.5F;
            bottom = std::clamp(middle - (kMinimumBodyHeight * 0.5F), -1.0F, 1.0F - kMinimumBodyHeight);
            top = bottom + kMinimumBodyHeight;
        }
        const auto color = window.close[row] >= window.open[row] ? kUpColor : kDownColor;

        append_body(geometry.body_vertices, left, right, bottom, top, color);
        append_wick(geometry.wick_vertices, center_x, low_y, high_y, color);
    }
    return geometry;
}

void CandleLayer::initialize(QOpenGLFunctions_3_3_Core& gl)
{
    program_.create(gl, kVertexShader, kFragmentShader);
    grid_vertex_array_.create(gl);
    body_vertex_array_.create(gl);
    wick_vertex_array_.create(gl);
    grid_buffer_.create(gl);
    body_buffer_.create(gl);
    wick_buffer_.create(gl);
}

void CandleLayer::release(QOpenGLFunctions_3_3_Core& gl)
{
    grid_buffer_.destroy(gl);
    body_buffer_.destroy(gl);
    wick_buffer_.destroy(gl);
    grid_vertex_array_.destroy(gl);
    body_vertex_array_.destroy(gl);
    wick_vertex_array_.destroy(gl);
    program_.destroy(gl);
    uploaded_generation_ = 0;
    uploaded_revision_ = 0;
    grid_vertex_count_ = 0;
    body_vertex_count_ = 0;
    wick_vertex_count_ = 0;
}

void CandleLayer::upload(
    QOpenGLFunctions_3_3_Core& gl,
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    std::uint64_t window_revision)
{
    if (uploaded_generation_ == window.generation && uploaded_revision_ == window_revision) {
        return;
    }

    initialize(gl);
    const auto geometry = build_candle_geometry(window, visible_dense_range);
    upload_vertices(gl, grid_vertex_array_, grid_buffer_, geometry.grid_vertices);
    upload_vertices(gl, body_vertex_array_, body_buffer_, geometry.body_vertices);
    upload_vertices(gl, wick_vertex_array_, wick_buffer_, geometry.wick_vertices);
    uploaded_generation_ = geometry.generation;
    uploaded_revision_ = window_revision;
    grid_vertex_count_ = geometry.grid_vertices.size();
    body_vertex_count_ = geometry.body_vertices.size();
    wick_vertex_count_ = geometry.wick_vertices.size();
}

void CandleLayer::render(QOpenGLFunctions_3_3_Core& gl) const
{
    if (!program_.valid()) {
        return;
    }

    gl.glUseProgram(program_.id());
    if (grid_vertex_count_ > 0) {
        gl.glBindVertexArray(grid_vertex_array_.id());
        gl.glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(grid_vertex_count_));
    }
    if (wick_vertex_count_ > 0) {
        gl.glBindVertexArray(wick_vertex_array_.id());
        gl.glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(wick_vertex_count_));
    }
    if (body_vertex_count_ > 0) {
        gl.glBindVertexArray(body_vertex_array_.id());
        gl.glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(body_vertex_count_));
    }
    gl.glBindVertexArray(0);
    gl.glUseProgram(0);
}

std::uint64_t CandleLayer::uploaded_generation() const
{
    return uploaded_generation_;
}

std::size_t CandleLayer::body_vertex_count() const
{
    return body_vertex_count_;
}

std::size_t CandleLayer::wick_vertex_count() const
{
    return wick_vertex_count_;
}

void CandleLayer::upload_vertices(
    QOpenGLFunctions_3_3_Core& gl,
    GLVertexArray& vertex_array,
    GLBuffer& buffer,
    const std::vector<CandleVertex>& vertices)
{
    gl.glBindVertexArray(vertex_array.id());
    gl.glBindBuffer(GL_ARRAY_BUFFER, buffer.id());
    gl.glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(CandleVertex)),
        vertices.empty() ? nullptr : vertices.data(),
        GL_STATIC_DRAW);

    gl.glEnableVertexAttribArray(0);
    gl.glVertexAttribPointer(
        0,
        2,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(CandleVertex)),
        reinterpret_cast<void*>(offsetof(CandleVertex, x)));
    gl.glEnableVertexAttribArray(1);
    gl.glVertexAttribPointer(
        1,
        4,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(CandleVertex)),
        reinterpret_cast<void*>(offsetof(CandleVertex, red)));
    gl.glBindBuffer(GL_ARRAY_BUFFER, 0);
    gl.glBindVertexArray(0);
}

} // namespace tradereview::chart::rendering
