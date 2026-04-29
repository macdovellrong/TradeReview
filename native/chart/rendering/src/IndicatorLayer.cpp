#include "tradereview/chart/rendering/IndicatorLayer.h"

#include "tradereview/data/IndicatorColumns.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <utility>
#include <string_view>

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

constexpr double kRsiMin = 0.0;
constexpr double kRsiMax = 100.0;
constexpr double kRsiLowerGuide = 20.0;
constexpr double kRsiUpperGuide = 80.0;
constexpr IndicatorVertex kRsiGuideColor{0.0F, 0.0F, 0.46F, 0.54F, 0.66F, 0.42F};

[[nodiscard]] bool finite(double value)
{
    return std::isfinite(value);
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

[[nodiscard]] float mapped_y(double value, double min_value, double max_value, PaneRect pane)
{
    if (max_value <= min_value) {
        return pane.bottom + pane.height() * 0.5F;
    }
    const auto normalized = (value - min_value) / (max_value - min_value);
    return pane.bottom + static_cast<float>(normalized) * pane.height();
}

[[nodiscard]] IndicatorVertex color_for_series(std::string_view name)
{
    using data::IndicatorColumns;
    if (name == IndicatorColumns::EMA20) {
        return {0.0F, 0.0F, 1.0F, 0.12F, 0.12F, 0.95F};
    }
    if (name == IndicatorColumns::EMA30) {
        return {0.0F, 0.0F, 1.0F, 0.53F, 0.0F, 0.95F};
    }
    if (name == IndicatorColumns::EMA40) {
        return {0.0F, 0.0F, 1.0F, 0.95F, 0.1F, 0.95F};
    }
    if (name == IndicatorColumns::EMA50) {
        return {0.0F, 0.0F, 0.1F, 0.9F, 0.28F, 0.95F};
    }
    if (name == IndicatorColumns::EMA60) {
        return {0.0F, 0.0F, 0.22F, 0.48F, 1.0F, 0.95F};
    }
    if (name == IndicatorColumns::EMA100) {
        return {0.0F, 0.0F, 0.0F, 0.74F, 1.0F, 0.95F};
    }
    if (name == IndicatorColumns::EMA240) {
        return {0.0F, 0.0F, 1.0F, 0.4F, 0.8F, 0.95F};
    }
    if (name == IndicatorColumns::MACD) {
        return {0.0F, 0.0F, 0.0F, 0.78F, 1.0F, 0.95F};
    }
    if (name == IndicatorColumns::MACD_Signal) {
        return {0.0F, 0.0F, 1.0F, 0.78F, 0.18F, 0.95F};
    }
    if (name == IndicatorColumns::RSI6) {
        return {0.0F, 0.0F, 0.55F, 1.0F, 0.45F, 0.95F};
    }
    if (name == IndicatorColumns::RSI12) {
        return {0.0F, 0.0F, 1.0F, 0.82F, 0.28F, 0.95F};
    }
    if (name == IndicatorColumns::RSI24) {
        return {0.0F, 0.0F, 0.42F, 0.68F, 1.0F, 0.95F};
    }
    if (name == IndicatorColumns::RSI) {
        return {0.0F, 0.0F, 0.72F, 1.0F, 0.72F, 0.85F};
    }
    return {0.0F, 0.0F, 0.72F, 0.72F, 0.72F, 0.9F};
}

void append_segment(
    IndicatorGeometry& geometry,
    float x1,
    float y1,
    float x2,
    float y2,
    IndicatorVertex color)
{
    color.x = x1;
    color.y = y1;
    geometry.vertices.push_back(color);
    color.x = x2;
    color.y = y2;
    geometry.vertices.push_back(color);
}

void add_series_values_to_range(
    const data::CandleWindow& window,
    const std::vector<std::string>& series_names,
    std::size_t first_row,
    std::size_t last_row,
    double& min_value,
    double& max_value)
{
    for (const auto& name : series_names) {
        const auto found = window.indicators.find(name);
        if (found == window.indicators.end() || found->second.size() != window.row_count()) {
            continue;
        }
        for (std::size_t row = first_row; row <= last_row; ++row) {
            const auto value = found->second[row];
            if (!finite(value)) {
                continue;
            }
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
    }
}

[[nodiscard]] IndicatorGeometry build_indicator_geometry(
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::vector<std::string>& series_names,
    bool include_price_range,
    std::optional<std::pair<double, double>> fixed_value_range = std::nullopt)
{
    IndicatorGeometry geometry;
    if (window.empty() || !window.has_consistent_columns()) {
        return geometry;
    }

    const auto range = normalized_visible_range(visible_dense_range);
    std::size_t first_row = 0;
    std::size_t last_row = 0;
    if (!visible_row_bounds(range, window.row_count(), first_row, last_row)) {
        return geometry;
    }

    auto min_value = std::numeric_limits<double>::max();
    auto max_value = std::numeric_limits<double>::lowest();
    if (fixed_value_range.has_value()) {
        min_value = fixed_value_range->first;
        max_value = fixed_value_range->second;
    } else if (include_price_range) {
        for (std::size_t row = first_row; row <= last_row; ++row) {
            if (!finite(window.low[row]) || !finite(window.high[row])) {
                continue;
            }
            min_value = std::min(min_value, window.low[row]);
            max_value = std::max(max_value, window.high[row]);
        }
    }
    if (!fixed_value_range.has_value()) {
        add_series_values_to_range(window, series_names, first_row, last_row, min_value, max_value);
    }
    if (min_value == std::numeric_limits<double>::max() || max_value == std::numeric_limits<double>::lowest()) {
        return geometry;
    }
    if (max_value <= min_value) {
        max_value = min_value + 1.0;
    }

    for (const auto& name : series_names) {
        const auto found = window.indicators.find(name);
        if (found == window.indicators.end() || found->second.size() != window.row_count()) {
            continue;
        }

        const auto color = color_for_series(name);
        for (std::size_t row = first_row + 1; row <= last_row; ++row) {
            const auto previous = found->second[row - 1];
            const auto current = found->second[row];
            if (!finite(previous) || !finite(current)) {
                continue;
            }
            append_segment(
                geometry,
                mapped_x(row - 1, range, pane),
                mapped_y(previous, min_value, max_value, pane),
                mapped_x(row, range, pane),
                mapped_y(current, min_value, max_value, pane),
                color);
        }
    }
    return geometry;
}

void append_rsi_guides(IndicatorGeometry& geometry, PaneRect pane)
{
    append_segment(
        geometry,
        pane.left,
        mapped_y(kRsiLowerGuide, kRsiMin, kRsiMax, pane),
        pane.right,
        mapped_y(kRsiLowerGuide, kRsiMin, kRsiMax, pane),
        kRsiGuideColor);
    append_segment(
        geometry,
        pane.left,
        mapped_y(kRsiUpperGuide, kRsiMin, kRsiMax, pane),
        pane.right,
        mapped_y(kRsiUpperGuide, kRsiMin, kRsiMax, pane),
        kRsiGuideColor);
}

} // namespace

IndicatorGeometry build_price_indicator_geometry(
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::vector<std::string>& series_names)
{
    return build_indicator_geometry(window, visible_dense_range, pane, series_names, true);
}

IndicatorGeometry build_panel_indicator_geometry(
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::vector<std::string>& series_names)
{
    return build_indicator_geometry(window, visible_dense_range, pane, series_names, false);
}

IndicatorGeometry build_rsi_indicator_geometry(
    const data::CandleWindow& window,
    DenseRange visible_dense_range,
    PaneRect pane,
    const std::vector<std::string>& series_names)
{
    auto geometry = build_indicator_geometry(
        window,
        visible_dense_range,
        pane,
        series_names,
        false,
        std::optional<std::pair<double, double>>{std::pair{kRsiMin, kRsiMax}});
    if (!geometry.vertices.empty()) {
        append_rsi_guides(geometry, pane);
    }
    return geometry;
}

void IndicatorLayer::initialize(QOpenGLFunctions_3_3_Core& gl)
{
    program_.create(gl, kVertexShader, kFragmentShader);
    vertex_array_.create(gl);
    buffer_.create(gl);
}

void IndicatorLayer::release(QOpenGLFunctions_3_3_Core& gl)
{
    buffer_.destroy(gl);
    vertex_array_.destroy(gl);
    program_.destroy(gl);
    uploaded_revision_ = 0;
    vertex_count_ = 0;
}

void IndicatorLayer::upload(QOpenGLFunctions_3_3_Core& gl, const IndicatorGeometry& geometry, std::uint64_t revision)
{
    if (uploaded_revision_ == revision) {
        return;
    }

    initialize(gl);
    gl.glBindVertexArray(vertex_array_.id());
    gl.glBindBuffer(GL_ARRAY_BUFFER, buffer_.id());
    gl.glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(geometry.vertices.size() * sizeof(IndicatorVertex)),
        geometry.vertices.empty() ? nullptr : geometry.vertices.data(),
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
    uploaded_revision_ = revision;
    vertex_count_ = geometry.vertices.size();
}

void IndicatorLayer::render(QOpenGLFunctions_3_3_Core& gl) const
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

std::size_t IndicatorLayer::vertex_count() const
{
    return vertex_count_;
}

} // namespace tradereview::chart::rendering
