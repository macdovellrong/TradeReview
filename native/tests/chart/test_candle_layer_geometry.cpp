#include "tradereview/chart/rendering/CandleLayer.h"

#include "tradereview/core/Assertions.h"
#include "tradereview/data/CandleWindow.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::data::CandleWindow sample_window()
{
    tradereview::data::CandleWindow window;
    window.generation = 9;
    window.timestamp_ns = {100, 200};
    window.open = {10.0, 12.0};
    window.high = {13.0, 13.0};
    window.low = {9.0, 10.0};
    window.close = {12.0, 10.5};
    window.volume = {100.0, 200.0};
    return window;
}

void test_candle_geometry_builds_body_and_wick_vertices()
{
    const auto geometry = tradereview::chart::rendering::build_candle_geometry(sample_window());

    tradereview::core::assert_equal(geometry.generation, std::uint64_t{9}, "geometry generation");
    tradereview::core::assert_equal(geometry.grid_vertices.size(), std::size_t{16}, "stable grid uses eight GL line segments");
    tradereview::core::assert_equal(geometry.body_vertices.size(), std::size_t{12}, "two candle bodies use two triangles each");
    tradereview::core::assert_equal(geometry.wick_vertices.size(), std::size_t{4}, "two candle wicks use two line vertices each");
    tradereview::core::assert_true(!geometry.empty(), "geometry is not empty");

    const auto& first_body = geometry.body_vertices.front();
    const auto& second_body = geometry.body_vertices.at(6);
    tradereview::core::assert_true(first_body.green > first_body.red, "up candle body is green-dominant");
    tradereview::core::assert_true(second_body.red > second_body.green, "down candle body is red-dominant");
    tradereview::core::assert_true(first_body.x >= -1.0F && first_body.x <= 1.0F, "body x is normalized");
    tradereview::core::assert_true(first_body.y >= -1.0F && first_body.y <= 1.0F, "body y is normalized");

    tradereview::core::assert_near(
        geometry.wick_vertices.at(0).x,
        geometry.wick_vertices.at(1).x,
        0.000001,
        "wick line keeps one x coordinate");
}

void test_candle_geometry_skips_inconsistent_windows()
{
    auto window = sample_window();
    window.close.pop_back();

    const auto geometry = tradereview::chart::rendering::build_candle_geometry(window);

    tradereview::core::assert_true(geometry.empty(), "inconsistent candle geometry is empty");
    tradereview::core::assert_equal(geometry.generation, std::uint64_t{9}, "empty geometry keeps generation");
}

void test_candle_geometry_keeps_doji_body_visible()
{
    auto window = sample_window();
    window.open = {10.0, 12.0};
    window.close = {10.0, 10.5};

    const auto geometry = tradereview::chart::rendering::build_candle_geometry(window);

    float min_y = geometry.body_vertices.at(0).y;
    float max_y = geometry.body_vertices.at(0).y;
    for (std::size_t index = 0; index < 6; ++index) {
        min_y = std::min(min_y, geometry.body_vertices.at(index).y);
        max_y = std::max(max_y, geometry.body_vertices.at(index).y);
    }
    tradereview::core::assert_true(max_y > min_y, "doji candle body has visible height");
}

struct RegisterCandleLayerGeometryTests {
    RegisterCandleLayerGeometryTests()
    {
        tradereview::tests::register_test(
            "candle geometry builds body and wick vertices",
            test_candle_geometry_builds_body_and_wick_vertices);
        tradereview::tests::register_test(
            "candle geometry skips inconsistent windows",
            test_candle_geometry_skips_inconsistent_windows);
        tradereview::tests::register_test(
            "candle geometry keeps doji body visible",
            test_candle_geometry_keeps_doji_body_visible);
    }
};

const RegisterCandleLayerGeometryTests register_candle_layer_geometry_tests;

} // namespace
