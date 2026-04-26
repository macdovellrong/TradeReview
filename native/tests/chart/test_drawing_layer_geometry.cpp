#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/chart/PaneLayout.h"
#include "tradereview/chart/rendering/DrawingLayer.h"
#include "tradereview/core/Assertions.h"
#include "tradereview/data/CandleWindow.h"
#include "tradereview/drawing/DrawingSpec.h"
#include "tradereview/drawing/FibSettings.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::drawing::DrawingPoint point(std::int64_t timestamp_ns, double price)
{
    return tradereview::drawing::DrawingPoint{timestamp_ns, price};
}

tradereview::data::CandleWindow drawing_window()
{
    tradereview::data::CandleWindow window;
    window.timestamp_ns = {100, 200, 300};
    window.open = {100.0, 120.0, 110.0};
    window.high = {110.0, 130.0, 125.0};
    window.low = {90.0, 105.0, 95.0};
    window.close = {105.0, 115.0, 118.0};
    window.volume = {1.0, 1.0, 1.0};
    return window;
}

tradereview::chart::ChartIndexMapper mapper_for(const tradereview::data::CandleWindow& window)
{
    tradereview::chart::ChartIndexMapper mapper;
    mapper.set_timestamps(window.timestamp_ns);
    return mapper;
}

void test_drawing_layer_maps_line_timestamp_price_to_pane_coordinates()
{
    const auto window = drawing_window();
    auto mapper = mapper_for(window);
    const auto line = tradereview::drawing::normalize_drawing_spec(
        tradereview::drawing::DrawingType::Line,
        {point(100, 100.0), point(300, 120.0)});

    const auto geometry = tradereview::chart::rendering::build_drawing_geometry(
        window,
        mapper,
        {0.0, 2.0},
        {-1.0F, 1.0F, -1.0F, 1.0F},
        {line},
        std::nullopt);

    tradereview::core::assert_equal(geometry.vertices.size(), std::size_t{2}, "line vertex count");
    tradereview::core::assert_near(geometry.vertices[0].x, -1.0, 0.000001, "line first x");
    tradereview::core::assert_near(geometry.vertices[0].y, -0.5, 0.000001, "line first y");
    tradereview::core::assert_near(geometry.vertices[1].x, 1.0, 0.000001, "line second x");
    tradereview::core::assert_near(geometry.vertices[1].y, 0.5, 0.000001, "line second y");
}

void test_drawing_layer_builds_horizontal_vertical_and_fib_geometry()
{
    const auto window = drawing_window();
    auto mapper = mapper_for(window);
    const auto settings = tradereview::drawing::FibSettings{
        tradereview::drawing::FibLevelsConfig{{0.5, 0.618}, ""},
        tradereview::drawing::FibLevelsConfig{{1.0}, ""},
    };
    std::vector<tradereview::drawing::DrawingSpec> drawings{
        tradereview::drawing::normalize_drawing_spec(
            tradereview::drawing::DrawingType::HorizontalLine,
            {point(100, 110.0)}),
        tradereview::drawing::normalize_drawing_spec(
            tradereview::drawing::DrawingType::VerticalLine,
            {point(200, 110.0)}),
        tradereview::drawing::create_drawing_spec(
            tradereview::drawing::DrawingType::FibRetracement,
            {point(100, 100.0), point(300, 120.0)},
            settings),
    };

    const auto geometry = tradereview::chart::rendering::build_drawing_geometry(
        window,
        mapper,
        {0.0, 2.0},
        {-1.0F, 1.0F, -1.0F, 1.0F},
        drawings,
        std::nullopt);

    tradereview::core::assert_equal(
        geometry.vertices.size(),
        std::size_t{12},
        "hline vline and fib geometry vertex count");
}

void test_drawing_layer_builds_fib_extension_and_preview_geometry()
{
    const auto window = drawing_window();
    auto mapper = mapper_for(window);
    const auto settings = tradereview::drawing::FibSettings{
        tradereview::drawing::FibLevelsConfig{{0.5}, ""},
        tradereview::drawing::FibLevelsConfig{{1.0}, ""},
    };
    const auto fib_ext = tradereview::drawing::create_drawing_spec(
        tradereview::drawing::DrawingType::FibExtension,
        {point(100, 100.0), point(200, 120.0), point(300, 110.0)},
        settings);
    const auto preview = tradereview::drawing::normalize_drawing_spec(
        tradereview::drawing::DrawingType::Line,
        {point(100, 100.0), point(200, 120.0)});

    const auto geometry = tradereview::chart::rendering::build_drawing_geometry(
        window,
        mapper,
        {0.0, 2.0},
        {-1.0F, 1.0F, -1.0F, 1.0F},
        {fib_ext},
        preview);

    tradereview::core::assert_equal(geometry.vertices.size(), std::size_t{8}, "fib extension plus preview vertex count");
    tradereview::core::assert_true(
        geometry.vertices.back().alpha < 1.0F,
        "preview vertices are translucent");
}

struct RegisterDrawingLayerGeometryTests {
    RegisterDrawingLayerGeometryTests()
    {
        tradereview::tests::register_test(
            "drawing layer maps line timestamp price to pane coordinates",
            test_drawing_layer_maps_line_timestamp_price_to_pane_coordinates);
        tradereview::tests::register_test(
            "drawing layer builds horizontal vertical and fib geometry",
            test_drawing_layer_builds_horizontal_vertical_and_fib_geometry);
        tradereview::tests::register_test(
            "drawing layer builds fib extension and preview geometry",
            test_drawing_layer_builds_fib_extension_and_preview_geometry);
    }
};

const RegisterDrawingLayerGeometryTests register_drawing_layer_geometry_tests;

} // namespace
