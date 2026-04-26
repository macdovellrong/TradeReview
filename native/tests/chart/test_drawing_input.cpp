#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/chart/DrawingInput.h"
#include "tradereview/chart/PaneLayout.h"
#include "tradereview/core/Assertions.h"

#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::chart::ChartIndexMapper sample_mapper()
{
    tradereview::chart::ChartIndexMapper mapper;
    mapper.set_timestamps({100, 200, 300});
    return mapper;
}

void test_drawing_input_rejects_indicator_pane_clicks()
{
    const auto mapper = sample_mapper();
    const auto layout = tradereview::chart::build_pane_layout(true);

    const auto point = tradereview::chart::drawing_point_from_widget_position(
        mapper,
        {0.0, 2.0},
        layout.price,
        1000,
        1000,
        {500.0, 900.0},
        90.0,
        130.0);

    tradereview::core::assert_true(!point.has_value(), "indicator pane click is rejected");
}

void test_drawing_input_maps_price_pane_click_to_canonical_point()
{
    const auto mapper = sample_mapper();
    const auto layout = tradereview::chart::build_pane_layout(true);

    const auto point = tradereview::chart::drawing_point_from_widget_position(
        mapper,
        {0.0, 2.0},
        layout.price,
        1000,
        1000,
        {500.0, 290.0},
        90.0,
        130.0);

    tradereview::core::assert_true(point.has_value(), "price pane click maps to drawing point");
    tradereview::core::assert_equal(point->timestamp_ns, std::int64_t{200}, "mapped timestamp");
    tradereview::core::assert_near(point->price, 110.0, 0.000001, "mapped price");
}

struct RegisterDrawingInputTests {
    RegisterDrawingInputTests()
    {
        tradereview::tests::register_test(
            "drawing input rejects indicator pane clicks",
            test_drawing_input_rejects_indicator_pane_clicks);
        tradereview::tests::register_test(
            "drawing input maps price pane click to canonical point",
            test_drawing_input_maps_price_pane_click_to_canonical_point);
    }
};

const RegisterDrawingInputTests register_drawing_input_tests;

} // namespace
