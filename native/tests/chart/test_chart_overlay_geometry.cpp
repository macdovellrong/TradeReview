#include "tradereview/chart/ChartOverlayGeometry.h"
#include "tradereview/chart/PaneLayout.h"
#include "tradereview/core/Assertions.h"

#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_overlay_geometry_maps_dense_x_to_widget_pixels()
{
    const auto x = tradereview::chart::widget_x_for_dense_x({10.0, 30.0}, 800, 20.0);

    tradereview::core::assert_near(x, 400.0, 0.000001, "dense x maps to widget x");
}

void test_overlay_geometry_maps_price_to_price_pane_pixels()
{
    const auto layout = tradereview::chart::build_pane_layout(true);
    const auto y = tradereview::chart::widget_y_for_price(layout.price, 600, 100.0, 200.0, 150.0);

    tradereview::core::assert_true(y.has_value(), "price y is available");
    tradereview::core::assert_near(*y, 174.0, 0.000001, "price maps inside price pane");
}

void test_overlay_geometry_formats_fib_ratios_as_percentages()
{
    tradereview::core::assert_equal(
        tradereview::chart::format_fib_ratio_label(0.0),
        std::string{"0%"},
        "zero fib ratio");
    tradereview::core::assert_equal(
        tradereview::chart::format_fib_ratio_label(0.5),
        std::string{"50%"},
        "half fib ratio");
    tradereview::core::assert_equal(
        tradereview::chart::format_fib_ratio_label(0.618),
        std::string{"61.8%"},
        "decimal fib ratio");
    tradereview::core::assert_equal(
        tradereview::chart::format_fib_ratio_label(1.618),
        std::string{"161.8%"},
        "extension fib ratio");
}

struct RegisterChartOverlayGeometryTests {
    RegisterChartOverlayGeometryTests()
    {
        tradereview::tests::register_test(
            "overlay geometry maps dense x to widget pixels",
            test_overlay_geometry_maps_dense_x_to_widget_pixels);
        tradereview::tests::register_test(
            "overlay geometry maps price to price pane pixels",
            test_overlay_geometry_maps_price_to_price_pane_pixels);
        tradereview::tests::register_test(
            "overlay geometry formats fib ratios as percentages",
            test_overlay_geometry_formats_fib_ratios_as_percentages);
    }
};

const RegisterChartOverlayGeometryTests register_chart_overlay_geometry_tests;

} // namespace
