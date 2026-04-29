#include "tradereview/chart/ChartInteractionController.h"

#include "tradereview/chart/ChartIndexMapper.h"
#include "tradereview/core/Assertions.h"
#include "tradereview/core/TimeRange.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::chart::ChartIndexMapper sample_mapper()
{
    tradereview::chart::ChartIndexMapper mapper;
    std::vector<std::int64_t> timestamps;
    timestamps.reserve(100);
    for (std::int64_t index = 0; index < 100; ++index) {
        timestamps.push_back(index * 60LL * 1'000'000'000LL);
    }
    mapper.set_timestamps(std::move(timestamps));
    return mapper;
}

void test_reset_adds_right_padding_to_dense_range()
{
    tradereview::chart::ChartInteractionController controller;

    controller.reset_for_row_count(100);
    const auto range = controller.visible_dense_range();

    tradereview::core::assert_near(range.start_x, 0.0, 0.000001, "visible dense start");
    tradereview::core::assert_near(range.end_x, 119.0, 0.000001, "visible dense end includes right padding");
}

void test_pan_by_pixels_moves_visible_dense_range()
{
    tradereview::chart::ChartInteractionController controller;
    controller.set_visible_dense_range({10.0, 110.0});

    controller.pan_by_pixels(100.0, 1000);
    const auto range = controller.visible_dense_range();

    tradereview::core::assert_near(range.start_x, 0.0, 0.000001, "dragging right moves range left");
    tradereview::core::assert_near(range.end_x, 100.0, 0.000001, "dragging right keeps range span");
}

void test_reset_for_visible_time_range_preserves_buffered_view()
{
    tradereview::chart::ChartInteractionController controller;
    const auto mapper = sample_mapper();
    const auto minute = 60LL * 1'000'000'000LL;

    controller.reset_for_visible_time_range(mapper, {20LL * minute, 40LL * minute});
    const auto range = controller.visible_dense_range();

    tradereview::core::assert_near(range.start_x, 20.0, 0.000001, "buffered visible start maps into loaded rows");
    tradereview::core::assert_near(range.end_x, 40.0, 0.000001, "buffered visible end maps into loaded rows");
}

void test_reset_for_visible_time_range_adds_padding_at_last_loaded_candle()
{
    tradereview::chart::ChartInteractionController controller;
    const auto mapper = sample_mapper();
    const auto minute = 60LL * 1'000'000'000LL;

    controller.reset_for_visible_time_range(mapper, {80LL * minute, 99LL * minute});
    const auto range = controller.visible_dense_range();

    tradereview::core::assert_near(range.start_x, 80.0, 0.000001, "visible start near loaded end");
    tradereview::core::assert_near(range.end_x, 119.0, 0.000001, "last loaded candle receives right padding");
}

void test_reset_for_visible_time_range_preserves_time_beyond_loaded_rows()
{
    tradereview::chart::ChartInteractionController controller;
    const auto mapper = sample_mapper();
    const auto minute = 60LL * 1'000'000'000LL;

    controller.reset_for_visible_time_range(mapper, {80LL * minute, 150LL * minute});
    const auto range = controller.visible_dense_range();

    tradereview::core::assert_near(range.start_x, 80.0, 0.000001, "visible start near loaded end");
    tradereview::core::assert_near(range.end_x, 150.0, 0.000001, "visible end preserves extrapolated time");
}

void test_zoom_at_pixel_preserves_anchor()
{
    tradereview::chart::ChartInteractionController controller;
    controller.set_visible_dense_range({0.0, 100.0});

    controller.zoom_at_pixel(250.0, 1000, 0.5);
    const auto range = controller.visible_dense_range();

    tradereview::core::assert_near(range.start_x, 12.5, 0.000001, "zoom start");
    tradereview::core::assert_near(range.end_x, 62.5, 0.000001, "zoom end");
}

void test_wheel_delta_scale_uses_magnitude_and_pixel_delta()
{
    tradereview::chart::ChartInteractionController controller;

    tradereview::core::assert_near(
        controller.zoom_scale_from_wheel_delta(240, 0),
        0.64,
        0.000001,
        "two wheel steps zoom in twice");
    tradereview::core::assert_true(
        controller.zoom_scale_from_wheel_delta(0, -60) > 1.0,
        "negative touchpad pixel delta zooms out");
}

void test_center_on_dense_x_preserves_visible_span()
{
    tradereview::chart::ChartInteractionController controller;
    controller.set_visible_dense_range({10.0, 50.0});

    tradereview::core::assert_true(controller.center_on_dense_x(100.0), "center change is accepted");
    const auto range = controller.visible_dense_range();

    tradereview::core::assert_near(range.start_x, 80.0, 0.000001, "centered range start");
    tradereview::core::assert_near(range.end_x, 120.0, 0.000001, "centered range end");
}

void test_reload_not_requested_inside_right_padding()
{
    tradereview::chart::ChartInteractionController controller;
    const auto mapper = sample_mapper();
    const tradereview::core::TimeRange loaded{0, 99LL * 60LL * 1'000'000'000LL};
    controller.set_visible_dense_range({80.0, 119.0});

    const auto decision = controller.reload_decision(mapper, loaded);

    tradereview::core::assert_true(!decision.requested, "visual right padding does not request reload");
}

void test_reload_requested_beyond_right_padding()
{
    tradereview::chart::ChartInteractionController controller;
    const auto mapper = sample_mapper();
    const tradereview::core::TimeRange loaded{0, 99LL * 60LL * 1'000'000'000LL};
    controller.set_visible_dense_range({100.0, 130.0});

    const auto decision = controller.reload_decision(mapper, loaded);

    tradereview::core::assert_true(decision.requested, "beyond visual padding requests reload");
    tradereview::core::assert_true(decision.visible_range.end_ns > loaded.end_ns, "reload carries requested visible time");
}

struct RegisterChartInteractionTests {
    RegisterChartInteractionTests()
    {
        tradereview::tests::register_test(
            "interaction reset adds right padding to dense range",
            test_reset_adds_right_padding_to_dense_range);
        tradereview::tests::register_test(
            "interaction pan by pixels moves visible dense range",
            test_pan_by_pixels_moves_visible_dense_range);
        tradereview::tests::register_test(
            "interaction reset for visible time range preserves buffered view",
            test_reset_for_visible_time_range_preserves_buffered_view);
        tradereview::tests::register_test(
            "interaction reset for visible time range adds padding at last loaded candle",
            test_reset_for_visible_time_range_adds_padding_at_last_loaded_candle);
        tradereview::tests::register_test(
            "interaction reset for visible time range preserves time beyond loaded rows",
            test_reset_for_visible_time_range_preserves_time_beyond_loaded_rows);
        tradereview::tests::register_test(
            "interaction zoom at pixel preserves anchor",
            test_zoom_at_pixel_preserves_anchor);
        tradereview::tests::register_test(
            "interaction wheel delta scale uses magnitude and pixel delta",
            test_wheel_delta_scale_uses_magnitude_and_pixel_delta);
        tradereview::tests::register_test(
            "interaction center on dense x preserves visible span",
            test_center_on_dense_x_preserves_visible_span);
        tradereview::tests::register_test(
            "interaction reload not requested inside right padding",
            test_reload_not_requested_inside_right_padding);
        tradereview::tests::register_test(
            "interaction reload requested beyond right padding",
            test_reload_requested_beyond_right_padding);
    }
};

const RegisterChartInteractionTests register_chart_interaction_tests;

} // namespace
