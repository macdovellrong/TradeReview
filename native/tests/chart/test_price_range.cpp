#include "tradereview/chart/PriceRange.h"

#include "tradereview/core/Assertions.h"

#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_normalize_price_range_orders_bounds()
{
    const auto range = tradereview::chart::normalize_price_range(20.0, 10.0);

    tradereview::core::assert_true(range.has_value(), "reversed finite range is accepted");
    tradereview::core::assert_near(range->first, 10.0, 0.000001, "normalized minimum");
    tradereview::core::assert_near(range->second, 20.0, 0.000001, "normalized maximum");
}

void test_normalize_price_range_expands_flat_bounds()
{
    const auto range = tradereview::chart::normalize_price_range(10.0, 10.0);

    tradereview::core::assert_true(range.has_value(), "flat finite range is accepted");
    tradereview::core::assert_near(range->first, 10.0, 0.000001, "flat minimum remains anchored");
    tradereview::core::assert_true(range->second > range->first, "flat maximum is expanded");
}

void test_zoom_price_range_anchors_to_mouse_price()
{
    const auto range = tradereview::chart::zoom_price_range({10.0, 20.0}, 12.0, 0.5);

    tradereview::core::assert_near(range.first, 11.0, 0.000001, "zoomed minimum keeps anchor relation");
    tradereview::core::assert_near(range.second, 16.0, 0.000001, "zoomed maximum keeps anchor relation");
}

void test_pan_price_range_shifts_bounds_without_resizing()
{
    const auto range = tradereview::chart::pan_price_range({10.0, 20.0}, -2.5);

    tradereview::core::assert_near(range.first, 7.5, 0.000001, "panned minimum");
    tradereview::core::assert_near(range.second, 17.5, 0.000001, "panned maximum");
}

void test_price_delta_for_pixel_pan_scales_by_visible_height()
{
    const auto delta = tradereview::chart::price_delta_for_pixel_pan({10.0, 20.0}, 25.0, 100.0);

    tradereview::core::assert_near(delta, 2.5, 0.000001, "price delta from pixel pan");
}

struct RegisterPriceRangeTests {
    RegisterPriceRangeTests()
    {
        tradereview::tests::register_test(
            "price range normalization orders bounds",
            test_normalize_price_range_orders_bounds);
        tradereview::tests::register_test(
            "price range normalization expands flat bounds",
            test_normalize_price_range_expands_flat_bounds);
        tradereview::tests::register_test(
            "price range zoom anchors to mouse price",
            test_zoom_price_range_anchors_to_mouse_price);
        tradereview::tests::register_test(
            "price range pan shifts bounds without resizing",
            test_pan_price_range_shifts_bounds_without_resizing);
        tradereview::tests::register_test(
            "price delta for pixel pan scales by visible height",
            test_price_delta_for_pixel_pan_scales_by_visible_height);
    }
};

const RegisterPriceRangeTests register_price_range_tests;

} // namespace
