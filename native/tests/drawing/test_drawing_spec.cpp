#include "tradereview/core/Assertions.h"
#include "tradereview/drawing/DrawingSpec.h"
#include "tradereview/drawing/FibSettings.h"

#include <cstdint>
#include <functional>
#include <stdexcept>
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

void test_drawing_point_uses_canonical_timestamp_price_coordinates()
{
    const auto p = point(1'775'779'400'000'000'000LL, 1935.25);

    tradereview::core::assert_equal(
        p.timestamp_ns,
        std::int64_t{1'775'779'400'000'000'000LL},
        "point timestamp ns");
    tradereview::core::assert_near(p.price, 1935.25, 0.000001, "point price");
}

void test_normalize_drawing_specs_clamps_points_to_tool_requirements()
{
    const std::vector<tradereview::drawing::DrawingPoint> points{
        point(100, 10.0),
        point(200, 20.0),
        point(300, 30.0),
        point(400, 40.0),
    };

    const auto hline = tradereview::drawing::normalize_drawing_spec(tradereview::drawing::DrawingType::HorizontalLine, points);
    const auto vline = tradereview::drawing::normalize_drawing_spec(tradereview::drawing::DrawingType::VerticalLine, points);
    const auto line = tradereview::drawing::normalize_drawing_spec(tradereview::drawing::DrawingType::Line, points);
    const auto fib = tradereview::drawing::normalize_drawing_spec(tradereview::drawing::DrawingType::FibRetracement, points);
    const auto fib_ext = tradereview::drawing::normalize_drawing_spec(tradereview::drawing::DrawingType::FibExtension, points);

    tradereview::core::assert_equal(hline.points.size(), std::size_t{1}, "hline point count");
    tradereview::core::assert_equal(vline.points.size(), std::size_t{1}, "vline point count");
    tradereview::core::assert_equal(line.points.size(), std::size_t{2}, "line point count");
    tradereview::core::assert_equal(fib.points.size(), std::size_t{2}, "fib point count");
    tradereview::core::assert_equal(fib_ext.points.size(), std::size_t{3}, "fib extension point count");
}

void test_normalize_drawing_spec_rejects_missing_required_points()
{
    try {
        (void)tradereview::drawing::normalize_drawing_spec(
            tradereview::drawing::DrawingType::FibExtension,
            {point(100, 10.0), point(200, 20.0)});
    } catch (const std::invalid_argument&) {
        return;
    }

    throw std::runtime_error("fib extension with two points should throw");
}

void test_fib_specs_snapshot_effective_levels_at_creation()
{
    auto settings = tradereview::drawing::FibSettings{
        tradereview::drawing::FibLevelsConfig{{0.5, 0.618}, "0.786"},
        tradereview::drawing::FibLevelsConfig{{1.0, 1.618}, "2.0"},
    };

    const auto fib = tradereview::drawing::create_drawing_spec(
        tradereview::drawing::DrawingType::FibRetracement,
        {point(100, 100.0), point(200, 120.0)},
        settings);
    const auto fib_ext = tradereview::drawing::create_drawing_spec(
        tradereview::drawing::DrawingType::FibExtension,
        {point(100, 100.0), point(200, 120.0), point(300, 110.0)},
        settings);

    settings.retracement.enabled_levels = {0.236};
    settings.retracement.custom_levels_text = "";
    settings.extension.enabled_levels = {0.618};
    settings.extension.custom_levels_text = "";

    tradereview::core::assert_true(fib.fib_snapshot.has_value(), "fib has snapshot");
    tradereview::core::assert_equal(fib.fib_snapshot->levels.size(), std::size_t{3}, "fib snapshot level count");
    tradereview::core::assert_near(fib.fib_snapshot->levels[2], 0.786, 0.000001, "fib snapshot custom level");
    tradereview::core::assert_true(fib_ext.fib_snapshot.has_value(), "fib extension has snapshot");
    tradereview::core::assert_equal(fib_ext.fib_snapshot->levels.size(), std::size_t{3}, "fib extension snapshot level count");
    tradereview::core::assert_near(fib_ext.fib_snapshot->levels[2], 2.0, 0.000001, "fib extension snapshot custom level");
}

void test_non_fib_specs_do_not_snapshot_fib_settings()
{
    const auto line = tradereview::drawing::create_drawing_spec(
        tradereview::drawing::DrawingType::Line,
        {point(100, 100.0), point(200, 120.0)},
        tradereview::drawing::default_fib_settings());

    tradereview::core::assert_true(!line.fib_snapshot.has_value(), "line should not have fib snapshot");
}

struct RegisterDrawingSpecTests {
    RegisterDrawingSpecTests()
    {
        tradereview::tests::register_test(
            "drawing point uses canonical timestamp price coordinates",
            test_drawing_point_uses_canonical_timestamp_price_coordinates);
        tradereview::tests::register_test(
            "normalize drawing specs clamps points to tool requirements",
            test_normalize_drawing_specs_clamps_points_to_tool_requirements);
        tradereview::tests::register_test(
            "normalize drawing spec rejects missing required points",
            test_normalize_drawing_spec_rejects_missing_required_points);
        tradereview::tests::register_test(
            "fib specs snapshot effective levels at creation",
            test_fib_specs_snapshot_effective_levels_at_creation);
        tradereview::tests::register_test(
            "non fib specs do not snapshot fib settings",
            test_non_fib_specs_do_not_snapshot_fib_settings);
    }
};

const RegisterDrawingSpecTests register_drawing_spec_tests;

} // namespace
