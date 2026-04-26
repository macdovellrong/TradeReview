#include "tradereview/core/Assertions.h"
#include "tradereview/drawing/DrawingSession.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::drawing::DrawingPoint point(std::int64_t timestamp_ns, double price)
{
    return tradereview::drawing::DrawingPoint{timestamp_ns, price};
}

void test_line_session_completes_after_two_points()
{
    auto session = tradereview::drawing::DrawingSession::for_type(
        tradereview::drawing::DrawingType::Line,
        tradereview::drawing::default_fib_settings());

    tradereview::core::assert_true(!session.add_point(point(100, 100.0)).has_value(), "first line point does not complete");
    const auto spec = session.add_point(point(200, 110.0));

    tradereview::core::assert_true(spec.has_value(), "second line point completes");
    tradereview::core::assert_equal(spec->type, tradereview::drawing::DrawingType::Line, "line spec type");
    tradereview::core::assert_equal(spec->points.size(), std::size_t{2}, "line point count");
    tradereview::core::assert_true(!spec->fib_snapshot.has_value(), "line has no fib snapshot");
}

void test_hline_and_vline_complete_after_one_point()
{
    auto hline = tradereview::drawing::DrawingSession::for_type(
        tradereview::drawing::DrawingType::HorizontalLine,
        tradereview::drawing::default_fib_settings());
    auto vline = tradereview::drawing::DrawingSession::for_type(
        tradereview::drawing::DrawingType::VerticalLine,
        tradereview::drawing::default_fib_settings());

    tradereview::core::assert_true(hline.add_point(point(100, 100.0)).has_value(), "hline completes after one point");
    tradereview::core::assert_true(vline.add_point(point(100, 100.0)).has_value(), "vline completes after one point");
}

void test_fib_extension_preview_switches_from_line_to_projection()
{
    auto session = tradereview::drawing::DrawingSession::for_type(
        tradereview::drawing::DrawingType::FibExtension,
        tradereview::drawing::FibSettings{
            tradereview::drawing::FibLevelsConfig{{0.5}, ""},
            tradereview::drawing::FibLevelsConfig{{1.0, 1.618}, ""},
        });

    session.add_point(point(100, 100.0));
    const auto first_preview = session.build_preview(point(200, 120.0));
    tradereview::core::assert_true(first_preview.has_value(), "fib extension first preview exists");
    tradereview::core::assert_equal(
        first_preview->type,
        tradereview::drawing::DrawingType::Line,
        "fib extension first preview is a line");

    session.add_point(point(200, 120.0));
    const auto second_preview = session.build_preview(point(300, 110.0));
    tradereview::core::assert_true(second_preview.has_value(), "fib extension second preview exists");
    tradereview::core::assert_equal(
        second_preview->type,
        tradereview::drawing::DrawingType::FibExtension,
        "fib extension second preview is projection");
    tradereview::core::assert_equal(second_preview->points.size(), std::size_t{3}, "fib extension preview point count");
    tradereview::core::assert_true(second_preview->fib_snapshot.has_value(), "preview carries fib snapshot");
}

void test_fib_session_snapshots_levels_at_activation()
{
    auto settings = tradereview::drawing::FibSettings{
        tradereview::drawing::FibLevelsConfig{{0.5}, "0.618"},
        tradereview::drawing::FibLevelsConfig{{1.0}, "1.618"},
    };
    auto session = tradereview::drawing::DrawingSession::for_type(
        tradereview::drawing::DrawingType::FibRetracement,
        settings);
    settings.retracement.enabled_levels = {0.236};
    settings.retracement.custom_levels_text.clear();

    session.add_point(point(100, 100.0));
    const auto spec = session.add_point(point(200, 120.0));

    tradereview::core::assert_true(spec.has_value(), "fib completes");
    tradereview::core::assert_true(spec->fib_snapshot.has_value(), "fib snapshot exists");
    tradereview::core::assert_equal(spec->fib_snapshot->levels.size(), std::size_t{2}, "snapshot level count");
    tradereview::core::assert_near(spec->fib_snapshot->levels[0], 0.5, 0.000001, "snapshot first level");
    tradereview::core::assert_near(spec->fib_snapshot->levels[1], 0.618, 0.000001, "snapshot custom level");
}

struct RegisterDrawingSessionTests {
    RegisterDrawingSessionTests()
    {
        tradereview::tests::register_test(
            "line drawing session completes after two points",
            test_line_session_completes_after_two_points);
        tradereview::tests::register_test(
            "hline and vline drawing sessions complete after one point",
            test_hline_and_vline_complete_after_one_point);
        tradereview::tests::register_test(
            "fib extension preview switches from line to projection",
            test_fib_extension_preview_switches_from_line_to_projection);
        tradereview::tests::register_test(
            "fib drawing session snapshots levels at activation",
            test_fib_session_snapshots_levels_at_activation);
    }
};

const RegisterDrawingSessionTests register_drawing_session_tests;

} // namespace
