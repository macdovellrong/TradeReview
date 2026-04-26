#include "tradereview/app/TimeNavigation.h"
#include "tradereview/core/Assertions.h"
#include "tradereview/data/CandleWindow.h"

#include <cstdint>
#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

constexpr std::int64_t kSecondNs = 1000LL * 1000LL * 1000LL;
constexpr std::int64_t kMinuteNs = 60LL * kSecondNs;

void test_time_navigation_normalizes_jump_to_minute()
{
    const auto input = 12LL * kMinuteNs + 59LL * kSecondNs + 123456789LL;

    const auto normalized = tradereview::app::normalize_jump_timestamp_ns(input);

    tradereview::core::assert_equal(normalized, 12LL * kMinuteNs, "jump timestamp floors to minute");
}

void test_time_navigation_clamps_to_dataset_range()
{
    const tradereview::core::TimeRange range{10LL * kMinuteNs, 20LL * kMinuteNs};

    tradereview::core::assert_equal(
        tradereview::app::clamp_jump_timestamp_ns(5LL * kMinuteNs, range),
        range.start_ns,
        "jump clamps to dataset start");
    tradereview::core::assert_equal(
        tradereview::app::clamp_jump_timestamp_ns(25LL * kMinuteNs, range),
        range.end_ns,
        "jump clamps to dataset end");
    tradereview::core::assert_equal(
        tradereview::app::clamp_jump_timestamp_ns(15LL * kMinuteNs, range),
        15LL * kMinuteNs,
        "jump inside range is preserved");
}

void test_time_navigation_resolves_right_side_chart_row()
{
    tradereview::data::CandleWindow window;
    window.timestamp_ns = {10LL * kMinuteNs, 11LL * kMinuteNs, 12LL * kMinuteNs};
    window.open = {100.0, 101.0, 102.0};
    window.high = {101.0, 102.0, 103.0};
    window.low = {99.0, 100.0, 101.0};
    window.close = {100.5, 101.5, 102.5};
    window.volume = {1.0, 1.0, 1.0};

    const auto target = tradereview::app::resolve_chart_target_row(window, 10LL * kMinuteNs + 30LL * kSecondNs);

    tradereview::core::assert_true(target.has_value(), "chart target exists");
    tradereview::core::assert_equal(target->row, std::size_t{1}, "target uses right-side bar");
    tradereview::core::assert_near(target->close, 101.5, 0.000001, "target close price");
}

void test_time_navigation_builds_centered_visible_range()
{
    const tradereview::core::TimeRange dataset{0, 24LL * 60LL * kMinuteNs};

    const auto range = tradereview::app::centered_visible_range(
        12LL * 60LL * kMinuteNs,
        dataset,
        6LL * 60LL * kMinuteNs);

    tradereview::core::assert_equal(range.start_ns, 9LL * 60LL * kMinuteNs, "centered start");
    tradereview::core::assert_equal(range.end_ns, 15LL * 60LL * kMinuteNs, "centered end");

    const auto clamped = tradereview::app::centered_visible_range(
        1LL * 60LL * kMinuteNs,
        dataset,
        6LL * 60LL * kMinuteNs);
    tradereview::core::assert_equal(clamped.start_ns, dataset.start_ns, "range clamps left edge");
    tradereview::core::assert_equal(clamped.end_ns, 6LL * 60LL * kMinuteNs, "range keeps requested width");
}

struct RegisterTimeNavigationTests {
    RegisterTimeNavigationTests()
    {
        tradereview::tests::register_test(
            "time navigation normalizes jump to minute",
            test_time_navigation_normalizes_jump_to_minute);
        tradereview::tests::register_test(
            "time navigation clamps to dataset range",
            test_time_navigation_clamps_to_dataset_range);
        tradereview::tests::register_test(
            "time navigation resolves right-side chart row",
            test_time_navigation_resolves_right_side_chart_row);
        tradereview::tests::register_test(
            "time navigation builds centered visible range",
            test_time_navigation_builds_centered_visible_range);
    }
};

const RegisterTimeNavigationTests register_time_navigation_tests;

} // namespace
