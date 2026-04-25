#include "tradereview/chart/Windowing.h"
#include "tradereview/core/Assertions.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_query_window_adds_buffer_on_both_sides()
{
    const tradereview::core::TimeRange visible{1000, 2000};
    const auto query = tradereview::chart::build_query_window(visible, 2.0);

    tradereview::core::assert_equal(query.start_ns, std::int64_t{-1000}, "query start");
    tradereview::core::assert_equal(query.end_ns, std::int64_t{4000}, "query end");
}

void test_view_inside_loaded_window_returns_true()
{
    const tradereview::core::TimeRange visible{1000, 2000};
    const tradereview::core::TimeRange loaded{0, 3000};

    tradereview::core::assert_true(
        tradereview::chart::is_view_inside_loaded_window(visible, loaded),
        "visible range should be inside loaded range");
}

void test_prefetch_returns_false_for_centered_visible_window()
{
    const tradereview::core::TimeRange visible{4000, 6000};
    const tradereview::core::TimeRange loaded{0, 10000};

    tradereview::core::assert_true(
        !tradereview::chart::should_prefetch_window(visible, loaded, 0.5),
        "centered visible range should not prefetch");
}

void test_prefetch_returns_true_near_loaded_edge()
{
    const tradereview::core::TimeRange visible{900, 2900};
    const tradereview::core::TimeRange loaded{0, 10000};

    tradereview::core::assert_true(
        tradereview::chart::should_prefetch_window(visible, loaded, 0.5),
        "visible range near loaded edge should prefetch");
}

void test_prefetch_returns_true_outside_loaded_window()
{
    const tradereview::core::TimeRange visible{-1000, 1000};
    const tradereview::core::TimeRange loaded{0, 10000};

    tradereview::core::assert_true(
        tradereview::chart::should_prefetch_window(visible, loaded, 0.5),
        "visible range outside loaded window should prefetch");
}

void test_query_window_saturates_near_int64_limits()
{
    const auto min_value = std::numeric_limits<std::int64_t>::min();
    const auto max_value = std::numeric_limits<std::int64_t>::max();
    const tradereview::core::TimeRange visible{min_value + 10, max_value - 10};
    const auto query = tradereview::chart::build_query_window(visible, 1.0);

    tradereview::core::assert_equal(query.start_ns, min_value, "saturated query start");
    tradereview::core::assert_equal(query.end_ns, max_value, "saturated query end");
}

struct RegisterWindowingTests {
    RegisterWindowingTests()
    {
        tradereview::tests::register_test(
            "query window adds buffer on both sides",
            test_query_window_adds_buffer_on_both_sides);
        tradereview::tests::register_test(
            "view inside loaded window returns true",
            test_view_inside_loaded_window_returns_true);
        tradereview::tests::register_test(
            "prefetch returns false for centered visible window",
            test_prefetch_returns_false_for_centered_visible_window);
        tradereview::tests::register_test(
            "prefetch returns true near loaded edge",
            test_prefetch_returns_true_near_loaded_edge);
        tradereview::tests::register_test(
            "prefetch returns true outside loaded window",
            test_prefetch_returns_true_outside_loaded_window);
        tradereview::tests::register_test(
            "query window saturates near int64 limits",
            test_query_window_saturates_near_int64_limits);
    }
};

const RegisterWindowingTests register_windowing_tests;

} // namespace
