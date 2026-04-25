#include "tradereview/chart/Windowing.h"
#include "tradereview/core/Assertions.h"

#include <cstdint>
#include <functional>
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

struct RegisterWindowingTests {
    RegisterWindowingTests()
    {
        tradereview::tests::register_test(
            "query window adds buffer on both sides",
            test_query_window_adds_buffer_on_both_sides);
        tradereview::tests::register_test(
            "view inside loaded window returns true",
            test_view_inside_loaded_window_returns_true);
    }
};

const RegisterWindowingTests register_windowing_tests;

} // namespace
