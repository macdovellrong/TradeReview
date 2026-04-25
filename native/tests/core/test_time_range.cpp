#include "tradereview/core/Assertions.h"
#include "tradereview/core/TimeRange.h"

#include <cstdint>
#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_time_range_span_is_end_minus_start()
{
    const tradereview::core::TimeRange range{100, 250};

    tradereview::core::assert_equal(range.span_ns(), std::int64_t{150}, "time range span");
}

void test_time_range_normalizes_reversed_endpoints()
{
    const auto range = tradereview::core::TimeRange::normalized(250, 100);

    tradereview::core::assert_equal(range.start_ns, std::int64_t{100}, "normalized start");
    tradereview::core::assert_equal(range.end_ns, std::int64_t{250}, "normalized end");
}

struct RegisterTimeRangeTests {
    RegisterTimeRangeTests()
    {
        tradereview::tests::register_test(
            "time range span is end minus start",
            test_time_range_span_is_end_minus_start);
        tradereview::tests::register_test(
            "time range normalizes reversed endpoints",
            test_time_range_normalizes_reversed_endpoints);
    }
};

const RegisterTimeRangeTests register_time_range_tests;

} // namespace
