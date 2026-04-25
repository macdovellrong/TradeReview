#include "tradereview/core/Assertions.h"
#include "tradereview/data/CandleWindow.h"

#include <cstddef>
#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_candle_window_reports_row_count()
{
    tradereview::data::CandleWindow window;
    window.timestamp_ns = {100, 200, 300};
    window.open = {1.0, 2.0, 3.0};
    window.high = {1.5, 2.5, 3.5};
    window.low = {0.5, 1.5, 2.5};
    window.close = {1.25, 2.25, 3.25};
    window.volume = {10.0, 20.0, 30.0};

    tradereview::core::assert_equal(window.row_count(), std::size_t{3}, "candle window row count");
    tradereview::core::assert_true(window.has_consistent_columns(), "candle window columns are consistent");
}

struct RegisterCandleWindowTests {
    RegisterCandleWindowTests()
    {
        tradereview::tests::register_test(
            "candle window reports row count",
            test_candle_window_reports_row_count);
    }
};

const RegisterCandleWindowTests register_candle_window_tests;

} // namespace
