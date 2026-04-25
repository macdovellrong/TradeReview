#include "tradereview/core/Assertions.h"
#include "tradereview/core/Period.h"

#include <cstdint>
#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_period_parses_minute_and_hour_strings()
{
    tradereview::core::assert_equal(
        tradereview::core::period_seconds("1min"),
        std::int64_t{60},
        "1min period seconds");
    tradereview::core::assert_equal(
        tradereview::core::period_seconds("4h"),
        std::int64_t{14400},
        "4h period seconds");
}

void test_period_keeps_month_distinct_from_minute()
{
    tradereview::core::assert_equal(
        tradereview::core::duckdb_candle_table("1min"),
        std::string{"candles_1m"},
        "minute candle table");
    tradereview::core::assert_equal(
        tradereview::core::duckdb_candle_table("1M"),
        std::string{"candles_1mo"},
        "month candle table");
}

struct RegisterPeriodTests {
    RegisterPeriodTests()
    {
        tradereview::tests::register_test(
            "period parses minute and hour strings",
            test_period_parses_minute_and_hour_strings);
        tradereview::tests::register_test(
            "period keeps month distinct from minute",
            test_period_keeps_month_distinct_from_minute);
    }
};

const RegisterPeriodTests register_period_tests;

} // namespace
