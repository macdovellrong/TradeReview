#include "tradereview/core/Assertions.h"
#include "tradereview/core/Period.h"

#include <cstdint>
#include <functional>
#include <stdexcept>
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

void test_period_accepts_uppercase_day_unit()
{
    tradereview::core::assert_equal(
        tradereview::core::period_seconds("1D"),
        std::int64_t{86400},
        "1D period seconds");
    tradereview::core::assert_equal(
        tradereview::core::duckdb_candle_table("1D"),
        std::string{"candles_1D"},
        "1D candle table");
}

void test_period_accepts_uppercase_week_unit()
{
    tradereview::core::assert_equal(
        tradereview::core::period_seconds("1W"),
        std::int64_t{604800},
        "1W period seconds");
    tradereview::core::assert_equal(
        tradereview::core::duckdb_candle_table("1W"),
        std::string{"candles_1W"},
        "1W candle table");
}

void test_period_rejects_zero_value()
{
    tradereview::core::assert_true(
        !tradereview::core::try_period_seconds("0min").has_value(),
        "0min should not parse");

    bool threw = false;
    try {
        static_cast<void>(tradereview::core::period_seconds("0min"));
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    tradereview::core::assert_true(threw, "0min should throw invalid_argument");
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
            "period accepts uppercase day unit",
            test_period_accepts_uppercase_day_unit);
        tradereview::tests::register_test(
            "period accepts uppercase week unit",
            test_period_accepts_uppercase_week_unit);
        tradereview::tests::register_test(
            "period rejects zero value",
            test_period_rejects_zero_value);
        tradereview::tests::register_test(
            "period keeps month distinct from minute",
            test_period_keeps_month_distinct_from_minute);
    }
};

const RegisterPeriodTests register_period_tests;

} // namespace
