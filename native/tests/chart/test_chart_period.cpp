#include "tradereview/chart/ChartPeriod.h"
#include "tradereview/core/Assertions.h"

#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_chart_period_normalizes_toolbar_minutes()
{
    tradereview::core::assert_equal(
        tradereview::chart::canonical_chart_period("1m"),
        std::string{"1min"},
        "1m toolbar period");
    tradereview::core::assert_equal(
        tradereview::chart::canonical_chart_period("5m"),
        std::string{"5min"},
        "5m toolbar period");
    tradereview::core::assert_equal(
        tradereview::chart::canonical_chart_period("90m"),
        std::string{"90min"},
        "90m toolbar period");
}

void test_chart_period_preserves_non_minute_periods()
{
    tradereview::core::assert_equal(
        tradereview::chart::canonical_chart_period("30s"),
        std::string{"30s"},
        "seconds period");
    tradereview::core::assert_equal(
        tradereview::chart::canonical_chart_period("1h"),
        std::string{"1h"},
        "hour period");
    tradereview::core::assert_equal(
        tradereview::chart::canonical_chart_period("1M"),
        std::string{"1M"},
        "month period");
}

void test_chart_period_maps_canonical_minutes_back_to_toolbar()
{
    tradereview::core::assert_equal(
        tradereview::chart::toolbar_chart_period("1min"),
        std::string{"1m"},
        "1min toolbar label");
    tradereview::core::assert_equal(
        tradereview::chart::toolbar_chart_period("15min"),
        std::string{"15m"},
        "15min toolbar label");
}

struct RegisterChartPeriodTests {
    RegisterChartPeriodTests()
    {
        tradereview::tests::register_test(
            "chart period normalizes toolbar minutes",
            test_chart_period_normalizes_toolbar_minutes);
        tradereview::tests::register_test(
            "chart period preserves non-minute periods",
            test_chart_period_preserves_non_minute_periods);
        tradereview::tests::register_test(
            "chart period maps canonical minutes back to toolbar",
            test_chart_period_maps_canonical_minutes_back_to_toolbar);
    }
};

const RegisterChartPeriodTests register_chart_period_tests;

} // namespace
