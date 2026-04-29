#include "tradereview/chart/ChartTimeFormat.h"
#include "tradereview/core/Assertions.h"

#include <cstdint>
#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

constexpr std::int64_t kTimestampNs = 1'704'164'645'000'000'000LL;

void test_axis_timestamp_uses_second_precision_for_minute_period()
{
    const auto label = tradereview::chart::format_axis_timestamp_label(kTimestampNs, "1min").toStdString();

    tradereview::core::assert_equal(
        label,
        std::string{"2024-01-02 03:04:05"},
        "minute period axis timestamp");
}

void test_axis_timestamp_uses_second_precision_for_seconds_period()
{
    const auto label = tradereview::chart::format_axis_timestamp_label(kTimestampNs, "30s").toStdString();

    tradereview::core::assert_equal(
        label,
        std::string{"2024-01-02 03:04:05"},
        "seconds period axis timestamp");
}

void test_axis_timestamp_uses_minute_precision_for_intraday_period()
{
    const auto label = tradereview::chart::format_axis_timestamp_label(kTimestampNs, "5min").toStdString();

    tradereview::core::assert_equal(
        label,
        std::string{"2024-01-02 03:04"},
        "intraday period axis timestamp");
}

void test_axis_timestamp_uses_date_precision_for_daily_period()
{
    const auto label = tradereview::chart::format_axis_timestamp_label(kTimestampNs, "1D").toStdString();

    tradereview::core::assert_equal(label, std::string{"2024-01-02"}, "daily period axis timestamp");
}

void test_axis_timestamp_falls_back_to_second_precision_for_unknown_period()
{
    const auto label = tradereview::chart::format_axis_timestamp_label(kTimestampNs, "unknown").toStdString();

    tradereview::core::assert_equal(
        label,
        std::string{"2024-01-02 03:04:05"},
        "unknown period axis timestamp");
}

struct RegisterChartTimeFormatTests {
    RegisterChartTimeFormatTests()
    {
        tradereview::tests::register_test(
            "chart time format minute period uses seconds",
            test_axis_timestamp_uses_second_precision_for_minute_period);
        tradereview::tests::register_test(
            "chart time format seconds period uses seconds",
            test_axis_timestamp_uses_second_precision_for_seconds_period);
        tradereview::tests::register_test(
            "chart time format intraday period uses minutes",
            test_axis_timestamp_uses_minute_precision_for_intraday_period);
        tradereview::tests::register_test(
            "chart time format daily period uses date",
            test_axis_timestamp_uses_date_precision_for_daily_period);
        tradereview::tests::register_test(
            "chart time format unknown period uses seconds",
            test_axis_timestamp_falls_back_to_second_precision_for_unknown_period);
    }
};

const RegisterChartTimeFormatTests register_chart_time_format_tests;

} // namespace
