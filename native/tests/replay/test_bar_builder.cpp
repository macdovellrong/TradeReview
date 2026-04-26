#include "tradereview/core/Assertions.h"
#include "tradereview/replay/BarBuilder.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

constexpr std::int64_t kSecondNs = 1000LL * 1000LL * 1000LL;
constexpr std::int64_t kMinuteNs = 60LL * kSecondNs;

void test_bar_builder_updates_active_bar_from_ticks()
{
    tradereview::replay::BarBuilder builder("1min", 10);

    builder.add_tick(5LL * kSecondNs, 100.0, 1.0);
    builder.add_tick(35LL * kSecondNs, 105.0, 2.5);

    const auto window = builder.to_window(7, 3, {0, kMinuteNs});

    tradereview::core::assert_equal(window.chart_id, std::uint64_t{7}, "chart id");
    tradereview::core::assert_equal(window.generation, std::uint64_t{3}, "generation");
    tradereview::core::assert_equal(window.requested_period, std::string{"1min"}, "requested period");
    tradereview::core::assert_equal(window.row_count(), std::size_t{1}, "active row count");
    tradereview::core::assert_equal(window.timestamp_ns.front(), std::int64_t{0}, "active bucket start");
    tradereview::core::assert_near(window.open.front(), 100.0, 0.000001, "active open");
    tradereview::core::assert_near(window.high.front(), 105.0, 0.000001, "active high");
    tradereview::core::assert_near(window.low.front(), 100.0, 0.000001, "active low");
    tradereview::core::assert_near(window.close.front(), 105.0, 0.000001, "active close");
    tradereview::core::assert_near(window.volume.front(), 3.5, 0.000001, "active volume");
}

void test_bar_builder_finalizes_gaps_and_trims_tail()
{
    tradereview::replay::BarBuilder builder("1min", 3);

    builder.add_tick(5LL * kSecondNs, 100.0, 1.0);
    builder.add_tick(3LL * kMinuteNs + 5LL * kSecondNs, 103.0, 4.0);
    builder.add_tick(4LL * kMinuteNs + 5LL * kSecondNs, 104.0, 5.0);

    const auto window = builder.to_window(1, 1, {0, 5LL * kMinuteNs});

    tradereview::core::assert_equal(window.row_count(), std::size_t{3}, "trimmed row count");
    tradereview::core::assert_equal(window.timestamp_ns[0], 2LL * kMinuteNs, "tail keeps second gap");
    tradereview::core::assert_true(std::isnan(window.open[0]), "gap open is NaN");
    tradereview::core::assert_equal(window.timestamp_ns[1], 3LL * kMinuteNs, "completed tick bucket");
    tradereview::core::assert_near(window.close[1], 103.0, 0.000001, "completed close");
    tradereview::core::assert_equal(window.timestamp_ns[2], 4LL * kMinuteNs, "active tick bucket");
    tradereview::core::assert_near(window.close[2], 104.0, 0.000001, "active close");
}

struct RegisterBarBuilderTests {
    RegisterBarBuilderTests()
    {
        tradereview::tests::register_test(
            "bar builder updates active bar from ticks",
            test_bar_builder_updates_active_bar_from_ticks);
        tradereview::tests::register_test(
            "bar builder finalizes gaps and trims tail",
            test_bar_builder_finalizes_gaps_and_trims_tail);
    }
};

const RegisterBarBuilderTests register_bar_builder_tests;

} // namespace
