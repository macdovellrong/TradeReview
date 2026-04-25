#include "tradereview/chart/LodResolver.h"
#include "tradereview/core/Assertions.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

constexpr std::int64_t kSecondNs = 1'000'000'000LL;

tradereview::core::TimeRange seconds_range(std::int64_t seconds)
{
    return tradereview::core::TimeRange{0, seconds * kSecondNs};
}

void test_lod_keeps_requested_period_when_density_fits()
{
    const std::vector<std::string> periods{"1min", "5min", "1h", "1D"};
    const auto chosen = tradereview::chart::choose_lod_period(
        "1min",
        seconds_range(6 * 60 * 60),
        1600,
        periods);

    tradereview::core::assert_equal(chosen, std::string{"1min"}, "density-fit lod period");
}

void test_lod_chooses_coarser_period_for_multi_year_view()
{
    const std::vector<std::string> periods{"30s", "1min", "5min", "1h", "4h", "1D"};
    const auto chosen = tradereview::chart::choose_lod_period(
        "30s",
        seconds_range(5LL * 365 * 24 * 60 * 60),
        1600,
        periods);

    tradereview::core::assert_equal(chosen, std::string{"1D"}, "multi-year lod period");
}

void test_lod_never_chooses_finer_period_than_requested()
{
    const std::vector<std::string> periods{"5min", "1h", "1D"};
    const auto chosen = tradereview::chart::choose_lod_period(
        "1h",
        seconds_range(10 * 24 * 60 * 60),
        1600,
        periods);

    tradereview::core::assert_equal(chosen, std::string{"1h"}, "not-finer lod period");
}

struct RegisterLodResolverTests {
    RegisterLodResolverTests()
    {
        tradereview::tests::register_test(
            "lod keeps requested period when density fits",
            test_lod_keeps_requested_period_when_density_fits);
        tradereview::tests::register_test(
            "lod chooses coarser period for multi-year view",
            test_lod_chooses_coarser_period_for_multi_year_view);
        tradereview::tests::register_test(
            "lod never chooses finer period than requested",
            test_lod_never_chooses_finer_period_than_requested);
    }
};

const RegisterLodResolverTests register_lod_resolver_tests;

} // namespace
