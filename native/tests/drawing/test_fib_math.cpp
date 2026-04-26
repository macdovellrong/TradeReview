#include "tradereview/core/Assertions.h"
#include "tradereview/drawing/FibMath.h"
#include "tradereview/drawing/FibSettings.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_retracement_levels_use_requested_ratios()
{
    const auto rows = tradereview::drawing::build_retracement_levels(100.0, 120.0, {0.5, 0.618, 0.786});

    tradereview::core::assert_equal(rows.size(), std::size_t{3}, "retracement level count");
    tradereview::core::assert_near(rows[0].ratio, 0.5, 0.000001, "first retracement ratio");
    tradereview::core::assert_near(rows[0].price, 110.0, 0.000001, "first retracement price");
    tradereview::core::assert_near(rows[1].price, 107.64, 0.000001, "second retracement price");
    tradereview::core::assert_near(rows[2].price, 104.28, 0.000001, "third retracement price");
}

void test_extension_levels_project_upward()
{
    const auto rows = tradereview::drawing::build_extension_levels(100.0, 120.0, 110.0, {1.0, 1.618});

    tradereview::core::assert_equal(rows.size(), std::size_t{2}, "extension level count");
    tradereview::core::assert_near(rows[0].price, 130.0, 0.000001, "first upward extension price");
    tradereview::core::assert_near(rows[1].price, 142.36, 0.000001, "second upward extension price");
}

void test_extension_levels_project_downward()
{
    const auto rows = tradereview::drawing::build_extension_levels(120.0, 100.0, 110.0, {1.0, 1.618});

    tradereview::core::assert_near(rows[0].price, 90.0, 0.000001, "first downward extension price");
    tradereview::core::assert_near(rows[1].price, 77.64, 0.000001, "second downward extension price");
}

void test_default_fib_settings_match_python_presets()
{
    const auto settings = tradereview::drawing::default_fib_settings();

    tradereview::core::assert_equal(settings.retracement.enabled_levels.size(), std::size_t{7}, "retracement preset count");
    tradereview::core::assert_near(settings.retracement.enabled_levels[0], 0.236, 0.000001, "first retracement preset");
    tradereview::core::assert_near(settings.retracement.enabled_levels[6], 0.8, 0.000001, "last retracement preset");
    tradereview::core::assert_equal(settings.extension.enabled_levels.size(), std::size_t{5}, "extension preset count");
    tradereview::core::assert_near(settings.extension.enabled_levels[0], 0.618, 0.000001, "first extension preset");
    tradereview::core::assert_near(settings.extension.enabled_levels[4], 2.0, 0.000001, "last extension preset");
}

void test_merge_fib_levels_sorts_deduplicates_and_rejects_invalid_tokens()
{
    const auto levels = tradereview::drawing::merge_fib_levels({0.618, 0.5}, "0.618, 0.786, 0.8");

    tradereview::core::assert_equal(levels.size(), std::size_t{4}, "merged level count");
    tradereview::core::assert_near(levels[0], 0.5, 0.000001, "first merged level");
    tradereview::core::assert_near(levels[1], 0.618, 0.000001, "second merged level");
    tradereview::core::assert_near(levels[2], 0.786, 0.000001, "third merged level");
    tradereview::core::assert_near(levels[3], 0.8, 0.000001, "fourth merged level");

    try {
        (void)tradereview::drawing::merge_fib_levels({0.5}, "0.7,abc");
    } catch (const std::invalid_argument&) {
        return;
    }

    throw std::runtime_error("invalid custom Fibonacci level should throw");
}

struct RegisterFibMathTests {
    RegisterFibMathTests()
    {
        tradereview::tests::register_test(
            "fib retracement levels use requested ratios",
            test_retracement_levels_use_requested_ratios);
        tradereview::tests::register_test(
            "fib extension levels project upward",
            test_extension_levels_project_upward);
        tradereview::tests::register_test(
            "fib extension levels project downward",
            test_extension_levels_project_downward);
        tradereview::tests::register_test(
            "default fib settings match python presets",
            test_default_fib_settings_match_python_presets);
        tradereview::tests::register_test(
            "merge fib levels sorts deduplicates and rejects invalid tokens",
            test_merge_fib_levels_sorts_deduplicates_and_rejects_invalid_tokens);
    }
};

const RegisterFibMathTests register_fib_math_tests;

} // namespace
