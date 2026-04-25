#include "tradereview/core/Assertions.h"

#include <functional>
#include <stdexcept>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_assert_equal_accepts_equal_integers()
{
    tradereview::core::assert_equal(3, 3, "integer equality");
}

void test_assert_true_throws_on_false()
{
    bool threw = false;
    try {
        tradereview::core::assert_true(false, "false condition");
    } catch (const std::runtime_error&) {
        threw = true;
    }

    tradereview::core::assert_true(threw, "assert_true should throw");
}

struct RegisterAssertionsTests {
    RegisterAssertionsTests()
    {
        tradereview::tests::register_test(
            "assert_equal accepts equal integers",
            test_assert_equal_accepts_equal_integers);
        tradereview::tests::register_test(
            "assert_true throws on false",
            test_assert_true_throws_on_false);
    }
};

const RegisterAssertionsTests register_assertions_tests;

} // namespace
