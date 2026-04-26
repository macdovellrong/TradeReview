#include "tradereview/chart/PaneLayout.h"

#include "tradereview/core/Assertions.h"

#include <functional>
#include <string>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_pane_layout_allocates_price_macd_and_rsi_rectangles()
{
    const auto layout = tradereview::chart::build_pane_layout(true);

    tradereview::core::assert_true(layout.macd_visible, "MACD pane is visible");
    tradereview::core::assert_true(layout.rsi_visible, "RSI pane is visible");
    tradereview::core::assert_near(layout.price.left, -1.0, 0.0001, "price pane left");
    tradereview::core::assert_near(layout.price.right, 1.0, 0.0001, "price pane right");
    tradereview::core::assert_near(layout.price.top, 1.0, 0.0001, "price pane top");
    tradereview::core::assert_near(layout.rsi.bottom, -1.0, 0.0001, "RSI pane bottom");
    tradereview::core::assert_true(layout.price.bottom > layout.macd.top, "price pane sits above MACD pane");
    tradereview::core::assert_true(layout.macd.bottom > layout.rsi.top, "MACD pane sits above RSI pane");
    tradereview::core::assert_true(layout.price.height() > layout.macd.height(), "price pane is taller than MACD pane");
    tradereview::core::assert_true(layout.macd.height() > 0.0F, "MACD pane has height");
    tradereview::core::assert_true(layout.rsi.height() > 0.0F, "RSI pane has height");
}

void test_pane_layout_uses_full_price_rect_when_panels_hidden()
{
    const auto layout = tradereview::chart::build_pane_layout(false);

    tradereview::core::assert_true(!layout.macd_visible, "MACD pane is hidden");
    tradereview::core::assert_true(!layout.rsi_visible, "RSI pane is hidden");
    tradereview::core::assert_near(layout.price.left, -1.0, 0.0001, "full price pane left");
    tradereview::core::assert_near(layout.price.right, 1.0, 0.0001, "full price pane right");
    tradereview::core::assert_near(layout.price.bottom, -1.0, 0.0001, "full price pane bottom");
    tradereview::core::assert_near(layout.price.top, 1.0, 0.0001, "full price pane top");
}

struct RegisterPaneLayoutTests {
    RegisterPaneLayoutTests()
    {
        tradereview::tests::register_test(
            "pane layout allocates price MACD and RSI rectangles",
            test_pane_layout_allocates_price_macd_and_rsi_rectangles);
        tradereview::tests::register_test(
            "pane layout uses full price rect when panels hidden",
            test_pane_layout_uses_full_price_rect_when_panels_hidden);
    }
};

const RegisterPaneLayoutTests register_pane_layout_tests;

} // namespace
