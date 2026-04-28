#include "tradereview/chart/PaneLayout.h"
#include "tradereview/chart/rendering/HistogramLayer.h"
#include "tradereview/chart/rendering/IndicatorLayer.h"

#include "tradereview/core/Assertions.h"
#include "tradereview/data/IndicatorColumns.h"

#include <functional>
#include <string>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

tradereview::data::CandleWindow sample_window()
{
    tradereview::data::CandleWindow window;
    window.generation = 3;
    window.timestamp_ns = {0, 60, 120, 180};
    window.open = {10.0, 11.0, 12.0, 11.0};
    window.high = {12.0, 13.0, 14.0, 13.0};
    window.low = {9.0, 10.0, 11.0, 10.0};
    window.close = {11.0, 12.0, 11.5, 12.5};
    window.volume = {1.0, 1.0, 1.0, 1.0};
    window.visible_range = {0, 180};
    window.loaded_range = {0, 180};
    window.indicators[std::string{tradereview::data::IndicatorColumns::EMA20}] = {10.5, 11.0, 11.7, 12.0};
    window.indicators[std::string{tradereview::data::IndicatorColumns::BB_Upper}] = {12.5, 13.2, 14.0, 13.5};
    window.indicators[std::string{tradereview::data::IndicatorColumns::BB_Lower}] = {8.5, 9.2, 10.0, 10.5};
    window.indicators[std::string{tradereview::data::IndicatorColumns::MACD}] = {-0.3, -0.1, 0.2, 0.4};
    window.indicators[std::string{tradereview::data::IndicatorColumns::MACD_Signal}] = {-0.2, -0.1, 0.1, 0.25};
    window.indicators[std::string{tradereview::data::IndicatorColumns::MACD_Hist}] = {-0.1, 0.0, 0.1, 0.15};
    window.indicators[std::string{tradereview::data::IndicatorColumns::RSI6}] = {42.0, 56.0, 68.0, 62.0};
    window.indicators[std::string{tradereview::data::IndicatorColumns::RSI12}] = {44.0, 54.0, 64.0, 60.0};
    window.indicators[std::string{tradereview::data::IndicatorColumns::RSI24}] = {46.0, 52.0, 60.0, 58.0};
    window.indicators[std::string{tradereview::data::IndicatorColumns::RSI}] = {45.0, 55.0, 65.0, 60.0};
    return window;
}

bool vertices_inside(
    const std::vector<tradereview::chart::rendering::IndicatorVertex>& vertices,
    tradereview::chart::PaneRect pane)
{
    for (const auto& vertex : vertices) {
        if (vertex.x < pane.left || vertex.x > pane.right || vertex.y < pane.bottom || vertex.y > pane.top) {
            return false;
        }
    }
    return true;
}

void test_price_indicator_geometry_uses_price_pane_scale()
{
    const auto window = sample_window();
    const auto layout = tradereview::chart::build_pane_layout(true);
    const auto geometry = tradereview::chart::rendering::build_price_indicator_geometry(
        window,
        {0.0, 3.0},
        layout.price,
        {
            std::string{tradereview::data::IndicatorColumns::EMA20},
            std::string{tradereview::data::IndicatorColumns::BB_Upper},
            std::string{tradereview::data::IndicatorColumns::BB_Lower},
        });

    tradereview::core::assert_equal(geometry.vertices.size(), std::size_t{18}, "three price indicator lines");
    tradereview::core::assert_true(vertices_inside(geometry.vertices, layout.price), "price indicators stay in price pane");
}

void test_indicator_panel_geometry_uses_independent_y_scale()
{
    const auto window = sample_window();
    const auto layout = tradereview::chart::build_pane_layout(true);
    const auto geometry = tradereview::chart::rendering::build_panel_indicator_geometry(
        window,
        {0.0, 3.0},
        layout.macd,
        {
            std::string{tradereview::data::IndicatorColumns::MACD},
            std::string{tradereview::data::IndicatorColumns::MACD_Signal},
        });

    tradereview::core::assert_equal(geometry.vertices.size(), std::size_t{12}, "two MACD indicator lines");
    tradereview::core::assert_true(vertices_inside(geometry.vertices, layout.macd), "MACD indicators stay in MACD pane");
}

void test_rsi_geometry_uses_rsi_pane()
{
    const auto window = sample_window();
    const auto layout = tradereview::chart::build_pane_layout(true);
    const auto geometry = tradereview::chart::rendering::build_panel_indicator_geometry(
        window,
        {0.0, 3.0},
        layout.rsi,
        {std::string{tradereview::data::IndicatorColumns::RSI}});

    tradereview::core::assert_equal(geometry.vertices.size(), std::size_t{6}, "RSI indicator line");
    tradereview::core::assert_true(vertices_inside(geometry.vertices, layout.rsi), "RSI indicator stays in RSI pane");
}

void test_multi_period_rsi_geometry_uses_rsi_pane()
{
    const auto window = sample_window();
    const auto layout = tradereview::chart::build_pane_layout(true);

    const auto geometry = tradereview::chart::rendering::build_panel_indicator_geometry(
        window,
        {0.0, 3.0},
        layout.rsi,
        {
            std::string{tradereview::data::IndicatorColumns::RSI6},
            std::string{tradereview::data::IndicatorColumns::RSI12},
            std::string{tradereview::data::IndicatorColumns::RSI24},
        });

    tradereview::core::assert_equal(geometry.vertices.size(), std::size_t{18}, "multi-period RSI lines");
    tradereview::core::assert_true(vertices_inside(geometry.vertices, layout.rsi), "multi-period RSI stays in RSI pane");
}

void test_histogram_geometry_splits_positive_and_negative_bars()
{
    const auto window = sample_window();
    const auto layout = tradereview::chart::build_pane_layout(true);
    const auto geometry = tradereview::chart::rendering::build_histogram_geometry(
        window,
        {0.0, 3.0},
        layout.macd,
        std::string{tradereview::data::IndicatorColumns::MACD_Hist});

    tradereview::core::assert_equal(geometry.positive_vertices.size(), std::size_t{12}, "two positive histogram bars");
    tradereview::core::assert_equal(geometry.negative_vertices.size(), std::size_t{6}, "one negative histogram bar");
}

struct RegisterIndicatorGeometryTests {
    RegisterIndicatorGeometryTests()
    {
        tradereview::tests::register_test(
            "price indicator geometry uses price pane scale",
            test_price_indicator_geometry_uses_price_pane_scale);
        tradereview::tests::register_test(
            "indicator panel geometry uses independent y scale",
            test_indicator_panel_geometry_uses_independent_y_scale);
        tradereview::tests::register_test(
            "RSI geometry uses RSI pane",
            test_rsi_geometry_uses_rsi_pane);
        tradereview::tests::register_test(
            "multi-period RSI geometry uses RSI pane",
            test_multi_period_rsi_geometry_uses_rsi_pane);
        tradereview::tests::register_test(
            "histogram geometry splits positive and negative bars",
            test_histogram_geometry_splits_positive_and_negative_bars);
    }
};

const RegisterIndicatorGeometryTests register_indicator_geometry_tests;

} // namespace
