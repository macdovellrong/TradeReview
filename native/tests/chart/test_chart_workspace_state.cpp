#include "tradereview/chart/ChartWorkspaceState.h"
#include "tradereview/core/Assertions.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_workspace_state_clamps_chart_count_and_keeps_ids()
{
    tradereview::chart::ChartWorkspaceState state;

    tradereview::core::assert_equal(state.chart_count(), std::size_t{4}, "default chart count");
    auto ids = state.enabled_chart_ids();
    tradereview::core::assert_equal(ids.size(), std::size_t{4}, "default enabled id count");
    tradereview::core::assert_equal(ids[0], std::uint64_t{1}, "first chart id");
    tradereview::core::assert_equal(ids[3], std::uint64_t{4}, "fourth chart id");

    tradereview::core::assert_true(state.set_chart_count(0), "count below range changes state");
    tradereview::core::assert_equal(state.chart_count(), std::size_t{1}, "chart count clamps to one");
    ids = state.enabled_chart_ids();
    tradereview::core::assert_equal(ids.size(), std::size_t{1}, "one enabled chart after clamp");
    tradereview::core::assert_equal(ids[0], std::uint64_t{1}, "chart one remains enabled");

    tradereview::core::assert_true(state.set_chart_count(9), "count above range changes state");
    tradereview::core::assert_equal(state.chart_count(), std::size_t{4}, "chart count clamps to four");
}

void test_workspace_state_tracks_layout_mode()
{
    tradereview::chart::ChartWorkspaceState state;

    tradereview::core::assert_equal(
        static_cast<int>(state.layout_mode()),
        static_cast<int>(tradereview::chart::ChartLayoutMode::Tabs),
        "default layout is tabs");
    tradereview::core::assert_true(
        state.set_layout_mode(tradereview::chart::ChartLayoutMode::Grid2x2),
        "layout mode change is accepted");
    tradereview::core::assert_equal(
        static_cast<int>(state.layout_mode()),
        static_cast<int>(tradereview::chart::ChartLayoutMode::Grid2x2),
        "grid layout is stored");
    tradereview::core::assert_true(
        !state.set_layout_mode(tradereview::chart::ChartLayoutMode::Grid2x2),
        "same layout mode is a no-op");
}

void test_workspace_state_preserves_each_chart_period()
{
    tradereview::chart::ChartWorkspaceState state;

    tradereview::core::assert_true(state.set_chart_period(1, "1m"), "chart one period changes");
    tradereview::core::assert_true(state.set_chart_period(2, "5m"), "chart two period changes");
    tradereview::core::assert_true(state.set_chart_period(4, "1h"), "chart four period changes");
    tradereview::core::assert_equal(state.chart_period(1), std::string{"1m"}, "chart one period");
    tradereview::core::assert_equal(state.chart_period(2), std::string{"5m"}, "chart two period");
    tradereview::core::assert_equal(state.chart_period(4), std::string{"1h"}, "chart four period");

    state.set_chart_count(2);
    state.set_chart_count(4);
    tradereview::core::assert_equal(state.chart_period(4), std::string{"1h"}, "hidden chart period is preserved");
    tradereview::core::assert_true(!state.set_chart_period(99, "1D"), "unknown chart period is rejected");
}

void test_workspace_state_active_chart_follows_enabled_charts()
{
    tradereview::chart::ChartWorkspaceState state;

    tradereview::core::assert_true(state.set_active_chart_id(3), "active chart can change");
    tradereview::core::assert_equal(state.active_chart_id(), std::uint64_t{3}, "active chart id");
    tradereview::core::assert_true(state.set_chart_count(2), "chart count changes");
    tradereview::core::assert_equal(state.active_chart_id(), std::uint64_t{2}, "active chart clamps to enabled chart");
    tradereview::core::assert_true(!state.set_active_chart_id(4), "disabled active chart is rejected");
}

struct RegisterChartWorkspaceStateTests {
    RegisterChartWorkspaceStateTests()
    {
        tradereview::tests::register_test(
            "workspace state clamps chart count and keeps ids",
            test_workspace_state_clamps_chart_count_and_keeps_ids);
        tradereview::tests::register_test(
            "workspace state tracks layout mode",
            test_workspace_state_tracks_layout_mode);
        tradereview::tests::register_test(
            "workspace state preserves each chart period",
            test_workspace_state_preserves_each_chart_period);
        tradereview::tests::register_test(
            "workspace state active chart follows enabled charts",
            test_workspace_state_active_chart_follows_enabled_charts);
    }
};

const RegisterChartWorkspaceStateTests register_chart_workspace_state_tests;

} // namespace
