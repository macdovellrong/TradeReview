#include "tradereview/app/SessionState.h"
#include "tradereview/chart/ChartWorkspaceState.h"
#include "tradereview/core/Assertions.h"

#include <QSettings>
#include <QTemporaryDir>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace tradereview::tests {
void register_test(std::string name, std::function<void()> run);
}

namespace {

void test_session_state_round_trips_workspace_state()
{
    QTemporaryDir dir;
    tradereview::core::assert_true(dir.isValid(), "temporary settings directory");
    QSettings settings(dir.filePath("session.ini"), QSettings::IniFormat);

    tradereview::app::SessionState state;
    state.dataset_path = R"(\\10.0.0.23\code\gold\TradeReview\data\sample.duckdb)";
    state.center_time_ns = 1773579540000000000LL;
    state.chart_count = 3;
    state.layout_mode = tradereview::chart::ChartLayoutMode::Grid2x2;
    state.periods = {"1min", "5min", "1h", "1D"};

    tradereview::app::save_session_state(settings, state);
    const auto loaded = tradereview::app::load_session_state(settings);

    tradereview::core::assert_true(loaded.has_value(), "session state loads");
    tradereview::core::assert_equal(loaded->dataset_path, state.dataset_path, "dataset path");
    tradereview::core::assert_equal(loaded->center_time_ns, state.center_time_ns, "center time");
    tradereview::core::assert_equal(loaded->chart_count, state.chart_count, "chart count");
    tradereview::core::assert_equal(
        static_cast<int>(loaded->layout_mode),
        static_cast<int>(state.layout_mode),
        "layout mode");
    tradereview::core::assert_equal(loaded->periods.size(), state.periods.size(), "period count");
    tradereview::core::assert_equal(loaded->periods[2], std::string{"1h"}, "chart period");
}

void test_session_state_rejects_missing_or_invalid_state()
{
    QTemporaryDir dir;
    tradereview::core::assert_true(dir.isValid(), "temporary settings directory");
    QSettings settings(dir.filePath("session.ini"), QSettings::IniFormat);

    tradereview::core::assert_true(!tradereview::app::load_session_state(settings).has_value(), "missing state");

    settings.setValue("session/db_path", "x.duckdb");
    settings.setValue("session/center_time_ns", "not-a-number");
    settings.sync();

    tradereview::core::assert_true(!tradereview::app::load_session_state(settings).has_value(), "invalid center time");
}

struct RegisterSessionStateTests {
    RegisterSessionStateTests()
    {
        tradereview::tests::register_test(
            "session state round trips workspace state",
            test_session_state_round_trips_workspace_state);
        tradereview::tests::register_test(
            "session state rejects missing or invalid state",
            test_session_state_rejects_missing_or_invalid_state);
    }
};

const RegisterSessionStateTests register_session_state_tests;

} // namespace
